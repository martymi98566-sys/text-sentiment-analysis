"""Install or verify the exact trained and pinned public model assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_target(relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe model asset path')
    target = (ROOT / relative).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError('Model path escaped repository')
    return target


def verify(files, root=ROOT):
    errors = []
    for relative, spec in files.items():
        target = root / relative
        expected = spec['sha256'] if isinstance(spec, dict) else spec
        if not target.is_file(): errors.append(relative + ': missing')
        elif digest(target) != expected: errors.append(relative + ': SHA256 mismatch')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--trained-archive', type=Path, help='Use a manually downloaded release ZIP')
    parser.add_argument('--link-local', type=Path, help='Reuse exact assets from a local checkout; hard link or copy without changing bytes')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/model_assets.json').read_text(encoding='utf-8'))
    runtime = json.loads((ROOT / 'configs/runtime_integrity.json').read_text(encoding='utf-8'))
    files = dict(manifest['trained_files'])
    for model in manifest['pretrained']:
        files.update({model['directory'] + '/' + name: spec for name, spec in model['files'].items()})
    if args.link_local:
        source = args.link_local.resolve()
        errors = verify(files, source)
        if errors: raise RuntimeError('Local source failed verification:\n' + '\n'.join(errors))
        for relative, spec in files.items():
            target = safe_target(relative)
            if target.exists():
                if digest(target) != spec['sha256']: raise RuntimeError('Refusing to overwrite mismatched asset: ' + relative)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try: os.link(source / relative, target)
            except OSError: shutil.copyfile(source / relative, target)
    elif not args.verify_only:
        missing = verify(manifest['trained_files'])
        if missing:
            archive = args.trained_archive
            if archive is None:
                archive = ROOT / 'data/downloads' / manifest['trained_archive']
                archive.parent.mkdir(parents=True, exist_ok=True)
                if not archive.exists() or digest(archive) != manifest['trained_archive_sha256']:
                    url = ('https://github.com/' + manifest['repository'] + '/releases/download/'
                           + manifest['release_tag'] + '/' + manifest['trained_archive'])
                    print('Downloading trained model pack:', url, flush=True)
                    partial = archive.with_suffix('.partial')
                    urllib.request.urlretrieve(url, partial)
                    if digest(partial) != manifest['trained_archive_sha256']:
                        raise RuntimeError('Downloaded archive SHA256 mismatch')
                    partial.replace(archive)
            if digest(archive) != manifest['trained_archive_sha256']:
                raise RuntimeError('Trained archive SHA256 mismatch')
            with zipfile.ZipFile(archive) as stream:
                if set(stream.namelist()) != set(manifest['trained_files']):
                    raise RuntimeError('Unexpected ZIP entries')
                for relative, spec in manifest['trained_files'].items():
                    target = safe_target(relative)
                    if target.exists():
                        if digest(target) != spec['sha256']: raise RuntimeError('Refusing to overwrite mismatched asset: ' + relative)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_suffix(target.suffix + '.partial')
                    with stream.open(relative) as source, partial.open('wb') as output:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                    if digest(partial) != spec['sha256']: raise RuntimeError('Extracted file SHA256 mismatch: ' + relative)
                    partial.replace(target)
        from huggingface_hub import snapshot_download
        for model in manifest['pretrained']:
            expected = {model['directory'] + '/' + k: v for k,v in model['files'].items()}
            if verify(expected):
                print('Downloading pinned pretrained model:', model['repo_id'], model['revision'], flush=True)
                snapshot_download(repo_id=model['repo_id'], revision=model['revision'],
                                  local_dir=ROOT / model['directory'], allow_patterns=list(model['files']))
                errors = verify(expected)
                if errors: raise RuntimeError('Pretrained model mismatch:\n' + '\n'.join(errors))
    errors = verify(runtime['protected_sha256'])
    print(json.dumps({'passed': not errors, 'checked_files': len(runtime['protected_sha256']), 'errors': errors}, ensure_ascii=False))
    if errors: return 1
    if not args.verify_only:
        import torch
        print(json.dumps({'cuda_available': torch.cuda.is_available(), 'bf16_supported': torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                          'note': 'The current inference pipeline requires NVIDIA CUDA with BF16 support.'}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
