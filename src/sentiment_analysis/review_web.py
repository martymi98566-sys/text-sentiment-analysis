"""Loopback-only interactive review app, using the unchanged single-review runner."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from sentiment_analysis.paths import ROOT

MAX_CHARACTERS = 12000
MAX_BODY = 65536
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
                "/icon.svg": ("icon.svg", "image/svg+xml")}


def validate_input(payload):
    if not isinstance(payload, dict):
        raise ValueError("请提交评论和片名。")
    text = payload.get("text")
    movie = payload.get("movie", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请先输入一条评论。")
    if len(text) > MAX_CHARACTERS:
        raise ValueError(f"单条评论最多 {MAX_CHARACTERS:,} 个字符，请缩短后重试。")
    if not re.search(r"[A-Za-z]", text):
        raise ValueError("当前版本支持英文评论，请输入英文原文。")
    if "\x00" in text:
        raise ValueError("评论含有无法处理的空字符，请删除后重试。")
    if not isinstance(movie, str) or len(movie) > 160 or "\x00" in movie:
        raise ValueError("片名最多160个字符。")
    return {"text": text, "movie": movie.strip() or "未提供片名"}


class JobManager:
    """One GPU job at a time; each runs in its own process to isolate model globals."""

    def __init__(self, directory, runner=None, max_pending=2):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runner = runner or self._run_model
        self.jobs = OrderedDict()
        self.lock = threading.RLock()
        self.pending = queue.Queue(maxsize=max_pending)
        self.stopping = threading.Event()
        self.process = None
        self.worker = threading.Thread(target=self._work, name="review-model-worker", daemon=True)
        self.worker.start()

    def submit(self, payload):
        payload = validate_input(payload)
        with self.lock:
            if self.stopping.is_set() or self.pending.full():
                raise queue.Full()
            job_id = uuid.uuid4().hex
            out = self.directory / job_id
            out.mkdir()
            (out / "review.txt").write_text(payload["text"], encoding="utf-8")
            job = {"id": job_id, "status": "queued", "created": time.time(), "started": None,
                   "finished": None, "error": None, "request": payload, "path": out}
            self.jobs[job_id] = job
            (out / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.pending.put_nowait(job_id)
            while len(self.jobs) > 100:
                old = next((k for k,v in self.jobs.items() if v["status"] in {"completed", "failed"}), None)
                if old is None:
                    break
                del self.jobs[old]
            return job_id

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            queued = [k for k,v in self.jobs.items() if v["status"] == "queued"]
            start = job["started"] or job["created"]
            return {"id": job_id, "status": job["status"], "error": job["error"],
                    "elapsed_seconds": max(0, int((job["finished"] or time.time()) - start)),
                    "queue_position": queued.index(job_id) + 1 if job_id in queued else 0,
                    "request": dict(job["request"])}

    def file(self, job_id, kind):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job["status"] != "completed":
                return None
            filenames = {"result": "results.json", "report": "report.html"}
            name = filenames.get(kind)
            return job["path"] / "analysis" / name if name else None

    def _work(self):
        while not self.stopping.is_set():
            try:
                job_id = self.pending.get(timeout=.2)
            except queue.Empty:
                continue
            with self.lock:
                job = self.jobs[job_id]
                job.update(status="running", started=time.time())
            try:
                self.runner(job)
                result_path = job["path"] / "analysis" / "results.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(result, list) or len(result) != 1 or result[0]["text"] != job["request"]["text"]:
                    raise ValueError("Model output did not match submitted review")
                with self.lock:
                    job.update(status="completed", finished=time.time())
            except Exception as exc:
                with self.lock:
                    job.update(status="failed", finished=time.time(),
                               error="本次分析未完成，请重试。详细日志已保存在本地任务目录。")
                (job["path"] / "error.txt").write_text(str(exc), encoding="utf-8")
            finally:
                self.pending.task_done()

    def _run_model(self, job):
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        command = [sys.executable, "-B", "-u", str(ROOT / "scripts/analyze_single_review_emotions.py"),
                   "--text-file", str(job["path"] / "review.txt"), "--movie", job["request"]["movie"],
                   "--output", str(job["path"] / "analysis")]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (job["path"] / "run.log").open("w", encoding="utf-8") as log:
            with self.lock:
                if self.stopping.is_set():
                    raise RuntimeError("Server stopped")
                self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                stderr=subprocess.STDOUT, creationflags=flags)
                proc = self.process
            try:
                code = proc.wait(timeout=900)
                if code:
                    raise RuntimeError(f"Review analysis exited with code {code}")
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError("Review analysis timed out") from None
            finally:
                with self.lock:
                    self.process = None

    def close(self):
        self.stopping.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        self.worker.join(timeout=5)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager, web_directory=None):
        self.manager = manager
        self.web_directory = Path(web_directory or ROOT / "web/review")
        config = json.loads((ROOT / "configs/review_application.json").read_text(encoding="utf-8"))
        self.categories = [{"id": c["id"], "name": c["name"]} for c in config["categories"]]
        super().__init__(address, ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "ReviewLocal/1.0"

    def log_message(self, format, *args):
        # Never log user text, query data, or raw request bodies.
        print(f"[review-web] {self.command} {self.path.split('?')[0]} {args[1] if len(args)>1 else ''}", flush=True)

    def _allowed_host(self):
        port = self.server.server_address[1]
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _send(self, body, mime="application/json; charset=utf-8", status=200, filename=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status, message):
        self._send({"error": message}, status=status)

    def do_GET(self):
        if not self._allowed_host():
            return self._error(403, "仅允许本机访问。")
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self._send({"application": "movie-review-local", "status": "ready", "max_characters": MAX_CHARACTERS,
                               "language": "en", "categories": self.server.categories})
        if path in STATIC_FILES:
            name, mime = STATIC_FILES[path]
            file = self.server.web_directory / name
            if not file.is_file():
                return self._error(404, "页面文件不存在。")
            return self._send(file.read_bytes(), mime)
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(result|json|report))?", path)
        if not match:
            return self._error(404, "没有找到该页面。")
        job_id, kind = match.groups()
        snapshot = self.server.manager.snapshot(job_id)
        if snapshot is None:
            return self._error(404, "没有找到该任务，服务重启后请重新提交。")
        if kind is None:
            return self._send(snapshot)
        file = self.server.manager.file(job_id, "report" if kind == "report" else "result")
        if file is None or not file.is_file():
            return self._error(409, "结果尚未生成。")
        if kind == "result":
            row = json.loads(file.read_text(encoding="utf-8"))[0]
            return self._send({"result": row, "categories": self.server.categories})
        return self._send(file.read_bytes(), "text/html; charset=utf-8" if kind == "report" else "application/json; charset=utf-8",
                          filename=f"review-{job_id}.{ 'html' if kind=='report' else 'json' }")

    def do_POST(self):
        if not self._allowed_host():
            return self._error(403, "仅允许本机访问。")
        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        if (origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}) or self.headers.get("Sec-Fetch-Site") == "cross-site":
            return self._error(403, "请从本机分析页面提交。")
        if urlsplit(self.path).path != "/api/jobs":
            return self._error(404, "没有找到该接口。")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._error(415, "请以JSON格式提交评论。")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error(400, "请求长度无效。")
        if not 0 < length <= MAX_BODY:
            return self._error(413, "提交内容过大或为空。")
        try:
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            validated = validate_input(payload)
        except (ValueError, UnicodeError, TimeoutError):
            return self._error(400, "请检查评论：需要非空英文原文，最多12000字符，片名最多160字符。")
        try:
            job_id = self.server.manager.submit(validated)
        except queue.Full:
            return self._error(429, "当前等待任务较多，请稍后重试。")
        except OSError:
            return self._error(500, "无法保存本地任务，请检查磁盘空间。")
        self._send({"id": job_id, "status": "queued"}, status=202)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("端口应在1024到65535之间。")
    if not (ROOT / "scripts/analyze_single_review_emotions.py").is_file():
        parser.error("未找到单条评论分析入口。")
    manager = JobManager(ROOT / "data/review_web/jobs")
    try:
        server = ReviewServer(("127.0.0.1", args.port), manager)
    except OSError:
        manager.close()
        parser.error(f"端口{args.port}不可用，请使用 --port 选择其他端口。")
    print(f"影评情绪分析已启动：http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == "__main__":
    main()
