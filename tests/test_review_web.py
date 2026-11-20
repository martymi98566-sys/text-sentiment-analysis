"""HTTP boundaries, Unicode source integrity, and serialized GPU task scheduling."""
import http.client
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sentiment_analysis.review_web import JobManager, ReviewServer, validate_input


def finish_job(job):
    out = job["path"] / "analysis"
    out.mkdir()
    (out / "results.json").write_text(json.dumps([{"text":job["request"]["text"]}],ensure_ascii=False),encoding="utf-8")
    (out / "report.html").write_text("<!doctype html><title>test export</title>",encoding="utf-8")


class ReviewWebTests(unittest.TestCase):
    def test_validate_preserves_original_unicode_and_whitespace(self):
        text = "  Great acting 😀.\nThe plot is awful.  "
        self.assertEqual(validate_input({"text":text,"movie":" Test "}),{"text":text,"movie":"Test"})
        for value in [None,[],{}, {"text":" "},{"text":"纯中文评论"},{"text":"a"*12001},{"text":"Good\x00"},{"text":"Good","movie":[]}]:
            with self.assertRaises(ValueError):validate_input(value)

    def test_queue_serializes_jobs_and_refuses_overflow(self):
        started = threading.Event(); release = threading.Event(); calls=[]
        def run(job):
            calls.append(job["id"]);started.set();release.wait(3);finish_job(job)
        with tempfile.TemporaryDirectory() as tmp:
            manager=JobManager(tmp,runner=run,max_pending=1)
            try:
                first=manager.submit({"text":"Good acting."});self.assertTrue(started.wait(2))
                second=manager.submit({"text":"Bad plot."})
                with self.assertRaises(queue.Full):manager.submit({"text":"Nice music."})
                self.assertEqual(calls,[first]);self.assertEqual(manager.snapshot(second)["status"],"queued")
                release.set();manager.pending.join()
                self.assertEqual(calls,[first,second]);self.assertEqual(manager.snapshot(second)["status"],"completed")
            finally:release.set();manager.close()

    def test_worker_failure_does_not_block_next_review(self):
        count=[0]
        def run(job):
            count[0]+=1
            if count[0]==1:raise RuntimeError("fake GPU error")
            finish_job(job)
        with tempfile.TemporaryDirectory() as tmp:
            manager=JobManager(tmp,runner=run)
            try:
                first=manager.submit({"text":"First review."});manager.pending.join()
                second=manager.submit({"text":"Second review."});manager.pending.join()
                self.assertEqual(manager.snapshot(first)["status"],"failed")
                self.assertEqual(manager.snapshot(second)["status"],"completed")
            finally:manager.close()


class ReviewHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.manager=JobManager(self.temp.name,runner=finish_job)
        self.server=ReviewServer(("127.0.0.1",0),self.manager)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.port=self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.manager.close();self.temp.cleanup()

    def request(self,method,path,body=None,headers=None):
        client=http.client.HTTPConnection("127.0.0.1",self.port,timeout=5)
        client.request(method,path,body=body,headers=headers or {})
        response=client.getresponse();status=response.status;data=response.read();h=dict(response.getheaders());client.close()
        return status,data,h

    def test_static_serving_does_not_expose_project_files(self):
        status,data,headers=self.request("GET","/")
        self.assertEqual(status,200);self.assertIn(b'review-form',data)
        self.assertIn("frame-ancestors 'none'",headers["Content-Security-Policy"])
        for path in ["/.git/config","/../../pyproject.toml","/api/jobs/../../README.md","/data/raw/test.csv"]:
            self.assertEqual(self.request("GET",path)[0],404)

    def test_rejects_cross_origin_and_invalid_payloads(self):
        data=json.dumps({"text":"Good film."})
        status,_,_=self.request("POST","/api/jobs",data,{"Content-Type":"application/json","Origin":"https://outside.example"})
        self.assertEqual(status,403)
        self.assertEqual(self.request("GET","/api/health",headers={"Host":"outside.example"})[0],403)
        self.assertEqual(self.request("POST","/api/jobs","{",{"Content-Type":"application/json"})[0],400)
        self.assertEqual(self.request("POST","/api/jobs",data,{"Content-Type":"text/plain"})[0],415)

    def test_submit_poll_and_download_preserve_exact_review(self):
        text="Great acting 😀. <script>plain review text</script>\nBad plot."
        status,data,_=self.request("POST","/api/jobs",json.dumps({"text":text}),{"Content-Type":"application/json"})
        self.assertEqual(status,202);job_id=json.loads(data)["id"];self.manager.pending.join()
        status,data,_=self.request("GET",f"/api/jobs/{job_id}")
        self.assertEqual(json.loads(data)["status"],"completed")
        status,data,headers=self.request("GET",f"/api/jobs/{job_id}/json")
        self.assertEqual(json.loads(data)[0]["text"],text);self.assertIn("attachment",headers["Content-Disposition"])
        self.assertEqual(self.request("GET",f"/api/jobs/{'a'*32}")[0],404)


if __name__=="__main__":unittest.main()
