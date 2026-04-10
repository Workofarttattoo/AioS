import unittest
import subprocess
import time
import json
import urllib.request
import urllib.error
import os
import signal

class TestToolLauncherSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = "test-token-123"
        env = os.environ.copy()
        env["AIOS_LAUNCHER_TOKEN"] = cls.token
        cls.server_proc = subprocess.Popen(
            ["python3", "web/tool_launcher.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Wait for server to start

    @classmethod
    def tearDownClass(cls):
        cls.server_proc.terminate()
        cls.server_proc.wait()

    def make_request(self, path, method="GET", headers=None, data=None):
        url = f"http://localhost:7777{path}"
        req = urllib.request.Request(url, method=method)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)

        if data:
            body = json.dumps(data).encode('utf-8')
            req.data = body
            req.add_header('Content-Type', 'application/json')

        try:
            with urllib.request.urlopen(req) as response:
                resp_data = response.read()
                data = None
                if resp_data:
                    data = json.loads(resp_data.decode('utf-8'))
                return response.getcode(), data, response.info()
        except urllib.error.HTTPError as e:
            return e.code, None, e.info()

    def test_health_public(self):
        code, data, _ = self.make_request("/health")
        self.assertEqual(code, 200)
        self.assertEqual(data["status"], "ok")

    def test_tools_unauthorized(self):
        code, _, _ = self.make_request("/tools")
        self.assertEqual(code, 401)

    def test_tools_authorized(self):
        code, data, _ = self.make_request("/tools", headers={"X-API-Key": self.token})
        self.assertEqual(code, 200)
        self.assertIn("tools", data)

    def test_launch_unauthorized(self):
        code, _, _ = self.make_request("/launch", method="POST", data={"tool": "aurorascan"})
        self.assertEqual(code, 401)

    def test_launch_authorized(self):
        # We don't want to actually launch a tool if possible, but let's see.
        # aurorascan exists in the list.
        code, data, _ = self.make_request("/launch", method="POST",
                                         headers={"X-API-Key": self.token},
                                         data={"tool": "aurorascan"})
        self.assertEqual(code, 200)
        self.assertTrue(data["success"])
        # Clean up
        if "tool" in data:
            self.make_request("/stop", method="POST",
                              headers={"X-API-Key": self.token},
                              data={"tool": data["tool"]})

    def test_invalid_token(self):
        code, _, _ = self.make_request("/tools", headers={"X-API-Key": "wrong-token"})
        self.assertEqual(code, 401)

    def test_options_preflight(self):
        code, _, info = self.make_request("/launch", method="OPTIONS",
                                         headers={"Origin": "http://example.com"})
        self.assertEqual(code, 200)
        self.assertEqual(info.get("Access-Control-Allow-Origin"), "http://example.com")
        self.assertIn("X-API-Key", info.get("Access-Control-Allow-Headers"))

if __name__ == "__main__":
    unittest.main()
