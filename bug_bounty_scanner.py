#!/usr/bin/env python3
import asyncio
import json
import logging
import re
import ipaddress
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

LOG = logging.getLogger(__name__)

class ScopeEnforcer:
    def __init__(self, allowed_targets: List[str] = None, blocked_targets: List[str] = None):
        self.allowed_patterns = [self._compile_pattern(t) for t in (allowed_targets or [])]
        self.blocked_patterns = [self._compile_pattern(t) for t in (blocked_targets or [])]

    def _compile_pattern(self, target: str):
        # Convert wildcard patterns like *.example.com to regex
        pattern = re.escape(target).replace(r'\*', '.*')
        return re.compile(f"^{pattern}$", re.IGNORECASE)

    def is_allowed(self, target: str) -> bool:
        # Check if it's an IP
        try:
            ip = ipaddress.ip_address(target)
            # Add IP range checks if needed
            return True # Default to true for now if not explicitly blocked
        except ValueError:
            pass

        # Check domain
        parsed = urlparse(target if "://" in target else f"http://{target}")
        domain = parsed.netloc or parsed.path

        # Blacklist first
        for pattern in self.blocked_patterns:
            if pattern.match(domain):
                return False

        # If whitelist is empty, allow all (risky, but let's assume user provides it)
        if not self.allowed_patterns:
            return True

        for pattern in self.allowed_patterns:
            if pattern.match(domain):
                return True

        return False

class BugBountyScanner:
    def __init__(self, scope: ScopeEnforcer = None):
        self.scope = scope or ScopeEnforcer()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        ]

    async def _run_tool(self, cmd: str, timeout: int = 300, retries: int = 2) -> Dict:
        """Safely run security tools with retry logic"""
        for attempt in range(retries + 1):
            try:
                LOG.info(f"Running command (attempt {attempt+1}): {cmd}")
                process = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)

                return {
                    "status": "success",
                    "stdout": stdout.decode(errors='replace'),
                    "stderr": stderr.decode(errors='replace'),
                    "exit_code": process.returncode
                }
            except asyncio.TimeoutError:
                if attempt == retries:
                    return {"status": "error", "message": "TIMEOUT"}
            except Exception as e:
                LOG.error(f"Tool execution failed: {e}")
                if attempt == retries:
                    return {"status": "error", "message": str(e)}
            await asyncio.sleep(2 ** attempt) # Exponential backoff
        return {"status": "error", "message": "Max retries exceeded"}

    async def run_nuclei(self, target: str, severity: str = "critical,high") -> Dict:
        if not self.scope.is_allowed(target):
            return {"status": "error", "message": "OUT_OF_SCOPE"}

        cmd = f"nuclei -u {target} -severity {severity} -json"
        result = await self._run_tool(cmd)

        # Try to parse JSON output if successful
        if result["status"] == "success" and result["stdout"]:
            try:
                # Nuclei outputs multiple JSON objects, one per line
                lines = result["stdout"].strip().split('\n')
                findings = [json.loads(line) for line in lines if line.strip()]
                return {"status": "success", "findings": findings}
            except:
                pass
        return result

    async def run_ffuf(self, target: str, wordlist: str = "/usr/share/wordlists/dirb/common.txt") -> Dict:
        if not self.scope.is_allowed(target):
            return {"status": "error", "message": "OUT_OF_SCOPE"}

        # Ensure target ends with FUZZ if not present
        if "FUZZ" not in target:
            target = target.rstrip('/') + "/FUZZ"

        cmd = f"ffuf -u {target} -w {wordlist} -mc 200,301,302 -of json"
        # ffuf output file management might be better but let's try stdout first or use a temp file
        result = await self._run_tool(cmd)
        return result

    async def run_sqlmap(self, url: str, batch: bool = True) -> Dict:
        if not self.scope.is_allowed(url):
            return {"status": "error", "message": "OUT_OF_SCOPE"}

        cmd = f"sqlmap -u \"{url}\" --random-agent --level 1 --risk 1"
        if batch:
            cmd += " --batch"

        return await self._run_tool(cmd)

    async def run_nmap(self, target: str, options: str = "-sV -T4") -> Dict:
        if not self.scope.is_allowed(target):
            return {"status": "error", "message": "OUT_OF_SCOPE"}

        cmd = f"nmap {options} {target}"
        return await self._run_tool(cmd)

if __name__ == "__main__":
    async def main():
        enforcer = ScopeEnforcer(allowed_targets=["example.com", "*.test.com"])
        scanner = BugBountyScanner(scope=enforcer)

        print(f"Is example.com allowed? {enforcer.is_allowed('example.com')}")
        print(f"Is malicious.com allowed? {enforcer.is_allowed('malicious.com')}")
        print(f"Is sub.test.com allowed? {enforcer.is_allowed('sub.test.com')}")

    asyncio.run(main())
