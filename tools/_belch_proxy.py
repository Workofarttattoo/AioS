from mitmproxy import http, options
from mitmproxy.tools.dump import DumpMaster
import asyncio
import http.client
import ssl
from urllib.parse import urlparse

class InterceptingProxy:
    """
    The core intercepting proxy for BelchStudio.
    Powered by mitmproxy.
    """
    def __init__(self, host="127.0.0.1", port=8080):
        self.host = host
        self.port = port
        self.master = None
        self.flow_queue = asyncio.Queue()

    def request(self, flow: http.HTTPFlow) -> None:
        """
        Called on every HTTP request.
        Puts the flow into a queue for the GUI to process.
        """
        self.flow_queue.put_nowait(flow)

    def response(self, flow: http.HTTPFlow) -> None:
        """
        Called on every HTTP response.
        Updates the flow in the queue.
        """
        pass

    async def run(self):
        """
        Starts the mitmproxy master.
        """
        opts = options.Options(listen_host=self.host, listen_port=self.port)
        self.master = DumpMaster(opts, with_termlog=False, with_dumper=False)
        self.master.addons.add(self)
        
        print(f"[BelchStudio] Intercepting proxy running on http://{self.host}:{self.port}")
        await self.master.run()

    def shutdown(self):
        if self.master:
            self.master.shutdown()

async def send_raw_request(raw_request: str) -> str:
    """
    Parses a raw HTTP request string, sends it, and returns the response.
    """
    try:
        headers_str, body = raw_request.strip().split('\r\n\r\n', 1)
        request_lines = headers_str.split('\r\n')
        method, path, _ = request_lines[0].split(' ')
        
        headers = {}
        host = None
        for line in request_lines[1:]:
            key, value = line.split(': ', 1)
            if key.lower() == 'host':
                host = value
            else:
                headers[key] = value
        
        if not host:
            return "Error: Host header is missing."

        is_ssl = False
        if ':' in host:
            host, port_str = host.split(':', 1)
            port = int(port_str)
            if port == 443:
                is_ssl = True
        else:
            is_ssl = path.lower().startswith('https')
            port = 443 if is_ssl else 80

        if is_ssl:
            context = ssl._create_unverified_context()
            conn = http.client.HTTPSConnection(host, port, context=context, timeout=10)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=10)

        conn.request(method, path, body, headers)
        
        response = conn.getresponse()
        
        response_status = f"HTTP/{response.version / 10.0} {response.status} {response.reason}\r\n"
        response_headers = "".join([f"{key}: {value}\r\n" for key, value in response.getheaders()])
        response_body = response.read().decode('utf-8', errors='ignore')
        
        conn.close()
        
        return f"{response_status}{response_headers}\r\n{response_body}"

    except Exception as e:
        return f"Error sending request: {e}"
