# Service probes and signatures for AuroraScan
# This file contains the data used for advanced service and version detection.

# The format is a dictionary where keys are port numbers and values are lists of probes.
# Each probe is a dictionary with 'payload' (bytes to send) and 'matches' (list of regexes to check).
# The first regex that matches the response determines the service.

import re

PROBE_DATABASE = {
    # Probes for HTTP
    80: [
        {
            "payload": b"GET / HTTP/1.0\r\n\r\n",
            "matches": [
                {"regex": re.compile(b"Server:.*Apache/([0-9\\.]+)", re.IGNORECASE), "service": "Apache httpd"},
                {"regex": re.compile(b"Server:.*nginx/([0-9\\.]+)", re.IGNORECASE), "service": "Nginx"},
                {"regex": re.compile(b"Server:.*Microsoft-IIS/([0-9\\.]+)", re.IGNORECASE), "service": "Microsoft IIS"},
                {"regex": re.compile(b"HTTP/1\\.[01] 200 OK", re.IGNORECASE), "service": "HTTP"},
            ]
        }
    ],
    # Probes for SSH
    22: [
        {
            "payload": b"SSH-2.0-AuroraScan\r\n",
            "matches": [
                {"regex": re.compile(b"SSH-2.0-OpenSSH_([\\w\\.]+)", re.IGNORECASE), "service": "OpenSSH"},
                {"regex": re.compile(b"SSH-2.0-(\\w+)", re.IGNORECASE), "service": "SSH"},
            ]
        }
    ],
    # Probes for FTP
    21: [
        {
            "payload": b"FEAT\r\n",
            "matches": [
                {"regex": re.compile(b"220.*vsFTPd (\\d.\\d.\\d)", re.IGNORECASE), "service": "vsFTPd"},
                {"regex": re.compile(b"220.*ProFTPD (\\d.\\d.\\d)", re.IGNORECASE), "service": "ProFTPD"},
                {"regex": re.compile(b"220.*Microsoft FTP Service", re.IGNORECASE), "service": "Microsoft FTP"},
            ]
        }
    ],
    # Default generic probe
    "default": [
        {
            "payload": b"\r\n\r\n",
            "matches": []
        }
    ]
}
