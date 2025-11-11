# OS fingerprinting probes and signatures for AuroraScan
# This data is a simplified representation of a TCP/IP stack fingerprint database.

# Each probe is a specially crafted TCP packet designed to elicit a unique
# response from different operating systems' TCP/IP stacks.
OS_PROBES = [
    # Sequence generation (SEQ) probe
    {
        "name": "SEQ",
        "protocol": "TCP",
        "options": [("WScale", 10), ("NOP",), ("MSS", 1460), ("Timestamp", "0,0"), ("SACK",)],
        "flags": "S"
    },
    # TCP options (T1) probe
    {
        "name": "T1",
        "protocol": "TCP",
        "options": [],
        "flags": "S"
    },
    # Window scan (WIN) probe
    {
        "name": "WIN",
        "protocol": "TCP",
        "options": [("WScale", 15)],
        "flags": "S"
    }
]

# Signatures are based on the responses to the probes. This is a highly
# simplified example. A real database would have hundreds of signatures.
OS_SIGNATURES = [
    {
        "name": "Linux (Kernel 2.6+)",
        "matches": {
            "SEQ": {"TCPWindow": 5840, "WScale": 7},
            "T1": {"TCPOptions": "MSS,SACK,Timestamp,NOP,WScale"},
            "WIN": {"TCPWindow": ">=16384"}
        }
    },
    {
        "name": "Windows 10",
        "matches": {
            "SEQ": {"TCPWindow": 65535, "WScale": 8},
            "T1": {"TCPOptions": "MSS,NOP,WScale,SACK,Timestamp"},
            "WIN": {"TCPWindow": 65535}
        }
    },
    {
        "name": "FreeBSD",
        "matches": {
            "SEQ": {"TCPWindow": 65535, "WScale": 0},
            "T1": {"TCPOptions": "MSS,NOP,WScale"},
            "WIN": {"TCPWindow": 65535}
        }
    }
]
