from fastapi import APIRouter
from pydantic import BaseModel
import math

router = APIRouter()

ASCII91 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+,-./:;<=>?@[]^_`{|}~"

def make_alphabet(base: int) -> str:
    if base < 2: raise ValueError("base>=2")
    if base <= len(ASCII91): return ASCII91[:base]
    # extend printable range if base>91 (keep lite simple)
    raise ValueError("Lite alphabet supports up to base 91")

def bits_per_symbol(base: int) -> float:
    return math.log2(base)

def _bytes_to_int(data: bytes) -> int:
    n=0
    for b in data: n = (n<<8) | b
    return n

def _int_to_bytes(n: int, length: int) -> bytes:
    return n.to_bytes(length, "big")

def encode_baseN(data: bytes, alphabet: str) -> str:
    if not data: return ""
    base = len(alphabet)
    n = _bytes_to_int(data)
    digits = []
    while n>0:
        n, rem = divmod(n, base)
        digits.append(alphabet[rem])
    return ''.join(reversed(digits or [alphabet[0]]))

def decode_baseN(s: str, alphabet: str, out_len: int|None=None) -> bytes:
    base = len(alphabet)
    idx = {c:i for i,c in enumerate(alphabet)}
    n = 0
    for ch in s: n = n*base + idx[ch]
    if out_len is None:
        bits = math.ceil(len(s)*math.log2(base))
        out_len = max(1, (bits+7)//8)
    raw = _int_to_bytes(n, out_len)
    return raw.lstrip(b"\x00") or b"\x00"

class EncodeReq(BaseModel):
    base: int
    data_hex: str  # hex for transport

class EncodeRes(BaseModel):
    base: int
    bits_per_symbol: float
    encoded: str

@router.post("/encode", response_model=EncodeRes)
def encode(req: EncodeReq):
    data = bytes.fromhex(req.data_hex)
    alpha = make_alphabet(req.base)
    return EncodeRes(
        base=req.base,
        bits_per_symbol=bits_per_symbol(req.base),
        encoded=encode_baseN(data, alpha)
    )

class DecodeReq(BaseModel):
    base: int
    encoded: str

class DecodeRes(BaseModel):
    data_hex: str

@router.post("/decode", response_model=DecodeRes)
def decode(req: DecodeReq):
    alpha = make_alphabet(req.base)
    out = decode_baseN(req.encoded, alpha)
    return DecodeRes(data_hex=out.hex())