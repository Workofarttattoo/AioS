from fastapi import APIRouter
from pydantic import BaseModel
import math
router=APIRouter()
ASCII91="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+,-./:;<=>?@[]^_`{|}~"
def make_alphabet(base:int)->str:
    if base<2: raise ValueError("base>=2")
    if base<=len(ASCII91): return ASCII91[:base]
    raise ValueError("Lite alphabet supports up to base 91")
def _bytes_to_int(b:bytes)->int:
    n=0
    for x in b: n=(n<<8)|x
    return n
def encode_baseN(b:bytes, alpha:str)->str:
    if not b: return ""
    base=len(alpha); n=_bytes_to_int(b); out=[]
    while n>0: n, r = divmod(n, base); out.append(alpha[r])
    return "".join(reversed(out or [alpha[0]]))
def decode_baseN(s:str, alpha:str, out_len:int|None=None)->bytes:
    base=len(alpha); idx={c:i for i,c in enumerate(alpha)}; n=0
    for ch in s: n=n*base+idx[ch]
    if out_len is None:
        bits=math.ceil(len(s)*math.log2(base)); out_len=max(1,(bits+7)//8)
    raw=n.to_bytes(out_len,"big")
    return raw.lstrip(b"\x00") or b"\x00"
class EncodeReq(BaseModel): base:int; data_hex:str
class EncodeRes(BaseModel): base:int; bits_per_symbol:float; encoded:str
@router.post("/encode", response_model=EncodeRes)
def enc(req:EncodeReq):
    alpha=make_alphabet(req.base); data=bytes.fromhex(req.data_hex)
    return EncodeRes(base=req.base, bits_per_symbol=math.log2(req.base), encoded=encode_baseN(data, alpha))
class DecodeReq(BaseModel): base:int; encoded:str
class DecodeRes(BaseModel): data_hex:str
@router.post("/decode", response_model=DecodeRes)
def dec(req:DecodeReq):
    alpha=make_alphabet(req.base); out=decode_baseN(req.encoded, alpha)
    return DecodeRes(data_hex=out.hex())
