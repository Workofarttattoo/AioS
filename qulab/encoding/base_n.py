"""
High-radix base-N encoding and decoding.

Implements efficient encoding systems for bases 2 through 1024 with
support for custom alphabets, error detection, and packing optimization.

References:
- Knuth, D. E. (1997). The art of computer programming, volume 2.
- Salomon, D. (2007). Data compression: the complete reference.
"""

from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import string
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class EncodingResult(BaseModel):
    """Result of base-N encoding operation."""
    
    encoded_data: str = Field(..., description="Encoded string")
    original_length: int = Field(..., description="Original data length in bytes")
    encoded_length: int = Field(..., description="Encoded string length")
    compression_ratio: float = Field(..., description="Compression ratio (original/encoded)")
    base: int = Field(..., description="Encoding base")
    alphabet: str = Field(..., description="Alphabet used for encoding")
    padding: int = Field(0, description="Number of padding characters added")
    
    class Config:
        arbitrary_types_allowed = True


class BaseNEncoder:
    """
    High-radix base-N encoder.
    
    Supports encoding of binary data into arbitrary bases (2-1024) with
    custom alphabets and error detection capabilities.
    """
    
    # Standard alphabets for different bases
    STANDARD_ALPHABETS = {
        2: "01",
        8: "01234567",
        10: "0123456789",
        16: "0123456789abcdef",
        32: "0123456789abcdefghijklmnopqrstuv",
        36: "0123456789abcdefghijklmnopqrstuvwxyz",
        58: "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz",  # Bitcoin alphabet
        64: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
        85: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~",
        91: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,./:;<=>?@[]^_`{|}~\"",
        94: "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
        128: "".join(chr(i) for i in range(128)),  # ASCII
        256: "".join(chr(i) for i in range(256)),  # Extended ASCII
    }
    
    def __init__(self, base: int, alphabet: Optional[str] = None, 
                 error_detection: bool = False):
        """
        Initialize base-N encoder.
        
        Args:
            base: Encoding base (2-1024)
            alphabet: Custom alphabet (if None, uses standard alphabet)
            error_detection: Enable error detection (checksum)
        """
        if not 2 <= base <= 1024:
            raise ValueError("Base must be between 2 and 1024")
        
        self.base = base
        self.error_detection = error_detection
        
        # Set alphabet
        if alphabet is None:
            if base in self.STANDARD_ALPHABETS:
                self.alphabet = self.STANDARD_ALPHABETS[base]
            else:
                # Generate default alphabet
                self.alphabet = self._generate_default_alphabet(base)
        else:
            if len(alphabet) != base:
                raise ValueError(f"Alphabet length ({len(alphabet)}) must equal base ({base})")
            self.alphabet = alphabet
        
        # Create mapping dictionaries
        self.char_to_value = {char: i for i, char in enumerate(self.alphabet)}
        self.value_to_char = {i: char for i, char in enumerate(self.alphabet)}
        
        # Calculate efficiency metrics
        self.bits_per_symbol = np.log2(base)
        self.symbols_per_byte = 8 / self.bits_per_symbol
    
    def _generate_default_alphabet(self, base: int) -> str:
        """Generate default alphabet for given base."""
        if base <= 10:
            return "0123456789"[:base]
        elif base <= 36:
            return "0123456789abcdefghijklmnopqrstuvwxyz"[:base]
        elif base <= 62:
            return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[:base]
        else:
            # For larger bases, use extended character set
            alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            # Add more characters as needed
            extended_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            alphabet += extended_chars
            
            # If still not enough, use Unicode characters
            if len(alphabet) < base:
                for i in range(0x00A0, 0x0100):  # Latin-1 Supplement
                    if len(alphabet) >= base:
                        break
                    alphabet += chr(i)
            
            return alphabet[:base]
    
    def encode(self, data: Union[bytes, str, List[int]]) -> EncodingResult:
        """
        Encode data using base-N encoding.
        
        Args:
            data: Data to encode (bytes, string, or list of integers)
            
        Returns:
            EncodingResult with encoded data and metadata
        """
        # Convert input to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, list):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        original_length = len(data_bytes)
        
        # Add error detection if enabled
        if self.error_detection:
            checksum = self._calculate_checksum(data_bytes)
            data_bytes += checksum.to_bytes(4, 'big')
        
        # Convert to base-N
        encoded_chars = self._bytes_to_base_n(data_bytes)
        encoded_data = ''.join(encoded_chars)
        
        # Calculate padding
        padding = self._calculate_padding(original_length)
        
        return EncodingResult(
            encoded_data=encoded_data,
            original_length=original_length,
            encoded_length=len(encoded_data),
            compression_ratio=original_length / len(encoded_data) if len(encoded_data) > 0 else 0,
            base=self.base,
            alphabet=self.alphabet,
            padding=padding
        )
    
    def _bytes_to_base_n(self, data_bytes: bytes) -> List[str]:
        """Convert bytes to base-N representation."""
        # Convert bytes to integer
        data_int = int.from_bytes(data_bytes, 'big')
        
        # Convert to base-N
        if data_int == 0:
            return [self.value_to_char[0]]
        
        result = []
        while data_int > 0:
            result.append(self.value_to_char[data_int % self.base])
            data_int //= self.base
        
        return result[::-1]  # Reverse to get most significant digit first
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate simple checksum for error detection."""
        return sum(data) & 0xFFFFFFFF
    
    def _calculate_padding(self, original_length: int) -> int:
        """Calculate padding needed for optimal encoding."""
        # For base-N encoding, we need to ensure the data length is compatible
        # with the base. This is a simplified calculation.
        bits_needed = original_length * 8
        symbols_needed = np.ceil(bits_needed / self.bits_per_symbol)
        padding_bits = int(symbols_needed * self.bits_per_symbol - bits_needed)
        return padding_bits // 8
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Get encoding efficiency metrics."""
        return {
            "bits_per_symbol": self.bits_per_symbol,
            "symbols_per_byte": self.symbols_per_byte,
            "theoretical_compression_ratio": 8 / self.bits_per_symbol,
            "alphabet_size": len(self.alphabet),
            "base": self.base
        }
    
    def validate_alphabet_safety(self) -> Dict[str, bool]:
        """Validate alphabet for common safety issues."""
        safety_checks = {
            "no_ambiguous_chars": self._check_ambiguous_chars(),
            "no_control_chars": self._check_control_chars(),
            "no_whitespace": self._check_whitespace(),
            "no_special_chars": self._check_special_chars(),
            "case_insensitive_safe": self._check_case_insensitive(),
        }
        return safety_checks
    
    def _check_ambiguous_chars(self) -> bool:
        """Check for visually ambiguous characters."""
        ambiguous_pairs = [
            ('0', 'O'), ('1', 'l'), ('1', 'I'), ('5', 'S'),
            ('6', 'G'), ('8', 'B'), ('9', 'g'), ('2', 'Z')
        ]
        
        for char1, char2 in ambiguous_pairs:
            if char1 in self.alphabet and char2 in self.alphabet:
                return False
        return True
    
    def _check_control_chars(self) -> bool:
        """Check for control characters."""
        control_chars = set(chr(i) for i in range(32)) | {chr(127)}
        return not any(char in control_chars for char in self.alphabet)
    
    def _check_whitespace(self) -> bool:
        """Check for whitespace characters."""
        whitespace_chars = {' ', '\t', '\n', '\r', '\f', '\v'}
        return not any(char in whitespace_chars for char in self.alphabet)
    
    def _check_special_chars(self) -> bool:
        """Check for special characters that might cause issues."""
        special_chars = {'"', "'", '\\', '/', ':', ';', '<', '>', '|', '*', '?'}
        return not any(char in special_chars for char in self.alphabet)
    
    def _check_case_insensitive(self) -> bool:
        """Check if alphabet is case-insensitive safe."""
        lower_chars = set(char.lower() for char in self.alphabet)
        upper_chars = set(char.upper() for char in self.alphabet)
        return len(lower_chars.intersection(upper_chars)) == 0


class BaseNDecoder:
    """
    High-radix base-N decoder.
    
    Decodes base-N encoded data back to original format with
    error detection and validation.
    """
    
    def __init__(self, base: int, alphabet: Optional[str] = None,
                 error_detection: bool = False):
        """
        Initialize base-N decoder.
        
        Args:
            base: Decoding base (2-1024)
            alphabet: Custom alphabet (if None, uses standard alphabet)
            error_detection: Enable error detection validation
        """
        if not 2 <= base <= 1024:
            raise ValueError("Base must be between 2 and 1024")
        
        self.base = base
        self.error_detection = error_detection
        
        # Set alphabet (same as encoder)
        if alphabet is None:
            if base in BaseNEncoder.STANDARD_ALPHABETS:
                self.alphabet = BaseNEncoder.STANDARD_ALPHABETS[base]
            else:
                self.alphabet = self._generate_default_alphabet(base)
        else:
            if len(alphabet) != base:
                raise ValueError(f"Alphabet length ({len(alphabet)}) must equal base ({base})")
            self.alphabet = alphabet
        
        # Create mapping dictionary
        self.char_to_value = {char: i for i, char in enumerate(self.alphabet)}
    
    def _generate_default_alphabet(self, base: int) -> str:
        """Generate default alphabet (same as encoder)."""
        encoder = BaseNEncoder(base)
        return encoder.alphabet
    
    def decode(self, encoded_data: str) -> bytes:
        """
        Decode base-N encoded data.
        
        Args:
            encoded_data: Base-N encoded string
            
        Returns:
            Decoded bytes
            
        Raises:
            ValueError: If encoded data contains invalid characters
        """
        # Validate input
        for char in encoded_data:
            if char not in self.char_to_value:
                raise ValueError(f"Invalid character '{char}' in encoded data")
        
        # Convert from base-N to integer
        data_int = 0
        for char in encoded_data:
            data_int = data_int * self.base + self.char_to_value[char]
        
        # Convert integer to bytes
        if data_int == 0:
            return b'\x00'
        
        # Calculate number of bytes needed
        num_bytes = (data_int.bit_length() + 7) // 8
        
        # Handle error detection
        if self.error_detection and num_bytes >= 4:
            # Extract checksum
            checksum_bytes = data_int.to_bytes(num_bytes, 'big')[-4:]
            data_bytes = data_int.to_bytes(num_bytes, 'big')[:-4]
            checksum = int.from_bytes(checksum_bytes, 'big')
            
            # Validate checksum
            calculated_checksum = self._calculate_checksum(data_bytes)
            if checksum != calculated_checksum:
                raise ValueError("Checksum validation failed - data may be corrupted")
            
            return data_bytes
        else:
            return data_int.to_bytes(num_bytes, 'big')
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate checksum (same as encoder)."""
        return sum(data) & 0xFFFFFFFF
    
    def decode_to_string(self, encoded_data: str, encoding: str = 'utf-8') -> str:
        """
        Decode base-N data to string.
        
        Args:
            encoded_data: Base-N encoded string
            encoding: String encoding (default: utf-8)
            
        Returns:
            Decoded string
        """
        data_bytes = self.decode(encoded_data)
        return data_bytes.decode(encoding)
    
    def decode_to_list(self, encoded_data: str) -> List[int]:
        """
        Decode base-N data to list of integers.
        
        Args:
            encoded_data: Base-N encoded string
            
        Returns:
            List of integers
        """
        data_bytes = self.decode(encoded_data)
        return list(data_bytes)
