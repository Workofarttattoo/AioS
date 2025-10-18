"""
Packing optimization for base-N encoding.

Implements optimal packing strategies for different data types and
encoding bases to maximize compression efficiency.

References:
- Salomon, D. (2007). Data compression: the complete reference.
- Witten, I. H., et al. (1999). Managing gigabytes: compressing and indexing documents and images.
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field
import logging

from .base_n import BaseNEncoder, BaseNDecoder

logger = logging.getLogger(__name__)


class PackingResult(BaseModel):
    """Result of packing optimization."""
    
    optimal_base: int = Field(..., description="Optimal encoding base")
    optimal_alphabet: str = Field(..., description="Optimal alphabet")
    compression_ratio: float = Field(..., description="Achieved compression ratio")
    packing_efficiency: float = Field(..., description="Packing efficiency (0-1)")
    data_type: str = Field(..., description="Type of data being packed")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    
    class Config:
        arbitrary_types_allowed = True


class PackingOptimizer:
    """
    Packing optimizer for base-N encoding.
    
    Analyzes data characteristics and recommends optimal encoding parameters
    for maximum compression efficiency.
    """
    
    def __init__(self, test_bases: Optional[List[int]] = None):
        """
        Initialize packing optimizer.
        
        Args:
            test_bases: List of bases to test (default: common bases)
        """
        self.test_bases = test_bases or [2, 4, 8, 16, 32, 64, 85, 91, 94, 128, 256]
    
    def optimize_packing(self, data: Union[bytes, str, List[int]], 
                        data_type: Optional[str] = None) -> PackingResult:
        """
        Optimize packing for given data.
        
        Args:
            data: Data to optimize packing for
            data_type: Type of data ('binary', 'text', 'numeric', 'mixed')
            
        Returns:
            PackingResult with optimization recommendations
        """
        # Analyze data characteristics
        if data_type is None:
            data_type = self._analyze_data_type(data)
        
        # Convert to bytes for analysis
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, list):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        # Test different bases
        results = []
        for base in self.test_bases:
            if base <= 256:  # Only test reasonable bases
                result = self._test_base_encoding(data_bytes, base, data_type)
                results.append(result)
        
        # Find optimal base
        optimal_result = max(results, key=lambda x: x['compression_ratio'])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(optimal_result, data_type)
        
        return PackingResult(
            optimal_base=optimal_result['base'],
            optimal_alphabet=optimal_result['alphabet'],
            compression_ratio=optimal_result['compression_ratio'],
            packing_efficiency=optimal_result['efficiency'],
            data_type=data_type,
            recommendations=recommendations
        )
    
    def _analyze_data_type(self, data: Union[bytes, str, List[int]]) -> str:
        """Analyze the type of data for optimization."""
        if isinstance(data, str):
            # Check if it's mostly ASCII text
            if all(ord(c) < 128 for c in data):
                return 'text'
            else:
                return 'unicode_text'
        elif isinstance(data, list):
            # Check if it's numeric data
            if all(isinstance(x, int) and 0 <= x <= 255 for x in data):
                return 'numeric'
            else:
                return 'mixed'
        else:
            # Analyze byte data
            data_bytes = data
            if len(data_bytes) == 0:
                return 'empty'
            
            # Check entropy and patterns
            entropy = self._calculate_entropy(data_bytes)
            if entropy < 2.0:
                return 'low_entropy'
            elif entropy > 7.0:
                return 'high_entropy'
            else:
                return 'binary'
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        byte_counts = np.bincount(data, minlength=256)
        probabilities = byte_counts / len(data)
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _test_base_encoding(self, data_bytes: bytes, base: int, data_type: str) -> Dict:
        """Test encoding with specific base."""
        try:
            encoder = BaseNEncoder(base)
            result = encoder.encode(data_bytes)
            
            # Calculate efficiency metrics
            theoretical_ratio = 8 / np.log2(base)
            actual_ratio = result.compression_ratio
            efficiency = actual_ratio / theoretical_ratio if theoretical_ratio > 0 else 0
            
            return {
                'base': base,
                'alphabet': encoder.alphabet,
                'compression_ratio': actual_ratio,
                'efficiency': efficiency,
                'encoded_length': result.encoded_length,
                'original_length': result.original_length,
                'padding': result.padding
            }
        except Exception as e:
            logger.warning(f"Failed to test base {base}: {e}")
            return {
                'base': base,
                'alphabet': '',
                'compression_ratio': 0.0,
                'efficiency': 0.0,
                'encoded_length': 0,
                'original_length': len(data_bytes),
                'padding': 0
            }
    
    def _generate_recommendations(self, optimal_result: Dict, data_type: str) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        base = optimal_result['base']
        efficiency = optimal_result['efficiency']
        
        # Base-specific recommendations
        if base == 2:
            recommendations.append("Binary encoding is optimal for this data")
        elif base == 16:
            recommendations.append("Hexadecimal encoding provides good readability")
        elif base == 64:
            recommendations.append("Base64 encoding is standard and widely supported")
        elif base == 85:
            recommendations.append("Base85 encoding provides excellent compression")
        elif base == 91:
            recommendations.append("Base91 encoding maximizes printable ASCII usage")
        elif base == 94:
            recommendations.append("Base94 encoding uses all printable ASCII characters")
        
        # Efficiency recommendations
        if efficiency > 0.9:
            recommendations.append("Excellent packing efficiency achieved")
        elif efficiency > 0.8:
            recommendations.append("Good packing efficiency")
        elif efficiency > 0.7:
            recommendations.append("Moderate packing efficiency - consider data preprocessing")
        else:
            recommendations.append("Low packing efficiency - data may not be suitable for base-N encoding")
        
        # Data type specific recommendations
        if data_type == 'text':
            recommendations.append("Consider text-specific compression before base-N encoding")
        elif data_type == 'numeric':
            recommendations.append("Numeric data may benefit from specialized encoding")
        elif data_type == 'low_entropy':
            recommendations.append("Low entropy data is ideal for base-N encoding")
        elif data_type == 'high_entropy':
            recommendations.append("High entropy data may not compress well with base-N encoding")
        
        return recommendations
    
    def compare_bases(self, data: Union[bytes, str, List[int]], 
                     bases: Optional[List[int]] = None) -> Dict[int, Dict]:
        """
        Compare encoding efficiency across multiple bases.
        
        Args:
            data: Data to encode
            bases: List of bases to compare (default: test_bases)
            
        Returns:
            Dictionary mapping base to encoding results
        """
        if bases is None:
            bases = self.test_bases
        
        results = {}
        for base in bases:
            if base <= 256:  # Only test reasonable bases
                result = self._test_base_encoding(data, base, 'mixed')
                results[base] = result
        
        return results
    
    def find_optimal_base_range(self, data: Union[bytes, str, List[int]], 
                              min_base: int = 2, max_base: int = 256) -> Dict:
        """
        Find optimal base within a range using binary search.
        
        Args:
            data: Data to optimize for
            min_base: Minimum base to test
            max_base: Maximum base to test
            
        Returns:
            Dictionary with optimal base and results
        """
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, list):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        best_base = min_base
        best_ratio = 0.0
        
        # Test bases in the range
        for base in range(min_base, min(max_base + 1, 257)):
            try:
                result = self._test_base_encoding(data_bytes, base, 'mixed')
                if result['compression_ratio'] > best_ratio:
                    best_ratio = result['compression_ratio']
                    best_base = base
            except Exception:
                continue
        
        return {
            'optimal_base': best_base,
            'compression_ratio': best_ratio,
            'tested_range': (min_base, max_base)
        }
    
    def analyze_padding_overhead(self, data: Union[bytes, str, List[int]], 
                               bases: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Analyze padding overhead for different bases.
        
        Args:
            data: Data to analyze
            bases: List of bases to test
            
        Returns:
            Dictionary mapping base to padding overhead percentage
        """
        if bases is None:
            bases = [2, 4, 8, 16, 32, 64, 85, 91, 94, 128, 256]
        
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, list):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        padding_overhead = {}
        for base in bases:
            if base <= 256:
                try:
                    encoder = BaseNEncoder(base)
                    result = encoder.encode(data_bytes)
                    overhead = (result.padding / result.original_length) * 100
                    padding_overhead[base] = overhead
                except Exception:
                    padding_overhead[base] = float('inf')
        
        return padding_overhead
    
    def recommend_preprocessing(self, data: Union[bytes, str, List[int]], 
                              data_type: str) -> List[str]:
        """
        Recommend data preprocessing steps for better compression.
        
        Args:
            data: Data to analyze
            data_type: Type of data
            
        Returns:
            List of preprocessing recommendations
        """
        recommendations = []
        
        if data_type == 'text':
            recommendations.extend([
                "Consider removing redundant whitespace",
                "Normalize line endings (CRLF -> LF)",
                "Remove comments or metadata if not needed"
            ])
        elif data_type == 'numeric':
            recommendations.extend([
                "Consider delta encoding for sequential numbers",
                "Use variable-length encoding for small numbers",
                "Group similar values together"
            ])
        elif data_type == 'binary':
            recommendations.extend([
                "Consider run-length encoding for repeated patterns",
                "Use dictionary compression for common byte sequences",
                "Apply entropy coding for optimal compression"
            ])
        elif data_type == 'high_entropy':
            recommendations.extend([
                "Data may already be compressed or encrypted",
                "Consider if base-N encoding is necessary",
                "Alternative compression methods may be more suitable"
            ])
        
        return recommendations
