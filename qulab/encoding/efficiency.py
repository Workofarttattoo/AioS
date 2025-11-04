"""
Efficiency analysis for base-N encoding.

Implements comprehensive efficiency analysis including compression ratios,
symbol efficiency, alphabet optimization, and performance benchmarking.

References:
- Salomon, D. (2007). Data compression: the complete reference.
- Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import time
from pydantic import BaseModel, Field
import logging

from .base_n import BaseNEncoder, BaseNDecoder
from .packing import PackingOptimizer

logger = logging.getLogger(__name__)


class EfficiencyReport(BaseModel):
    """Comprehensive efficiency analysis report."""
    
    data_size: int = Field(..., description="Original data size in bytes")
    encoding_time: float = Field(..., description="Encoding time in seconds")
    decoding_time: float = Field(..., description="Decoding time in seconds")
    compression_ratio: float = Field(..., description="Compression ratio")
    space_efficiency: float = Field(..., description="Space efficiency (0-1)")
    time_efficiency: float = Field(..., description="Time efficiency (0-1)")
    symbol_efficiency: float = Field(..., description="Symbol efficiency (0-1)")
    alphabet_efficiency: float = Field(..., description="Alphabet efficiency (0-1)")
    overall_efficiency: float = Field(..., description="Overall efficiency score")
    recommendations: List[str] = Field(..., description="Efficiency improvement recommendations")
    
    class Config:
        arbitrary_types_allowed = True


class EfficiencyAnalyzer:
    """
    Comprehensive efficiency analyzer for base-N encoding.
    
    Analyzes multiple aspects of encoding efficiency including compression,
    speed, symbol usage, and alphabet optimization.
    """
    
    def __init__(self, benchmark_iterations: int = 10):
        """
        Initialize efficiency analyzer.
        
        Args:
            benchmark_iterations: Number of iterations for timing benchmarks
        """
        self.benchmark_iterations = benchmark_iterations
    
    def analyze_efficiency(self, data: Union[bytes, str, List[int]], 
                          base: int, alphabet: Optional[str] = None) -> EfficiencyReport:
        """
        Perform comprehensive efficiency analysis.
        
        Args:
            data: Data to analyze
            base: Encoding base
            alphabet: Custom alphabet (optional)
            
        Returns:
            EfficiencyReport with detailed analysis
        """
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, list):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        data_size = len(data_bytes)
        
        # Create encoder and decoder
        encoder = BaseNEncoder(base, alphabet)
        decoder = BaseNDecoder(base, alphabet)
        
        # Benchmark encoding/decoding times
        encoding_time = self._benchmark_encoding(encoder, data_bytes)
        decoding_time = self._benchmark_decoding(decoder, data_bytes)
        
        # Encode data for analysis
        encoding_result = encoder.encode(data_bytes)
        
        # Calculate efficiency metrics
        compression_ratio = encoding_result.compression_ratio
        space_efficiency = self._calculate_space_efficiency(encoding_result)
        time_efficiency = self._calculate_time_efficiency(encoding_time, decoding_time, data_size)
        symbol_efficiency = self._calculate_symbol_efficiency(encoding_result.encoded_data, encoder.alphabet)
        alphabet_efficiency = self._calculate_alphabet_efficiency(encoder.alphabet, base)
        
        # Calculate overall efficiency
        overall_efficiency = self._calculate_overall_efficiency(
            space_efficiency, time_efficiency, symbol_efficiency, alphabet_efficiency
        )
        
        # Generate recommendations
        recommendations = self._generate_efficiency_recommendations(
            compression_ratio, space_efficiency, time_efficiency, 
            symbol_efficiency, alphabet_efficiency, base
        )
        
        return EfficiencyReport(
            data_size=data_size,
            encoding_time=encoding_time,
            decoding_time=decoding_time,
            compression_ratio=compression_ratio,
            space_efficiency=space_efficiency,
            time_efficiency=time_efficiency,
            symbol_efficiency=symbol_efficiency,
            alphabet_efficiency=alphabet_efficiency,
            overall_efficiency=overall_efficiency,
            recommendations=recommendations
        )
    
    def _benchmark_encoding(self, encoder: BaseNEncoder, data: bytes) -> float:
        """Benchmark encoding performance."""
        times = []
        
        for _ in range(self.benchmark_iterations):
            start_time = time.perf_counter()
            encoder.encode(data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return np.mean(times)
    
    def _benchmark_decoding(self, decoder: BaseNDecoder, data: bytes) -> float:
        """Benchmark decoding performance."""
        # First encode the data
        encoder = BaseNEncoder(decoder.base, decoder.alphabet)
        encoded_data = encoder.encode(data).encoded_data
        
        times = []
        
        for _ in range(self.benchmark_iterations):
            start_time = time.perf_counter()
            decoder.decode(encoded_data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return np.mean(times)
    
    def _calculate_space_efficiency(self, encoding_result) -> float:
        """Calculate space efficiency (compression effectiveness)."""
        # Theoretical maximum compression ratio for this base
        theoretical_ratio = 8 / np.log2(encoding_result.base)
        
        # Actual compression ratio
        actual_ratio = encoding_result.compression_ratio
        
        # Space efficiency is the ratio of actual to theoretical
        if theoretical_ratio > 0:
            efficiency = min(1.0, actual_ratio / theoretical_ratio)
        else:
            efficiency = 0.0
        
        return efficiency
    
    def _calculate_time_efficiency(self, encoding_time: float, decoding_time: float, 
                                 data_size: int) -> float:
        """Calculate time efficiency (speed relative to data size)."""
        # Calculate throughput (bytes per second)
        total_time = encoding_time + decoding_time
        if total_time > 0:
            throughput = data_size / total_time
        else:
            throughput = float('inf')
        
        # Normalize throughput (higher is better, cap at 1GB/s)
        max_throughput = 1e9  # 1 GB/s
        efficiency = min(1.0, throughput / max_throughput)
        
        return efficiency
    
    def _calculate_symbol_efficiency(self, encoded_data: str, alphabet: str) -> float:
        """Calculate symbol efficiency (alphabet usage)."""
        if not encoded_data:
            return 0.0
        
        # Count symbol usage
        symbol_counts = {}
        for char in encoded_data:
            symbol_counts[char] = symbol_counts.get(char, 0) + 1
        
        # Calculate entropy of symbol distribution
        total_symbols = len(encoded_data)
        entropy = 0.0
        
        for count in symbol_counts.values():
            probability = count / total_symbols
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Maximum possible entropy (uniform distribution)
        max_entropy = np.log2(len(alphabet))
        
        # Symbol efficiency is the ratio of actual to maximum entropy
        if max_entropy > 0:
            efficiency = entropy / max_entropy
        else:
            efficiency = 0.0
        
        return efficiency
    
    def _calculate_alphabet_efficiency(self, alphabet: str, base: int) -> float:
        """Calculate alphabet efficiency (character set optimization)."""
        # Check for safety issues
        encoder = BaseNEncoder(base, alphabet)
        safety_checks = encoder.validate_alphabet_safety()
        
        # Count safety violations
        violations = sum(1 for check in safety_checks.values() if not check)
        
        # Alphabet efficiency decreases with violations
        max_violations = len(safety_checks)
        efficiency = 1.0 - (violations / max_violations)
        
        return max(0.0, efficiency)
    
    def _calculate_overall_efficiency(self, space_eff: float, time_eff: float, 
                                    symbol_eff: float, alphabet_eff: float) -> float:
        """Calculate overall efficiency score."""
        # Weighted average of different efficiency metrics
        weights = {
            'space': 0.4,      # Compression is most important
            'time': 0.3,       # Speed is important
            'symbol': 0.2,     # Symbol usage is moderately important
            'alphabet': 0.1    # Alphabet safety is least important
        }
        
        overall = (weights['space'] * space_eff + 
                  weights['time'] * time_eff + 
                  weights['symbol'] * symbol_eff + 
                  weights['alphabet'] * alphabet_eff)
        
        return overall
    
    def _generate_efficiency_recommendations(self, compression_ratio: float, 
                                           space_eff: float, time_eff: float,
                                           symbol_eff: float, alphabet_eff: float,
                                           base: int) -> List[str]:
        """Generate efficiency improvement recommendations."""
        recommendations = []
        
        # Space efficiency recommendations
        if space_eff < 0.7:
            recommendations.append("Consider using a different base for better compression")
            if base < 64:
                recommendations.append("Try higher bases (64, 85, 91) for better compression")
            elif base > 128:
                recommendations.append("Try lower bases (64, 85, 91) for better compression")
        
        # Time efficiency recommendations
        if time_eff < 0.5:
            recommendations.append("Encoding/decoding is slow - consider optimizing implementation")
            recommendations.append("Large data may benefit from chunked processing")
        
        # Symbol efficiency recommendations
        if symbol_eff < 0.6:
            recommendations.append("Alphabet usage is uneven - consider data preprocessing")
            recommendations.append("Some symbols in alphabet are unused - consider smaller base")
        
        # Alphabet efficiency recommendations
        if alphabet_eff < 0.8:
            recommendations.append("Alphabet has safety issues - consider safer character set")
            recommendations.append("Review alphabet for ambiguous or problematic characters")
        
        # General recommendations
        if compression_ratio < 1.0:
            recommendations.append("Encoding increases data size - base-N may not be suitable")
        
        if compression_ratio > 2.0:
            recommendations.append("Excellent compression achieved")
        
        return recommendations
    
    def compare_bases_efficiency(self, data: Union[bytes, str, List[int]], 
                               bases: Optional[List[int]] = None) -> Dict[int, EfficiencyReport]:
        """
        Compare efficiency across multiple bases.
        
        Args:
            data: Data to analyze
            bases: List of bases to compare
            
        Returns:
            Dictionary mapping base to efficiency report
        """
        if bases is None:
            bases = [2, 4, 8, 16, 32, 64, 85, 91, 94, 128, 256]
        
        results = {}
        for base in bases:
            if base <= 256:  # Only test reasonable bases
                try:
                    report = self.analyze_efficiency(data, base)
                    results[base] = report
                except Exception as e:
                    logger.warning(f"Failed to analyze base {base}: {e}")
        
        return results
    
    def find_optimal_base(self, data: Union[bytes, str, List[int]], 
                         bases: Optional[List[int]] = None) -> Tuple[int, EfficiencyReport]:
        """
        Find the most efficient base for given data.
        
        Args:
            data: Data to optimize for
            bases: List of bases to test
            
        Returns:
            Tuple of (optimal_base, efficiency_report)
        """
        results = self.compare_bases_efficiency(data, bases)
        
        if not results:
            raise ValueError("No valid bases found for analysis")
        
        # Find base with highest overall efficiency
        optimal_base = max(results.keys(), key=lambda b: results[b].overall_efficiency)
        
        return optimal_base, results[optimal_base]
    
    def benchmark_against_alternatives(self, data: Union[bytes, str, List[int]], 
                                     base: int = 64) -> Dict[str, float]:
        """
        Benchmark base-N encoding against alternative compression methods.
        
        Args:
            data: Data to benchmark
            base: Base for base-N encoding
            
        Returns:
            Dictionary with benchmark results
        """
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, list):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        results = {}
        
        # Base-N encoding
        encoder = BaseNEncoder(base)
        encoding_result = encoder.encode(data_bytes)
        
        start_time = time.perf_counter()
        encoded = encoder.encode(data_bytes)
        encoding_time = time.perf_counter() - start_time
        
        decoder = BaseNDecoder(base)
        start_time = time.perf_counter()
        decoded = decoder.decode(encoded.encoded_data)
        decoding_time = time.perf_counter() - start_time
        
        results['base_n'] = {
            'compression_ratio': encoded.compression_ratio,
            'encoding_time': encoding_time,
            'decoding_time': decoding_time,
            'total_time': encoding_time + decoding_time
        }
        
        # Compare with built-in compression (if available)
        try:
            import gzip
            import bz2
            import lzma
            
            # Gzip compression
            start_time = time.perf_counter()
            gzip_compressed = gzip.compress(data_bytes)
            gzip_encoding_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            gzip_decompressed = gzip.decompress(gzip_compressed)
            gzip_decoding_time = time.perf_counter() - start_time
            
            results['gzip'] = {
                'compression_ratio': len(data_bytes) / len(gzip_compressed),
                'encoding_time': gzip_encoding_time,
                'decoding_time': gzip_decoding_time,
                'total_time': gzip_encoding_time + gzip_decoding_time
            }
            
            # Bzip2 compression
            start_time = time.perf_counter()
            bz2_compressed = bz2.compress(data_bytes)
            bz2_encoding_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            bz2_decompressed = bz2.decompress(bz2_compressed)
            bz2_decoding_time = time.perf_counter() - start_time
            
            results['bz2'] = {
                'compression_ratio': len(data_bytes) / len(bz2_compressed),
                'encoding_time': bz2_encoding_time,
                'decoding_time': bz2_decoding_time,
                'total_time': bz2_encoding_time + bz2_decoding_time
            }
            
        except ImportError:
            logger.warning("Compression libraries not available for comparison")
        
        return results
