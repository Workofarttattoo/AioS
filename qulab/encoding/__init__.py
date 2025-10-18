"""
High-radix base-N encoding library.

Implements efficient encoding systems for bases 2 through 1024 with
packing optimization, alphabet safety profiles, and efficiency analysis.
"""

from .base_n import BaseNEncoder, BaseNDecoder, EncodingResult
from .packing import PackingOptimizer, PackingResult
from .efficiency import EfficiencyAnalyzer, EfficiencyReport

__all__ = [
    "BaseNEncoder",
    "BaseNDecoder", 
    "EncodingResult",
    "PackingOptimizer",
    "PackingResult",
    "EfficiencyAnalyzer",
    "EfficiencyReport",
]
