# This file intentionally left blank to indicate that this directory is a package.

from .aurorascan import main as aurorascan_main, health_check as aurorascan_health_check
from .cipherspear import main as cipherspear_main, health_check as cipherspear_health_check
from .mythickey import main as mythickey_main, health_check as mythickey_health_check
from .nemesishydra import main as nemesishydra_main, health_check as nemesishydra_health_check
from .obsidianhunt import main as obsidianhunt_main, health_check as obsidianhunt_health_check
from .skybreaker import main as skybreaker_main, health_check as skybreaker_health_check
from .spectratrace import main as spectratrace_main, health_check as spectratrace_health_check
from .vectorflux import main as vectorflux_main, health_check as vectorflux_health_check
from .patentprobe import main as patentprobe_main, health_check as patentprobe_health_check
from .autoscythe import main as autoscythe_main, health_check as autoscythe_health_check
from .quantum_leap_assessor import main as quantum_leap_assessor_main, health_check as quantum_leap_assessor_health_check

TOOL_REGISTRY = {
    "AuroraScan": {"main": aurorascan_main, "health_check": aurorascan_health_check},
    "CipherSpear": {"main": cipherspear_main, "health_check": cipherspear_health_check},
    "MythicKey": {"main": mythickey_main, "health_check": mythickey_health_check},
    "NemesisHydra": {"main": nemesishydra_main, "health_check": nemesishydra_health_check},
    "ObsidianHunt": {"main": obsidianhunt_main, "health_check": obsidianhunt_health_check},
    "SkyBreaker": {"main": skybreaker_main, "health_check": skybreaker_health_check},
    "SpectraTrace": {"main": spectratrace_main, "health_check": spectratrace_health_check},
    "VectorFlux": {"main": vectorflux_main, "health_check": vectorflux_health_check},
    "PatentProbe": {"main": patentprobe_main, "health_check": patentprobe_health_check},
    "AutoScythe": {"main": autoscythe_main, "health_check": autoscythe_health_check},
    "QuantumLeapAssessor": {"main": quantum_leap_assessor_main, "health_check": quantum_leap_assessor_health_check},
}
