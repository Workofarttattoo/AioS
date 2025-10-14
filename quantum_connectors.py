"""
Quantum Platform Connectors for AgentaOS
Connects to IBM Quantum, AWS Braket, D-Wave, Google Cirq, and Rigetti

Enables AgentaOS to leverage real quantum hardware for:
- Quantum optimization
- Quantum machine learning
- Quantum simulation
- Hybrid classical-quantum algorithms
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

# ═══════════════════════════════════════════════════════════════════════
# QUANTUM PLATFORM TYPES
# ═══════════════════════════════════════════════════════════════════════

class QuantumPlatform(Enum):
    """Supported quantum platforms."""
    IBM_QUANTUM = "ibm_quantum"
    AWS_BRAKET = "aws_braket"
    DWAVE = "dwave"
    GOOGLE_CIRQ = "google_cirq"
    RIGETTI = "rigetti"
    AZURE_QUANTUM = "azure_quantum"
    IONQ = "ionq"


@dataclass
class QuantumJob:
    """Quantum job specification."""
    job_id: str
    platform: QuantumPlatform
    circuit: Any  # Platform-specific circuit object
    backend: str
    status: str = "pending"
    result: Optional[Any] = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None


@dataclass
class QuantumBackend:
    """Quantum backend information."""
    name: str
    platform: QuantumPlatform
    num_qubits: int
    is_simulator: bool
    is_available: bool
    queue_length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# IBM QUANTUM CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class IBMQuantumConnector:
    """
    Connector for IBM Quantum Platform.
    Provides access to IBM's quantum computers and simulators.
    """

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.provider = None
        self.backends = {}
        self.connected = False

    async def connect(self) -> bool:
        """Connect to IBM Quantum."""
        try:
            # Try to import Qiskit
            from qiskit import IBMQ

            if self.api_token:
                IBMQ.save_account(self.api_token, overwrite=True)

            # Load account
            provider = IBMQ.load_account()
            self.provider = provider
            self.connected = True

            # Discover backends
            await self._discover_backends()

            print(f"[IBM Quantum] Connected successfully")
            print(f"[IBM Quantum] Available backends: {len(self.backends)}")
            return True

        except ImportError:
            print("[IBM Quantum] Qiskit not installed (pip install qiskit)")
            return False
        except Exception as e:
            print(f"[IBM Quantum] Connection failed: {e}")
            return False

    async def _discover_backends(self):
        """Discover available quantum backends."""
        if not self.provider:
            return

        from qiskit import IBMQ

        for backend in self.provider.backends():
            config = backend.configuration()
            status = backend.status()

            self.backends[backend.name()] = QuantumBackend(
                name=backend.name(),
                platform=QuantumPlatform.IBM_QUANTUM,
                num_qubits=config.n_qubits,
                is_simulator=config.simulator,
                is_available=status.operational and not status.status_msg == 'maintenance',
                queue_length=status.pending_jobs,
                metadata={
                    'basis_gates': config.basis_gates,
                    'coupling_map': str(config.coupling_map)[:100]
                }
            )

    async def submit_circuit(self, circuit, backend_name: str = "ibmq_qasm_simulator") -> QuantumJob:
        """Submit quantum circuit to IBM backend."""
        from qiskit import execute

        backend = self.provider.get_backend(backend_name)

        job = execute(circuit, backend=backend, shots=1024)
        job_id = job.job_id()

        quantum_job = QuantumJob(
            job_id=job_id,
            platform=QuantumPlatform.IBM_QUANTUM,
            circuit=circuit,
            backend=backend_name,
            status="submitted"
        )

        print(f"[IBM Quantum] Job {job_id} submitted to {backend_name}")
        return quantum_job

    async def get_result(self, job_id: str) -> Optional[Any]:
        """Get job result."""
        try:
            job = self.provider.backends.retrieve_job(job_id)
            status = job.status()

            if status.name == 'DONE':
                result = job.result()
                return result
            else:
                return None
        except Exception as e:
            print(f"[IBM Quantum] Error retrieving job {job_id}: {e}")
            return None

    def get_backends(self) -> List[QuantumBackend]:
        """Get list of available backends."""
        return list(self.backends.values())


# ═══════════════════════════════════════════════════════════════════════
# AWS BRAKET CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class AWSBraketConnector:
    """
    Connector for AWS Braket.
    Provides access to multiple quantum hardware providers through AWS.
    """

    def __init__(self, aws_region: str = "us-east-1"):
        self.aws_region = aws_region
        self.braket_client = None
        self.backends = {}
        self.connected = False

    async def connect(self) -> bool:
        """Connect to AWS Braket."""
        try:
            import boto3

            self.braket_client = boto3.client('braket', region_name=self.aws_region)
            self.connected = True

            # Discover devices
            await self._discover_devices()

            print(f"[AWS Braket] Connected to region {self.aws_region}")
            print(f"[AWS Braket] Available devices: {len(self.backends)}")
            return True

        except ImportError:
            print("[AWS Braket] boto3 not installed (pip install boto3)")
            return False
        except Exception as e:
            print(f"[AWS Braket] Connection failed: {e}")
            return False

    async def _discover_devices(self):
        """Discover available quantum devices."""
        if not self.braket_client:
            return

        try:
            response = self.braket_client.search_devices()

            for device in response.get('devices', []):
                device_arn = device['deviceArn']
                device_name = device['deviceName']
                device_type = device['deviceType']
                provider = device['providerName']
                status = device['deviceStatus']

                self.backends[device_name] = QuantumBackend(
                    name=device_name,
                    platform=QuantumPlatform.AWS_BRAKET,
                    num_qubits=device.get('deviceCapabilities', {}).get('paradigm', {}).get('qubitCount', 0),
                    is_simulator=device_type == 'SIMULATOR',
                    is_available=status == 'ONLINE',
                    metadata={
                        'provider': provider,
                        'device_arn': device_arn,
                        'device_type': device_type
                    }
                )

        except Exception as e:
            print(f"[AWS Braket] Error discovering devices: {e}")

    async def submit_circuit(self, circuit, device_arn: str, s3_bucket: str, s3_prefix: str) -> QuantumJob:
        """Submit quantum circuit to AWS Braket."""
        try:
            from braket.aws import AwsDevice
            from braket.circuits import Circuit

            device = AwsDevice(device_arn)

            task = device.run(
                circuit,
                s3_destination_folder=(s3_bucket, s3_prefix),
                shots=1024
            )

            job_id = task.id

            quantum_job = QuantumJob(
                job_id=job_id,
                platform=QuantumPlatform.AWS_BRAKET,
                circuit=circuit,
                backend=device_arn,
                status="submitted"
            )

            print(f"[AWS Braket] Task {job_id} submitted to {device_arn}")
            return quantum_job

        except Exception as e:
            print(f"[AWS Braket] Error submitting circuit: {e}")
            raise

    def get_backends(self) -> List[QuantumBackend]:
        """Get list of available backends."""
        return list(self.backends.values())


# ═══════════════════════════════════════════════════════════════════════
# D-WAVE CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class DWaveConnector:
    """
    Connector for D-Wave quantum annealers.
    Specialized for quantum optimization problems.
    """

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.client = None
        self.solvers = {}
        self.connected = False

    async def connect(self) -> bool:
        """Connect to D-Wave."""
        try:
            from dwave.system import DWaveSampler, LeapHybridSampler

            if self.api_token:
                from dwave.cloud import Client
                self.client = Client.from_config(token=self.api_token)
            else:
                self.client = Client.from_config()

            self.connected = True

            # Discover solvers
            await self._discover_solvers()

            print(f"[D-Wave] Connected successfully")
            print(f"[D-Wave] Available solvers: {len(self.solvers)}")
            return True

        except ImportError:
            print("[D-Wave] dwave-system not installed (pip install dwave-ocean-sdk)")
            return False
        except Exception as e:
            print(f"[D-Wave] Connection failed: {e}")
            return False

    async def _discover_solvers(self):
        """Discover available D-Wave solvers."""
        if not self.client:
            return

        try:
            solvers = self.client.get_solvers()

            for solver in solvers:
                properties = solver.properties

                self.solvers[solver.id] = QuantumBackend(
                    name=solver.id,
                    platform=QuantumPlatform.DWAVE,
                    num_qubits=properties.get('num_qubits', 0),
                    is_simulator='sim' in solver.id.lower(),
                    is_available=solver.status().get('status') == 'online',
                    metadata={
                        'annealing_time_range': properties.get('annealing_time_range'),
                        'topology_type': properties.get('topology', {}).get('type')
                    }
                )

        except Exception as e:
            print(f"[D-Wave] Error discovering solvers: {e}")

    async def solve_qubo(self, qubo_dict: Dict, solver_name: Optional[str] = None) -> QuantumJob:
        """Solve QUBO (Quadratic Unconstrained Binary Optimization) problem."""
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite

            if not solver_name:
                # Use first available hardware solver
                hardware_solvers = [s for s in self.solvers.values() if not s.is_simulator]
                if hardware_solvers:
                    solver_name = hardware_solvers[0].name

            sampler = EmbeddingComposite(DWaveSampler(solver=solver_name))

            # Submit problem
            sampleset = sampler.sample_qubo(qubo_dict, num_reads=1000)

            job_id = f"dwave_{int(time.time() * 1000)}"

            quantum_job = QuantumJob(
                job_id=job_id,
                platform=QuantumPlatform.DWAVE,
                circuit=qubo_dict,
                backend=solver_name or "default",
                status="completed",
                result=sampleset
            )

            print(f"[D-Wave] QUBO problem solved on {solver_name}")
            return quantum_job

        except Exception as e:
            print(f"[D-Wave] Error solving QUBO: {e}")
            raise

    def get_backends(self) -> List[QuantumBackend]:
        """Get list of available solvers."""
        return list(self.solvers.values())


# ═══════════════════════════════════════════════════════════════════════
# GOOGLE CIRQ CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class GoogleCirqConnector:
    """
    Connector for Google Quantum AI (Cirq).
    Provides access to Google's quantum processors.
    """

    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id
        self.engine = None
        self.processors = {}
        self.connected = False

    async def connect(self) -> bool:
        """Connect to Google Quantum AI."""
        try:
            import cirq
            import cirq_google

            if self.project_id:
                self.engine = cirq_google.Engine(project_id=self.project_id)
                self.connected = True

                # Discover processors
                await self._discover_processors()

                print(f"[Google Cirq] Connected to project {self.project_id}")
                print(f"[Google Cirq] Available processors: {len(self.processors)}")
                return True
            else:
                print("[Google Cirq] No project_id provided, using simulators only")
                self.connected = True
                return True

        except ImportError:
            print("[Google Cirq] cirq not installed (pip install cirq cirq-google)")
            return False
        except Exception as e:
            print(f"[Google Cirq] Connection failed: {e}")
            return False

    async def _discover_processors(self):
        """Discover available Google quantum processors."""
        if not self.engine:
            return

        try:
            processors = self.engine.list_processors()

            for processor in processors:
                processor_id = processor.processor_id

                self.processors[processor_id] = QuantumBackend(
                    name=processor_id,
                    platform=QuantumPlatform.GOOGLE_CIRQ,
                    num_qubits=len(processor.get_device().qubits) if hasattr(processor.get_device(), 'qubits') else 0,
                    is_simulator=False,
                    is_available=True,
                    metadata={
                        'processor_id': processor_id
                    }
                )

        except Exception as e:
            print(f"[Google Cirq] Error discovering processors: {e}")

    async def submit_circuit(self, circuit, processor_id: Optional[str] = None) -> QuantumJob:
        """Submit circuit to Google quantum processor."""
        try:
            import cirq

            if processor_id and self.engine:
                # Submit to hardware
                job = self.engine.run_sweep(
                    program=circuit,
                    processor_ids=[processor_id],
                    repetitions=1000
                )
                job_id = f"cirq_{processor_id}_{int(time.time() * 1000)}"
            else:
                # Run on simulator
                simulator = cirq.Simulator()
                result = simulator.run(circuit, repetitions=1000)
                job_id = f"cirq_sim_{int(time.time() * 1000)}"

            quantum_job = QuantumJob(
                job_id=job_id,
                platform=QuantumPlatform.GOOGLE_CIRQ,
                circuit=circuit,
                backend=processor_id or "simulator",
                status="submitted"
            )

            print(f"[Google Cirq] Job {job_id} submitted")
            return quantum_job

        except Exception as e:
            print(f"[Google Cirq] Error submitting circuit: {e}")
            raise

    def get_backends(self) -> List[QuantumBackend]:
        """Get list of available processors."""
        return list(self.processors.values())


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED QUANTUM MANAGER
# ═══════════════════════════════════════════════════════════════════════

class QuantumPlatformManager:
    """
    Unified manager for all quantum platforms.
    Provides single interface to multiple quantum providers.
    """

    def __init__(self):
        self.connectors = {}
        self.jobs = {}
        self.initialized = False

    async def initialize(self, config: Dict[str, Any] = None) -> Dict[QuantumPlatform, bool]:
        """
        Initialize all available quantum platforms.

        Args:
            config: Platform-specific configuration
                {
                    'ibm_quantum': {'api_token': '...'},
                    'aws_braket': {'region': 'us-east-1'},
                    'dwave': {'api_token': '...'},
                    'google_cirq': {'project_id': '...'}
                }

        Returns:
            Dict mapping platforms to connection status
        """
        config = config or {}
        results = {}

        # Initialize IBM Quantum
        if 'ibm_quantum' in config:
            ibm = IBMQuantumConnector(api_token=config['ibm_quantum'].get('api_token'))
            if await ibm.connect():
                self.connectors[QuantumPlatform.IBM_QUANTUM] = ibm
                results[QuantumPlatform.IBM_QUANTUM] = True
            else:
                results[QuantumPlatform.IBM_QUANTUM] = False

        # Initialize AWS Braket
        if 'aws_braket' in config:
            braket = AWSBraketConnector(aws_region=config['aws_braket'].get('region', 'us-east-1'))
            if await braket.connect():
                self.connectors[QuantumPlatform.AWS_BRAKET] = braket
                results[QuantumPlatform.AWS_BRAKET] = True
            else:
                results[QuantumPlatform.AWS_BRAKET] = False

        # Initialize D-Wave
        if 'dwave' in config:
            dwave = DWaveConnector(api_token=config['dwave'].get('api_token'))
            if await dwave.connect():
                self.connectors[QuantumPlatform.DWAVE] = dwave
                results[QuantumPlatform.DWAVE] = True
            else:
                results[QuantumPlatform.DWAVE] = False

        # Initialize Google Cirq
        if 'google_cirq' in config:
            cirq = GoogleCirqConnector(project_id=config['google_cirq'].get('project_id'))
            if await cirq.connect():
                self.connectors[QuantumPlatform.GOOGLE_CIRQ] = cirq
                results[QuantumPlatform.GOOGLE_CIRQ] = True
            else:
                results[QuantumPlatform.GOOGLE_CIRQ] = False

        self.initialized = True
        return results

    def get_all_backends(self) -> List[QuantumBackend]:
        """Get all available quantum backends across all platforms."""
        backends = []
        for connector in self.connectors.values():
            backends.extend(connector.get_backends())
        return backends

    def get_best_backend(self, min_qubits: int = 0, prefer_hardware: bool = True) -> Optional[QuantumBackend]:
        """
        Find best available backend for job.

        Args:
            min_qubits: Minimum qubits required
            prefer_hardware: Prefer real hardware over simulators

        Returns:
            Best matching backend or None
        """
        backends = self.get_all_backends()

        # Filter by requirements
        candidates = [
            b for b in backends
            if b.num_qubits >= min_qubits and b.is_available
        ]

        if not candidates:
            return None

        # Sort by preference
        if prefer_hardware:
            # Prefer hardware, then by queue length, then by qubit count
            candidates.sort(key=lambda b: (b.is_simulator, b.queue_length, -b.num_qubits))
        else:
            # Prefer simulators (faster), then by qubit count
            candidates.sort(key=lambda b: (not b.is_simulator, -b.num_qubits))

        return candidates[0]

    async def submit_job(self, circuit: Any, platform: QuantumPlatform, backend_name: Optional[str] = None, **kwargs) -> Optional[QuantumJob]:
        """
        Submit job to specified quantum platform.

        Args:
            circuit: Platform-specific circuit object
            platform: Target quantum platform
            backend_name: Specific backend name (optional)
            **kwargs: Platform-specific parameters

        Returns:
            QuantumJob if successful, None otherwise
        """
        if platform not in self.connectors:
            print(f"[QuantumManager] Platform {platform.value} not connected")
            return None

        connector = self.connectors[platform]

        try:
            if platform == QuantumPlatform.IBM_QUANTUM:
                job = await connector.submit_circuit(circuit, backend_name or "ibmq_qasm_simulator")
            elif platform == QuantumPlatform.AWS_BRAKET:
                job = await connector.submit_circuit(circuit, kwargs['device_arn'], kwargs['s3_bucket'], kwargs['s3_prefix'])
            elif platform == QuantumPlatform.DWAVE:
                job = await connector.solve_qubo(circuit, backend_name)
            elif platform == QuantumPlatform.GOOGLE_CIRQ:
                job = await connector.submit_circuit(circuit, backend_name)
            else:
                print(f"[QuantumManager] Platform {platform.value} not yet implemented")
                return None

            self.jobs[job.job_id] = job
            return job

        except Exception as e:
            print(f"[QuantumManager] Error submitting job: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get status of quantum platform manager."""
        return {
            'initialized': self.initialized,
            'connected_platforms': [p.value for p in self.connectors.keys()],
            'total_backends': len(self.get_all_backends()),
            'active_jobs': len([j for j in self.jobs.values() if j.status in ['pending', 'submitted', 'running']])
        }


def check_quantum_platform_dependencies() -> Dict[str, bool]:
    """Check which quantum platform SDKs are installed."""
    deps = {}

    try:
        import qiskit
        deps['qiskit'] = True
    except ImportError:
        deps['qiskit'] = False

    try:
        import boto3
        deps['boto3'] = True
    except ImportError:
        deps['boto3'] = False

    try:
        import dwave.system
        deps['dwave'] = True
    except ImportError:
        deps['dwave'] = False

    try:
        import cirq
        import cirq_google
        deps['cirq'] = True
    except ImportError:
        deps['cirq'] = False

    return deps
