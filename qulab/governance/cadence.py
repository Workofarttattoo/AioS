"""
Cadence planning for optimal measurement scheduling.

Implements optimal measurement scheduling based on evidence accumulation
and uncertainty reduction for teleportation experiments.

References:
- Lindley, D. V. (1956). On a measure of the information provided by an experiment.
- Chaloner, K., & Verdinelli, I. (1995). Bayesian experimental design.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from .ledger import EvidenceLedger
from .forecasting import MonteCarloForecaster

logger = logging.getLogger(__name__)


class CadencePlan(BaseModel):
    """Optimal cadence plan for measurements."""
    
    measurement_times: List[datetime] = Field(..., description="Scheduled measurement times")
    expected_uncertainty: List[float] = Field(..., description="Expected uncertainty at each measurement")
    information_gain: List[float] = Field(..., description="Expected information gain")
    total_cost: float = Field(..., description="Total cost of measurement plan")
    optimization_objective: str = Field(..., description="Optimization objective used")
    
    class Config:
        arbitrary_types_allowed = True


class CadencePlanner:
    """
    Optimal cadence planner for teleportation measurements.
    
    Plans measurement schedules to optimize information gain, uncertainty reduction,
    or cost-effectiveness based on Bayesian experimental design principles.
    """
    
    def __init__(self, ledger: EvidenceLedger, 
                 measurement_cost: float = 1.0,
                 time_horizon_days: int = 30):
        """
        Initialize cadence planner.
        
        Args:
            ledger: Evidence ledger with current state
            measurement_cost: Cost per measurement
            time_horizon_days: Planning horizon in days
        """
        self.ledger = ledger
        self.measurement_cost = measurement_cost
        self.time_horizon_days = time_horizon_days
        self.forecaster = MonteCarloForecaster(ledger)
    
    def plan_optimal_cadence(self, n_measurements: int, 
                           objective: str = 'uncertainty_reduction',
                           min_interval_hours: int = 1) -> CadencePlan:
        """
        Plan optimal measurement cadence.
        
        Args:
            n_measurements: Number of measurements to schedule
            objective: Optimization objective ('uncertainty_reduction', 'information_gain', 'cost_effective')
            min_interval_hours: Minimum interval between measurements in hours
            
        Returns:
            CadencePlan with optimal schedule
        """
        if n_measurements <= 0:
            raise ValueError("Number of measurements must be positive")
        
        # Define optimization objective
        if objective == 'uncertainty_reduction':
            optimal_times = self._optimize_uncertainty_reduction(n_measurements, min_interval_hours)
        elif objective == 'information_gain':
            optimal_times = self._optimize_information_gain(n_measurements, min_interval_hours)
        elif objective == 'cost_effective':
            optimal_times = self._optimize_cost_effectiveness(n_measurements, min_interval_hours)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Calculate expected outcomes
        expected_uncertainty = self._calculate_expected_uncertainty(optimal_times)
        information_gain = self._calculate_information_gain(optimal_times)
        total_cost = len(optimal_times) * self.measurement_cost
        
        return CadencePlan(
            measurement_times=optimal_times,
            expected_uncertainty=expected_uncertainty,
            information_gain=information_gain,
            total_cost=total_cost,
            optimization_objective=objective
        )
    
    def _optimize_uncertainty_reduction(self, n_measurements: int, 
                                      min_interval_hours: int) -> List[datetime]:
        """Optimize for maximum uncertainty reduction."""
        # Start with current time
        start_time = datetime.now()
        end_time = start_time + timedelta(days=self.time_horizon_days)
        
        # Convert to hours for optimization
        start_hours = 0
        end_hours = self.time_horizon_days * 24
        
        def uncertainty_objective(measurement_hours):
            """Objective function: negative of uncertainty reduction."""
            measurement_times = [start_time + timedelta(hours=h) for h in measurement_hours]
            expected_uncertainty = self._calculate_expected_uncertainty(measurement_times)
            return -np.sum(expected_uncertainty)  # Minimize negative = maximize reduction
        
        # Constraint: minimum interval between measurements
        constraints = []
        for i in range(n_measurements - 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i+1] - x[i] - min_interval_hours
            })
        
        # Bounds: measurements must be within time horizon
        bounds = [(start_hours, end_hours) for _ in range(n_measurements)]
        
        # Initial guess: evenly spaced measurements
        x0 = np.linspace(start_hours, end_hours, n_measurements)
        
        # Optimize
        result = minimize(uncertainty_objective, x0, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_hours = result.x
            optimal_times = [start_time + timedelta(hours=h) for h in optimal_hours]
            return sorted(optimal_times)
        else:
            # Fallback: evenly spaced measurements
            return self._create_evenly_spaced_plan(n_measurements, min_interval_hours)
    
    def _optimize_information_gain(self, n_measurements: int, 
                                 min_interval_hours: int) -> List[datetime]:
        """Optimize for maximum information gain."""
        # Similar to uncertainty reduction but with information-theoretic objective
        start_time = datetime.now()
        end_time = start_time + timedelta(days=self.time_horizon_days)
        
        start_hours = 0
        end_hours = self.time_horizon_days * 24
        
        def information_objective(measurement_hours):
            """Objective function: negative of information gain."""
            measurement_times = [start_time + timedelta(hours=h) for h in measurement_hours]
            information_gain = self._calculate_information_gain(measurement_times)
            return -np.sum(information_gain)  # Minimize negative = maximize gain
        
        # Constraints and bounds (same as uncertainty reduction)
        constraints = []
        for i in range(n_measurements - 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i+1] - x[i] - min_interval_hours
            })
        
        bounds = [(start_hours, end_hours) for _ in range(n_measurements)]
        x0 = np.linspace(start_hours, end_hours, n_measurements)
        
        result = minimize(information_objective, x0, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_hours = result.x
            optimal_times = [start_time + timedelta(hours=h) for h in optimal_hours]
            return sorted(optimal_times)
        else:
            return self._create_evenly_spaced_plan(n_measurements, min_interval_hours)
    
    def _optimize_cost_effectiveness(self, n_measurements: int, 
                                   min_interval_hours: int) -> List[datetime]:
        """Optimize for cost-effectiveness (information gain per cost)."""
        # Find optimal number of measurements first
        best_n = self._find_optimal_number_of_measurements()
        
        # Then optimize timing
        if best_n <= n_measurements:
            return self._optimize_information_gain(best_n, min_interval_hours)
        else:
            return self._optimize_information_gain(n_measurements, min_interval_hours)
    
    def _find_optimal_number_of_measurements(self) -> int:
        """Find optimal number of measurements for cost-effectiveness."""
        max_measurements = min(50, self.time_horizon_days * 24 // 1)  # Max 1 per hour
        
        best_ratio = 0
        best_n = 1
        
        for n in range(1, max_measurements + 1):
            # Calculate information gain and cost
            plan = self._create_evenly_spaced_plan(n, 1)
            information_gain = sum(self._calculate_information_gain(plan))
            cost = n * self.measurement_cost
            
            if cost > 0:
                ratio = information_gain / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_n = n
        
        return best_n
    
    def _calculate_expected_uncertainty(self, measurement_times: List[datetime]) -> List[float]:
        """Calculate expected uncertainty at each measurement time."""
        uncertainties = []
        
        # Simulate evidence accumulation
        temp_ledger = EvidenceLedger(self.ledger.alpha_prior, self.ledger.beta_prior)
        temp_ledger.alpha = self.ledger.alpha
        temp_ledger.beta = self.ledger.beta
        
        for measurement_time in measurement_times:
            # Simulate adding evidence (assume typical fidelity and confidence)
            expected_fidelity = temp_ledger.get_mean()
            expected_confidence = 0.95  # Assume high confidence measurements
            expected_shots = 1000  # Standard number of shots
            
            # Update temporary ledger
            temp_ledger.update_evidence(
                fidelity=expected_fidelity,
                confidence=expected_confidence,
                shots=expected_shots,
                experiment_id=f"simulated_{measurement_time.isoformat()}",
                timestamp=measurement_time,
            )
            
            # Record uncertainty
            uncertainties.append(temp_ledger.get_std())
        
        return uncertainties
    
    def _calculate_information_gain(self, measurement_times: List[datetime]) -> List[float]:
        """Calculate expected information gain at each measurement time."""
        information_gains = []
        
        # Simulate evidence accumulation
        temp_ledger = EvidenceLedger(self.ledger.alpha_prior, self.ledger.beta_prior)
        temp_ledger.alpha = self.ledger.alpha
        temp_ledger.beta = self.ledger.beta
        
        for measurement_time in measurement_times:
            # Calculate current entropy
            current_entropy = temp_ledger.get_entropy()
            
            # Simulate adding evidence
            expected_fidelity = temp_ledger.get_mean()
            expected_confidence = 0.95
            expected_shots = 1000
            
            temp_ledger.update_evidence(
                fidelity=expected_fidelity,
                confidence=expected_confidence,
                shots=expected_shots,
                experiment_id=f"simulated_{measurement_time.isoformat()}"
            )
            
            # Calculate new entropy
            new_entropy = temp_ledger.get_entropy()
            
            # Information gain is reduction in entropy
            information_gain = current_entropy - new_entropy
            information_gains.append(max(0, information_gain))
        
        return information_gains
    
    def _create_evenly_spaced_plan(self, n_measurements: int, 
                                 min_interval_hours: int) -> List[datetime]:
        """Create evenly spaced measurement plan."""
        start_time = datetime.now()
        end_time = start_time + timedelta(days=self.time_horizon_days)
        
        # Calculate total available time
        total_hours = self.time_horizon_days * 24
        min_total_hours = (n_measurements - 1) * min_interval_hours
        
        if n_measurements <= 0:
            return []

        if n_measurements == 1:
            return [start_time]

        if min_total_hours > total_hours:
            interval_hours = total_hours / (n_measurements - 1)
        else:
            interval_hours = ((total_hours - min_total_hours) / (n_measurements - 1)) + min_interval_hours

        measurement_times = []
        for i in range(n_measurements):
            measurement_time = start_time + timedelta(hours=i * interval_hours)
            if measurement_time > end_time:
                measurement_time = end_time
            measurement_times.append(measurement_time)

        return measurement_times
    
    def adaptive_cadence_plan(self, target_uncertainty: float, 
                            max_measurements: int = 100) -> CadencePlan:
        """
        Create adaptive cadence plan to achieve target uncertainty.
        
        Args:
            target_uncertainty: Target uncertainty level
            max_measurements: Maximum number of measurements
            
        Returns:
            CadencePlan with adaptive schedule
        """
        measurement_times = []
        expected_uncertainty = []
        information_gain = []
        
        # Start with current state
        temp_ledger = EvidenceLedger(self.ledger.alpha_prior, self.ledger.beta_prior)
        temp_ledger.alpha = self.ledger.alpha
        temp_ledger.beta = self.ledger.beta
        
        current_time = datetime.now()
        end_time = current_time + timedelta(days=self.time_horizon_days)
        
        while (len(measurement_times) < max_measurements and 
               current_time < end_time and 
               temp_ledger.get_std() > target_uncertainty):
            
            # Calculate optimal next measurement time
            next_time = self._find_optimal_next_measurement_time(
                current_time, end_time, temp_ledger
            )
            
            if next_time is None:
                break
            
            measurement_times.append(next_time)
            
            # Simulate measurement
            current_entropy = temp_ledger.get_entropy()
            expected_fidelity = temp_ledger.get_mean()
            
            temp_ledger.update_evidence(
                fidelity=expected_fidelity,
                confidence=0.95,
                shots=1000,
                experiment_id=f"adaptive_{next_time.isoformat()}",
                timestamp=next_time,
            )
            
            expected_uncertainty.append(temp_ledger.get_std())
            information_gain.append(current_entropy - temp_ledger.get_entropy())
            
            current_time = next_time + timedelta(hours=1)  # Minimum 1 hour between measurements
        
        total_cost = len(measurement_times) * self.measurement_cost
        
        return CadencePlan(
            measurement_times=measurement_times,
            expected_uncertainty=expected_uncertainty,
            information_gain=information_gain,
            total_cost=total_cost,
            optimization_objective='adaptive_uncertainty_target'
        )
    
    def _find_optimal_next_measurement_time(self, start_time: datetime, 
                                          end_time: datetime,
                                          current_ledger: EvidenceLedger) -> Optional[datetime]:
        """Find optimal time for next measurement."""
        # Simple heuristic: measure when uncertainty reduction is maximized
        # In practice, this could be more sophisticated
        
        current_uncertainty = current_ledger.get_std()
        if current_uncertainty < 0.01:  # Already very certain
            return None
        
        # Try different time intervals
        best_time = None
        best_reduction = 0
        
        for hours_ahead in range(1, min(24, int((end_time - start_time).total_seconds() / 3600))):
            candidate_time = start_time + timedelta(hours=hours_ahead)
            
            # Simulate measurement at this time
            temp_ledger = EvidenceLedger(current_ledger.alpha_prior, current_ledger.beta_prior)
            temp_ledger.alpha = current_ledger.alpha
            temp_ledger.beta = current_ledger.beta
            
            expected_fidelity = temp_ledger.get_mean()
            temp_ledger.update_evidence(
                fidelity=expected_fidelity,
                confidence=0.95,
                shots=1000,
                experiment_id=f"candidate_{candidate_time.isoformat()}"
            )
            
            uncertainty_reduction = current_uncertainty - temp_ledger.get_std()
            
            if uncertainty_reduction > best_reduction:
                best_reduction = uncertainty_reduction
                best_time = candidate_time
        
        return best_time
