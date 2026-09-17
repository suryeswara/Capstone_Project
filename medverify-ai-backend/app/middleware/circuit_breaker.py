"""
MedVerify AI - Stage 12: Circuit Breaker Middleware

Implements the circuit breaker pattern for external API calls
(PubMed NCBI API, ClinicalTrials.gov API).

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests fail fast with fallback
- HALF_OPEN: Testing if the service has recovered

Configuration:
- Failure threshold: 3 consecutive failures
- Recovery timeout: 120 seconds
- Monitoring window: 60 seconds
"""

import time
import logging
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitConfig:
    """Configuration for a circuit breaker instance."""
    name: str
    failure_threshold: int = 3
    recovery_timeout_seconds: int = 120
    monitoring_window_seconds: int = 60


class CircuitBreaker:
    """
    Individual circuit breaker for a single external service.

    Usage:
        breaker = CircuitBreaker(CircuitConfig(name="pubmed"))
        if breaker.can_execute():
            try:
                result = call_pubmed_api()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                result = fallback_result()
        else:
            result = fallback_result()
    """

    def __init__(self, config: CircuitConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._last_state_change = time.time()
        self._consecutive_failures = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, auto-transitioning from OPEN to HALF_OPEN after recovery timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_state_change
            if elapsed >= self.config.recovery_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def can_execute(self) -> bool:
        """Check if the circuit allows a request to pass through."""
        current_state = self.state  # Triggers auto-transition
        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            return True  # Allow one test request
        else:  # OPEN
            return False

    def record_success(self):
        """Record a successful external call."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("[CircuitBreaker:%s] Recovery successful, closing circuit.", self.config.name)
            self._transition_to(CircuitState.CLOSED)
        self._consecutive_failures = 0

    def record_failure(self):
        """Record a failed external call."""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            logger.warning("[CircuitBreaker:%s] Recovery attempt failed, re-opening circuit.", self.config.name)
            self._transition_to(CircuitState.OPEN)
        elif self._consecutive_failures >= self.config.failure_threshold:
            logger.error(
                "[CircuitBreaker:%s] Failure threshold reached (%d/%d), opening circuit for %ds.",
                self.config.name, self._consecutive_failures,
                self.config.failure_threshold, self.config.recovery_timeout_seconds
            )
            self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        if new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0
        logger.info("[CircuitBreaker:%s] State transition: %s -> %s", self.config.name, old_state.value, new_state.value)

    def get_status(self) -> Dict:
        """Get current circuit breaker status for monitoring."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
            "last_failure_time": self._last_failure_time,
            "seconds_since_last_state_change": round(time.time() - self._last_state_change, 1),
        }


# ---------------------------------------------------------------------------
# PRE-CONFIGURED CIRCUIT BREAKERS
# ---------------------------------------------------------------------------

_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a named external service.

    Pre-configured services:
    - 'pubmed': PubMed NCBI E-utilities API
    - 'clinical_trials': ClinicalTrials.gov API v2
    """
    if service_name not in _circuit_breakers:
        configs = {
            "pubmed": CircuitConfig(
                name="pubmed",
                failure_threshold=3,
                recovery_timeout_seconds=120,
                monitoring_window_seconds=60,
            ),
            "clinical_trials": CircuitConfig(
                name="clinical_trials",
                failure_threshold=3,
                recovery_timeout_seconds=180,
                monitoring_window_seconds=60,
            ),
        }
        config = configs.get(service_name, CircuitConfig(name=service_name))
        _circuit_breakers[service_name] = CircuitBreaker(config)
        logger.info("[CircuitBreaker] Created breaker for service: %s", service_name)

    return _circuit_breakers[service_name]


def get_all_circuit_statuses() -> list:
    """Get status of all registered circuit breakers."""
    return [cb.get_status() for cb in _circuit_breakers.values()]
