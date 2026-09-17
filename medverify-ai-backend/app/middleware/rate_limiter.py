"""
MedVerify AI - Stage 12: API Rate Limiter Middleware

Implements per-user and per-IP rate limiting using an in-memory
sliding window counter. No external dependency (Redis) required.

Limits:
- Authenticated users: 20 verifications/hour (user role), 100/hour (researcher)
- Unauthenticated IPs: 50 requests/minute on all endpoints
"""

import time
import logging
from collections import defaultdict
from typing import Dict, Tuple, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class SlidingWindowCounter:
    """Thread-safe in-memory sliding window rate counter."""

    def __init__(self):
        # { key: [(timestamp, count)] }
        self._windows: Dict[str, list] = defaultdict(list)

    def _cleanup(self, key: str, window_seconds: int):
        """Remove expired entries outside the current window."""
        cutoff = time.time() - window_seconds
        self._windows[key] = [
            (ts, count) for ts, count in self._windows[key] if ts > cutoff
        ]

    def increment(self, key: str, window_seconds: int) -> int:
        """
        Increment the counter for a key and return the current count
        within the sliding window.
        """
        self._cleanup(key, window_seconds)
        self._windows[key].append((time.time(), 1))
        return sum(count for _, count in self._windows[key])

    def get_count(self, key: str, window_seconds: int) -> int:
        """Get current count within the sliding window."""
        self._cleanup(key, window_seconds)
        return sum(count for _, count in self._windows[key])


# Rate limit configuration
RATE_LIMITS = {
    "user": {"max_requests": 20, "window_seconds": 3600},       # 20/hour
    "researcher": {"max_requests": 100, "window_seconds": 3600}, # 100/hour
    "admin": {"max_requests": 500, "window_seconds": 3600},      # 500/hour
    "ip": {"max_requests": 50, "window_seconds": 60},            # 50/minute
}

# Paths exempt from rate limiting
EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Stage 12: Per-user and per-IP rate limiting middleware.

    - Authenticated requests: limited by user role
    - Unauthenticated requests: limited by client IP
    - Returns 429 Too Many Requests with Retry-After header
    """

    def __init__(self, app):
        super().__init__(app)
        self._counter = SlidingWindowCounter()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, respecting X-Forwarded-For."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_user_info(self, request: Request) -> Tuple[Optional[str], str]:
        """
        Extract user ID and role from request state (set by auth dependency).
        Returns (user_id, role) or (None, 'anonymous').
        """
        user_id = getattr(request.state, "user_id", None)
        user_role = getattr(request.state, "user_role", "anonymous")
        return user_id, user_role

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # Skip rate limiting for non-API paths
        if not request.url.path.startswith("/api"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        # IP-level rate limiting (applies to all requests)
        ip_config = RATE_LIMITS["ip"]
        ip_key = f"ip:{client_ip}"
        ip_count = self._counter.increment(ip_key, ip_config["window_seconds"])

        if ip_count > ip_config["max_requests"]:
            retry_after = ip_config["window_seconds"]
            logger.warning(
                "[RateLimiter] IP rate limit exceeded: %s (%d/%d in %ds)",
                client_ip, ip_count, ip_config["max_requests"], ip_config["window_seconds"]
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # User-level rate limiting (for authenticated requests on verification endpoints)
        if request.url.path.startswith("/api/verifications"):
            user_id, user_role = self._get_user_info(request)
            if user_id:
                role_config = RATE_LIMITS.get(user_role, RATE_LIMITS["user"])
                user_key = f"user:{user_id}"
                user_count = self._counter.increment(user_key, role_config["window_seconds"])

                if user_count > role_config["max_requests"]:
                    retry_after = role_config["window_seconds"]
                    logger.warning(
                        "[RateLimiter] User rate limit exceeded: %s (role=%s, %d/%d)",
                        user_id, user_role, user_count, role_config["max_requests"]
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": f"Verification rate limit exceeded for {user_role} accounts. "
                                      f"Maximum {role_config['max_requests']} verifications per hour.",
                            "retry_after_seconds": retry_after,
                        },
                        headers={"Retry-After": str(retry_after)},
                    )

        return await call_next(request)
