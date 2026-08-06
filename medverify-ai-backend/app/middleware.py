import uuid
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

# Custom logging filter that provides default request_id if missing
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "SYSTEM"
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [req_id=%(request_id)s] %(message)s"
)

# Apply filter to root logger
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIDFilter())

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        
        logging.info(
            f"HTTP {request.method} {request.url.path} -> Status {response.status_code} ({process_time:.2f}ms)",
            extra={"request_id": request_id}
        )
        return response
