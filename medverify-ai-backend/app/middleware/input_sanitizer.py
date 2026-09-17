"""
MedVerify AI - Stage 12: Input Sanitizer Middleware

Implements input validation and sanitization:
1. Max claim length enforcement (2000 characters)
2. HTML/script tag stripping
3. Unicode normalization (NFKC)
4. Prompt injection defense (detects manipulation patterns)
"""

import re
import unicodedata
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Maximum allowed claim text length
MAX_CLAIM_LENGTH = 2000

# HTML/script tag pattern
HTML_TAG_PATTERN = re.compile(r'<[^>]+>', re.IGNORECASE)

# Prompt injection patterns — detect attempts to manipulate LLM behavior
PROMPT_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?prior\s+instructions",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+(if\s+)?(you\s+are\s+)?a",
        r"pretend\s+(to\s+be|you\s+are)",
        r"new\s+instructions?\s*:",
        r"system\s*prompt\s*:",
        r"override\s+(all\s+)?safety",
        r"bypass\s+(all\s+)?filters?",
        r"jailbreak",
        r"DAN\s+mode",
        r"\[SYSTEM\]",
        r"\[INST\]",
        r"<\|im_start\|>",
        r"<\|system\|>",
    ]
]


def sanitize_text(text: str) -> str:
    """
    Sanitize user input text:
    1. Unicode normalize (NFKC)
    2. Strip HTML/script tags
    3. Remove control characters (except newlines)
    4. Collapse excessive whitespace
    """
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Strip HTML tags
    text = HTML_TAG_PATTERN.sub("", text)

    # Remove control characters (keep newlines and tabs)
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t", "\r") or not unicodedata.category(ch).startswith("C")
    )

    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{3,}', '  ', text)

    return text.strip()


def detect_prompt_injection(text: str) -> bool:
    """
    Check if input text contains prompt injection patterns.
    Returns True if injection detected.
    """
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


class InputSanitizerMiddleware(BaseHTTPMiddleware):
    """
    Stage 12: Input sanitization middleware.

    Applied to all POST requests to /api/verifications/.
    Validates and sanitizes the claim text before it reaches the pipeline.
    """

    async def dispatch(self, request: Request, call_next):
        # Only intercept POST requests to verification endpoint
        if (
            request.method == "POST"
            and request.url.path.startswith("/api/verifications")
            and not request.url.path.endswith("/stream")
        ):
            try:
                # Read and parse body
                body = await request.body()
                if body:
                    import json
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        return JSONResponse(
                            status_code=400,
                            content={"detail": "Invalid JSON in request body."},
                        )

                    claim_text = data.get("claimText", "") or data.get("claim_text", "")

                    if claim_text:
                        # Check length before sanitization
                        if len(claim_text) > MAX_CLAIM_LENGTH:
                            logger.warning(
                                "[InputSanitizer] Claim too long: %d chars (max %d)",
                                len(claim_text), MAX_CLAIM_LENGTH
                            )
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "detail": f"Claim text exceeds maximum length of {MAX_CLAIM_LENGTH} characters.",
                                    "max_length": MAX_CLAIM_LENGTH,
                                    "received_length": len(claim_text),
                                },
                            )

                        # Check for prompt injection
                        if detect_prompt_injection(claim_text):
                            logger.warning(
                                "[InputSanitizer] Prompt injection detected in claim text."
                            )
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "detail": "Input contains invalid patterns and cannot be processed. "
                                              "Please submit a genuine medical claim for verification.",
                                },
                            )

            except Exception as e:
                logger.error("[InputSanitizer] Error processing request: %s", e)
                # Don't block on sanitizer errors — let the request pass through

        return await call_next(request)
