import re

PII_PATTERNS = [re.compile(r"\b\d{3}-\d{2}-\d{4}\b")]  # Example: SSN-like


def redact_pii(text: str) -> str:
    if not text:
        return text
    redacted = text
    for pat in PII_PATTERNS:
        redacted = pat.sub("[REDACTED]", redacted)
    return redacted
