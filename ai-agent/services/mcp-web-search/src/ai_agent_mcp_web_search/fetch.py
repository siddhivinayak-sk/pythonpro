"""SSRF-safe URL fetching + readability extraction.

Security model (design doc, MCP deep dive §8):
- Only http/https schemes.
- Resolve the host and refuse private / loopback / link-local / reserved / multicast / unspecified
  addresses (blocks cloud metadata endpoints like 169.254.169.254 and internal services).
- Follow redirects manually, re-validating every hop (prevents redirect-based SSRF).
- Enforce a content-type allowlist and byte/char caps.

Note: pre-resolution IP checks do not fully defeat DNS-rebinding; for hostile environments, pin the
resolved IP at connect time or run behind an egress proxy. This is documented as a known limitation.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

ALLOWED_SCHEMES = {"http", "https"}

DEFAULT_ALLOWED_CONTENT_TYPES: tuple[str, ...] = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
    "application/json",
    "text/",
)

_HTML_TYPES = {"text/html", "application/xhtml+xml"}


class FetchError(Exception):
    """A URL could not be fetched (network, content type, size, redirects)."""


class SsrfError(FetchError):
    """A URL was refused because it targets a disallowed / internal address."""


@dataclass
class FetchResult:
    url: str
    status_code: int
    content_type: str
    text: str
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ip_is_blocked(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def validate_url(url: str, *, allow_private: bool = False) -> str:
    """Return the URL if safe to fetch, else raise ``SsrfError``/``FetchError``."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise SsrfError(f"scheme '{parsed.scheme or '(none)'}' is not allowed (http/https only)")
    if not parsed.hostname:
        raise SsrfError("URL has no host")

    if allow_private:
        return url

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"DNS resolution failed for '{parsed.hostname}': {exc}") from exc

    for info in infos:
        ip = str(info[4][0])
        if _ip_is_blocked(ip):
            raise SsrfError(f"host '{parsed.hostname}' resolves to blocked address {ip}")
    return url


def extract_text(html: str) -> str:
    """Extract readable main text from HTML (strip scripts/nav/ads; prefer <main>/<article>)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "aside",
            "header",
            "form",
            "svg",
            "template",
        ]
    ):
        tag.decompose()

    container = soup.find("main") or soup.find("article") or soup.body or soup
    text = container.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def safe_fetch(
    url: str,
    *,
    max_chars: int = 8000,
    max_bytes: int = 2_000_000,
    timeout: int = 20,
    max_redirects: int = 3,
    user_agent: str = "ai-agent-mcp-web-search/1.0",
    allow_private: bool = False,
    allowed_content_types: tuple[str, ...] = DEFAULT_ALLOWED_CONTENT_TYPES,
) -> FetchResult:
    """Fetch ``url`` safely and return extracted text. Raises ``FetchError``/``SsrfError`` on problems."""
    import httpx

    current = validate_url(url, allow_private=allow_private)
    headers = {"User-Agent": user_agent}

    with httpx.Client(follow_redirects=False, timeout=timeout, headers=headers) as client:
        for _ in range(max_redirects + 1):
            resp = client.get(current)

            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise FetchError("redirect response without a Location header")
                current = validate_url(urljoin(current, location), allow_private=allow_private)
                continue

            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if not any(content_type.startswith(prefix) for prefix in allowed_content_types):
                raise FetchError(f"unsupported content type '{content_type or '(unknown)'}'")

            raw = resp.content[:max_bytes]
            body = raw.decode(resp.encoding or "utf-8", errors="replace")
            text = extract_text(body) if content_type in _HTML_TYPES else body
            truncated = len(text) > max_chars
            return FetchResult(
                url=current,
                status_code=resp.status_code,
                content_type=content_type,
                text=text[:max_chars],
                truncated=truncated,
            )

    raise FetchError(f"exceeded max redirects ({max_redirects})")
