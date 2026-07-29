"""WebBrowserTool — lightweight HTTP page fetcher.

Mirrors ``claude-code-best/packages/builtin-tools/.../WebBrowserTool.ts``.
This is NOT a full browser engine: there is no JavaScript execution, no
click/type/scroll. It performs an HTTP fetch and extracts the page title
plus stripped text content. The ``screenshot`` action returns a text
snapshot (visual screenshots would require a real browser runtime, which
is out of scope for the builtin tool).

Uses the standard-library ``urllib`` to avoid introducing a ``playwright``
dependency (F-71 listed ``playwright`` as optional). SSRF protection is
delegated to the same private-host guard used by ``WebFetchTool`` so the
two tools stay consistent.

Read-only; NOT concurrency-safe (network I/O, mutable cache).
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..build_tool import Tool, build_tool
from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..protocol import ToolResult

# Reuse the SSRF / redirect guards already shipped with WebFetchTool so the
# two HTTP-fetching tools stay in lock-step. Importing lazily would complicate
# the module; a direct import keeps the contract explicit.
from .web_fetch import _is_private_host, _validate_url


_TITLE_RE = re.compile(r"<title[^>]*>([^<]*)</title>", re.IGNORECASE)
_SCRIPT_RE = re.compile(r"<script[\s\S]*?</script>", re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[\s\S]*?</style>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_MAX_CONTENT_CHARS = 50_000
_MAX_FETCH_SECONDS = 30

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


def _extract_title(html_text: str) -> str:
    m = _TITLE_RE.search(html_text)
    return m.group(1).strip() if m and m.group(1) else ""


def _html_to_text(html_text: str) -> str:
    """Strip scripts/styles/tags and collapse whitespace."""
    text = _SCRIPT_RE.sub("", html_text)
    text = _STYLE_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _truncate(text: str) -> str:
    if len(text) > _MAX_CONTENT_CHARS:
        return text[:_MAX_CONTENT_CHARS] + "\n[truncated]"
    return text


def _fetch(url: str) -> tuple[int, str, str]:
    """Return ``(status, final_url, html)``. Raises on transport error."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": _ACCEPT,
        },
    )
    with urllib.request.urlopen(req, timeout=_MAX_FETCH_SECONDS) as resp:  # noqa: S310 — validated URL
        body = resp.read()
        encoding = resp.headers.get_content_charset() or "utf-8"
        try:
            text = body.decode(encoding, errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")
        return resp.status, resp.geturl(), text


def _web_browser_call(tool_input: dict[str, Any], _context: ToolContext) -> ToolResult:
    raw_url = tool_input.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ToolInputError("url must be a non-empty string")

    action = tool_input.get("action") or "navigate"
    if action not in ("navigate", "screenshot"):
        raise ToolInputError(f"unknown action: {action!r}")

    # Validate + reject private hosts (SSRF guard shared with WebFetchTool).
    try:
        url = _validate_url(raw_url)
    except (ToolPermissionError, ValueError, urllib.error.URLError) as exc:
        return ToolResult(
            name="WebBrowser",
            output={
                "title": "Error",
                "url": raw_url,
                "content": f"Invalid URL: {exc}",
            },
            is_error=True,
        )
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname and _is_private_host(parsed.hostname):
        return ToolResult(
            name="WebBrowser",
            output={
                "title": "Error",
                "url": url,
                "content": "Refusing to fetch private/internal host.",
            },
            is_error=True,
        )

    try:
        _status, final_url, html_text = _fetch(url)
    except urllib.error.HTTPError as exc:
        return ToolResult(
            name="WebBrowser",
            output={
                "title": f"HTTP {exc.code}",
                "url": url,
                "content": f"Error: {exc.code} {exc.reason}",
            },
            is_error=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface any fetch failure to the model
        return ToolResult(
            name="WebBrowser",
            output={
                "title": "Error",
                "url": url,
                "content": f"Failed to fetch: {exc}",
            },
            is_error=True,
        )

    title = _extract_title(html_text)
    text_content = _truncate(_html_to_text(html_text))

    if action == "screenshot":
        content = "[Text snapshot — visual screenshots require a full browser runtime]\n\n" + text_content
    else:
        content = text_content

    return ToolResult(
        name="WebBrowser",
        output={
            "title": title,
            "url": final_url or url,
            "content": content,
        },
    )


def _map_result_to_api(output: Any, tool_use_id: str) -> dict[str, Any]:
    if isinstance(output, dict):
        parts = [f"{output.get('title', '')} ({output.get('url', '')})"]
        if output.get("content"):
            parts.append(str(output["content"]))
        content: str | list[dict[str, Any]] = "\n".join(parts)
    else:
        content = str(output)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


WebBrowserTool: Tool = build_tool(
    name="WebBrowser",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch and extract content from.",
            },
            "action": {
                "type": "string",
                "enum": ["navigate", "screenshot"],
                "description": (
                    "Action to perform. 'navigate' (default) fetches page "
                    "content. 'screenshot' returns a text snapshot of the "
                    "page (not a visual screenshot)."
                ),
            },
        },
        "required": ["url"],
    },
    call=_web_browser_call,
    prompt=(
        "WebBrowser: fetch web pages via HTTP and extract their text content. "
        "This is a lightweight browser tool (HTTP fetch, not a full browser "
        "engine).\n\n"
        "Supported actions:\n"
        "- navigate: fetch a URL and extract page title + text content\n"
        "- screenshot: same as navigate (returns a text snapshot, not a "
        "visual screenshot)\n\n"
        "Limitations:\n"
        "- No JavaScript execution — only sees server-rendered HTML\n"
        "- click/type/scroll require a full browser runtime (not available)\n\n"
        "Use this for reading web page content, documentation, or HTML API "
        "endpoints. Read-only; not concurrency-safe."
    ),
    description="Fetch and read web page content via HTTP.",
    search_hint="web browser navigate url page screenshot",
    aliases=("browser", "web_page"),
    max_result_size_chars=100_000,
    should_defer=True,
    is_read_only=lambda _input: True,
    is_concurrency_safe=lambda _input: False,
    map_result_to_api=_map_result_to_api,
    to_auto_classifier_input=lambda input_data: (
        f"WebBrowser {input_data.get('action', 'navigate')} {input_data.get('url', '')}"
    ),
)
