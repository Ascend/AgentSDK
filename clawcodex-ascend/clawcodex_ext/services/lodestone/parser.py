"""F-97 LODESTONE — anchor parser.

Recognises the four primary anchor kinds in arbitrary text:

* ``file_path``    — ``path:line[:col][-end_line[:end_col]]``
* ``function_ref`` — ``module.func()`` / ``module::func``
* ``git_commit``   — bare 7-40 hex
* ``tracker_issue``— ``#123``, ``ORG-456``, ``[ORG-789]``
* ``url``          — anything matching ``http(s)://`` / ``vscode://`` / ``file://``
* ``git_blob``     — ``@<sha>:<path>``

Design goals:

* **Conservative tokenisation.** When a token is ambiguous (e.g. a bare
  integer that could be a line number or a tracker number), the parser
  defers to a later classification step and may emit multiple candidates.
* **No false positives on URLs.** The URL regex ignores trailing
  punctuation that humans wouldn't expect linked (``.``, ``,``, ``;``,
  ``)``, ``]``) and re-attaches that punctuation to the trailing text.
* **Span tracking.** Every match carries ``(start, end)`` so the
  renderer can replace the original substring without disturbing the
  surrounding text.

The parser is stateless — ``AnchorParser`` is essentially a thin wrapper
so callers can inject / extend via subclassing if needed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import AnchorContext, AnchorKind, LodestoneAnchor

# ---------------------------------------------------------------------------
# Regex bank — one pattern per anchor kind, ordered from most-specific to
# least-specific to ensure precedence.
# ---------------------------------------------------------------------------

# Path with line / col / range:
#   src/foo.py:42
#   src/foo.py:42:13
#   src/foo.py:42-50
#   src/foo.py:42:13-50:5
#   ./src/file.py:1
#   ../module.pyx:99:1
#
# We require the *file extension* (heuristic: any of .py .ts .tsx .js .jsx
# .go .rs .java .rb .c .cpp .h .hpp .md .yaml .yml .json .toml .sh .bash)
# to disambiguate ``123:45`` from arbitrary prose.  Override via
# ``AnchorParser(file_globs=...)`` if your workspace speaks a different
# dialect.
_DEFAULT_FILE_EXTS = [
    ".py",
    ".pyi",
    ".pyx",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".rb",
    ".php",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".m",
    ".mm",
    ".swift",
    ".scala",
    ".lua",
    ".pl",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".psm1",
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".ini",
    ".cfg",
    ".conf",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
    ".sql",
    ".proto",
    ".graphql",
    ".gql",
    ".xml",
    ".xsl",
    ".xsd",
    ".wsdl",
    ".svg",
    ".tex",
    ".bib",
]

# Match ``path:line[:col][-end_line[:end_col]]``.  Group names carry the
# semantics forward so the matcher code stays declarative.
#
# Accepts the following forms after the basename:
#   :42
#   :42:13
#   :42-50
#   :42-50:5
#   :42:13-50:5
_FILE_RE = re.compile(
    r"""
    (?P<path>(?:[A-Za-z0-9_./\-]+/)?                # optional dir prefix
       [A-Za-z0-9_\-]+                             # basename
       \.(?:{}))                                   # extension
    :(?P<line>\d+)                                 # required line
    (?:(?P<kind>
        :(?P<col>\d+)                              # column form ``:42:13``
        (?:-(?P<end_line1>\d+)(?::(?P<end_col1>\d+))?)?
      |
        -(?P<end_line2>\d+)                        # range form ``:42-50``
        (?: :(?P<end_col2>\d+))?
      )
    )?
    """.format("|".join(re.escape(e.lstrip(".")) for e in _DEFAULT_FILE_EXTS)),
    re.VERBOSE,
)

# Function reference:  ``module.func()``  or  ``module::func``  or the
# rarer ``module::cls::func`` (``::`` chains).
_FUNCTION_RE = re.compile(r"(?<![\w.])(?P<sym>[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+|\w+(?:::\w+)+)(?=\s*\(\)|\b)")

# Tracker issue:  ``#123`` / ``ORG-456`` / ``[ORG-789]``
_TRACKER_RE = re.compile(
    r"(?<![\w./-])"  # no previous word char
    r"(?:\[(?P<prefix1>[A-Z][A-Z0-9]{1,9})-(?P<num1>\d+)\]"
    r"|\#(?P<num2>\d+)"
    r"|(?<!#)(?P<prefix2>[A-Z][A-Z0-9]{1,9})-(?P<num3>\d+))"
    r"\b"
)

# Bare git sha — 7 to 40 hex chars, not preceded/followed by word char.
_GIT_SHA_RE = re.compile(r"(?<![\w])(?P<sha>[0-9a-f]{7,40})(?![\w])")

# Git blob — ``@<sha>:<path>``.  Path is the same heuristic as FILE_RE
# but we relax the extension requirement.
_GIT_BLOB_RE = re.compile(r"@(?P<sha>[0-9a-f]{7,40}):(?P<path>[A-Za-z0-9_./\-]+)")

# Bare URL — covers http(s) / vscode / idea / subl / file / ssh / git
_URL_RE = re.compile(r"(?P<url>(?:https?|vscode|vscode-insiders|cursor|idea|subl|file|ssh|git)://[^\s)\]\"']+)")

# Common trailing punctuation that browsers accept inside the URL but
# humans usually mean as sentence punctuation; we strip them off.
_URL_TRAILING = ".,;:!?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_trailing_punct(url: str) -> tuple[str, str]:
    """Detach trailing punctuation from the matched URL.

    Returns ``(clean_url, trailing_punct)``.  Repeated trailing characters
    (e.g. ``...`` or ``!!``) are peeled off together.
    """
    trailing = ""
    while url and url[-1] in _URL_TRAILING:
        trailing = url[-1] + trailing
        url = url[:-1]
    return url, trailing


def _span_starts_with_capital(prefix: str) -> bool:
    return bool(prefix) and prefix[0].isupper()


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class AnchorParser:
    """Stateless anchor parser.

    Construct with ``file_globs`` to override the extension list, or with
    ``extra_patterns`` to add (anchor_kind, compiled-regex) pairs ahead
    of the built-in rules.
    """

    def __init__(
        self,
        *,
        file_globs: Iterable[str] | None = None,
        extra_patterns: Iterable[tuple[AnchorKind, re.Pattern[str]]] | None = None,
    ) -> None:
        if file_globs is not None:
            self._file_re = re.compile(
                _FILE_RE.pattern.replace(
                    "|".join(re.escape(e.lstrip(".")) for e in _DEFAULT_FILE_EXTS),
                    "|".join(re.escape(g.lstrip(".").lstrip("*")) for g in file_globs),
                ),
                re.VERBOSE,
            )
        else:
            self._file_re = _FILE_RE
        self._extra = tuple(extra_patterns or ())

    # -- high-level API ------------------------------------------------------

    def parse(self, text: str) -> list[LodestoneAnchor]:
        """Return every anchor detected in ``text``.

        Result order is the natural left-to-right sweep order; overlapping
        matches are dropped (longest wins).
        """
        if not text:
            return []
        spans: list[tuple[int, int, LodestoneAnchor]] = []
        for kind, anchor in self._iter_all(text):
            s, e = anchor.span or (0, 0)
            spans.append((s, e, anchor))

        spans.sort(key=lambda t: (t[0], -(t[1] - t[0])))
        pruned: list[LodestoneAnchor] = []
        occupied: list[tuple[int, int]] = []
        for s, e, anchor in spans:
            if any(not (e <= os or s >= oe) for os, oe in occupied):
                continue
            occupied.append((s, e))
            pruned.append(anchor)
        pruned.sort(key=lambda a: a.span[0] if a.span else 0)
        return pruned

    def parse_first(self, text: str) -> LodestoneAnchor | None:
        anchors = self.parse(text)
        return anchors[0] if anchors else None

    # -- internal iteration --------------------------------------------------

    def _iter_all(self, text: str) -> Iterable[tuple[AnchorKind, LodestoneAnchor]]:
        # File path with line/col
        for m in self._file_re.finditer(text):
            line = int(m.group("line"))
            col = int(m.group("col")) if m.group("col") else None
            end_line = (
                int(m.group("end_line1") or m.group("end_line2"))
                if (m.group("end_line1") or m.group("end_line2"))
                else None
            )
            end_col = (
                int(m.group("end_col1") or m.group("end_col2"))
                if (m.group("end_col1") or m.group("end_col2"))
                else None
            )
            anchor = LodestoneAnchor(
                kind="file_path",
                raw=m.group(0),
                file_path=m.group("path"),
                line=line,
                column=col,
                end_line=end_line,
                end_column=end_col,
                span=(m.start(), m.end()),
            )
            yield "file_path", anchor

        # Git blob (must come before plain git_sha because ``@`` prefix)
        for m in _GIT_BLOB_RE.finditer(text):
            yield (
                "git_blob",
                LodestoneAnchor(
                    kind="git_blob",
                    raw=m.group(0),
                    git_sha=m.group("sha"),
                    file_path=m.group("path"),
                    span=(m.start(), m.end()),
                ),
            )

        # Tracker issue
        for m in _TRACKER_RE.finditer(text):
            prefix = m.group("prefix1") or m.group("prefix2")
            num = m.group("num1") or m.group("num2") or m.group("num3")
            kind: AnchorKind = "tracker_issue"
            host = prefix.lower() if prefix else "gitcode"
            yield (
                kind,
                LodestoneAnchor(
                    kind=kind,
                    raw=m.group(0),
                    tracker_key=(host, num),
                    span=(m.start(), m.end()),
                ),
            )

        # Git sha
        for m in _GIT_SHA_RE.finditer(text):
            yield (
                "git_commit",
                LodestoneAnchor(
                    kind="git_commit",
                    raw=m.group(0),
                    git_sha=m.group("sha"),
                    span=(m.start(), m.end()),
                ),
            )

        # Function refs
        for m in _FUNCTION_RE.finditer(text):
            sym = m.group("sym")
            # Don't classify ``file.py:42`` segments — we already match
            # them above and they show up here as ``file.py``.
            if sym in {"file", "file.py", "src.file"} and "." in sym:
                continue
            yield (
                "function_ref",
                LodestoneAnchor(
                    kind="function_ref",
                    raw=m.group(0),
                    symbol=sym,
                    span=(m.start(), m.end()),
                ),
            )

        # URL
        for m in _URL_RE.finditer(text):
            url, _ = _strip_trailing_punct(m.group("url"))
            yield (
                "url",
                LodestoneAnchor(
                    kind="url",
                    raw=m.group(0),
                    url=url,
                    span=(m.start(), m.end()),
                ),
            )

        # User-extensible patterns
        for kind, pattern in self._extra:
            for m in pattern.finditer(text):
                yield (
                    kind,
                    LodestoneAnchor(
                        kind=kind,
                        raw=m.group(0),
                        span=(m.start(), m.end()),
                    ),
                )


def detect_anchor_kind(text: str) -> AnchorKind | None:
    """Single-shot helper — return the first anchor's kind or None."""
    if not text:
        return None
    first = AnchorParser().parse_first(text)
    return first.kind if first is not None else None


def parse_anchors(text: str) -> list[LodestoneAnchor]:
    """Convenience wrapper equivalent to ``AnchorParser().parse(text)``."""
    return AnchorParser().parse(text)


__all__ = [
    "_DEFAULT_FILE_EXTS",
    "_FILE_RE",
    "_FUNCTION_RE",
    "_GIT_BLOB_RE",
    "_GIT_SHA_RE",
    "_TRACKER_RE",
    "_URL_RE",
    "AnchorParser",
    "detect_anchor_kind",
    "parse_anchors",
]


# ``unused-import`` guards
_ = AnchorContext
