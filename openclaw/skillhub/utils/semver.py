"""Semantic versioning utilities."""

import re
from typing import Optional, Tuple


def parse_version(version: str) -> Tuple[int, int, int, Optional[str]]:
    """Parse semantic version string.

    Args:
        version: Version string like "1.2.3" or "1.2.3-beta"

    Returns:
        Tuple of (major, minor, patch, prerelease)
    """
    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?$"
    match = re.match(pattern, version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch, prerelease = match.groups()
    return int(major), int(minor), int(patch), prerelease


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.

    Args:
        v1: First version string
        v2: Second version string

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    major1, minor1, patch1, pre1 = parse_version(v1)
    major2, minor2, patch2, pre2 = parse_version(v2)

    if major1 != major2:
        return 1 if major1 > major2 else -1
    if minor1 != minor2:
        return 1 if minor1 > minor2 else -1
    if patch1 != patch2:
        return 1 if patch1 > patch2 else -1

    if pre1 is None and pre2 is None:
        return 0
    if pre1 is None:
        return 1
    if pre2 is None:
        return -1

    return 0 if pre1 == pre2 else (1 if pre1 > pre2 else -1)


def satisfies_version(version: str, spec: str) -> bool:
    """Check if version satisfies version specifier.

    Args:
        version: Version string to check
        spec: Version specifier like "^1.0.0", "~1.0.0", ">=1.0.0"

    Returns:
        True if version satisfies spec
    """
    if spec in ("*", "latest"):
        return True

    major, minor, patch, prerelease = parse_version(version)
    spec_clean = spec.lstrip("^~>=<")
    spec_major, spec_minor, spec_patch, spec_prerelease = parse_version(spec_clean)

    if prerelease is not None and spec_prerelease is None:
        return False

    if spec.startswith("^"):
        return major == spec_major and (minor > spec_minor or (minor == spec_minor and patch >= spec_patch))
    elif spec.startswith("~"):
        return major == spec_major and minor == spec_minor and patch >= spec_patch
    elif spec.startswith(">="):
        return compare_versions(version, spec.lstrip(">=")) >= 0
    elif spec.startswith("<="):
        return compare_versions(version, spec.lstrip("<=")) <= 0
    elif spec.startswith(">"):
        return compare_versions(version, spec.lstrip(">")) > 0
    elif spec.startswith("<"):
        return compare_versions(version, spec.lstrip("<")) < 0
    else:
        return version == spec
