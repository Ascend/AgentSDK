"""Checksum utilities for integrity verification."""

import hashlib
from typing import Union


def compute_checksum(content: Union[str, bytes], algorithm: str = "sha256") -> str:
    """Compute checksum of content.

    Args:
        content: Content to hash
        algorithm: Hash algorithm (sha256, sha512, md5)

    Returns:
        Hex digest of checksum
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    if algorithm == "sha256":
        return hashlib.sha256(content).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(content).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(content, usedforsecurity=False).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def verify_checksum(content: Union[str, bytes], expected: str, algorithm: str = "sha256") -> bool:
    """Verify content checksum.

    Args:
        content: Content to verify
        expected: Expected checksum
        algorithm: Hash algorithm

    Returns:
        True if checksum matches
    """
    actual = compute_checksum(content, algorithm)
    return actual.lower() == expected.lower()


def compute_file_checksum(file_path: str, algorithm: str = "sha256") -> str:
    """Compute checksum of file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm

    Returns:
        Hex digest of checksum
    """
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "md5":
        hasher = hashlib.md5(usedforsecurity=False)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()
