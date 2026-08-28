"""Pure SDK fixture — no workflow patterns."""


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Calculator:
    """Simple calculator."""

    def multiply(self, x: int, y: int) -> int:
        """Multiply."""
        return x * y
