"""
Command registry for Claw Codex.

Manages registration and lookup of commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from .types import Command


@dataclass
class CommandRegistry:
    """Registry for commands."""

    _commands: dict[str, Command] = field(default_factory=dict)

    def register(self, command: Command) -> None:
        """
        Register a command.

        Args:
            command: The command to register
        """
        name = command.name.lower()
        self._commands[name] = command

        # Register aliases as direct command entries so list_commands
        # and the slash-command display surface them as invocable names.
        for alias in command.aliases:
            alias_lower = alias.lower()
            if alias_lower not in self._commands:
                self._commands[alias_lower] = command

    def unregister(self, name: str) -> None:
        """
        Unregister a command and its aliases.

        Args:
            name: Name of the command to unregister
        """
        name_lower = name.lower()
        command = self._commands.get(name_lower)
        if command is None:
            return

        # Remove the primary name
        del self._commands[name_lower]

        # Remove all alias entries pointing to the same command object
        aliases_to_remove = [alias for alias, cmd in self._commands.items() if alias != name_lower and cmd is command]
        for alias in aliases_to_remove:
            del self._commands[alias]

    def get(self, name: str) -> Optional[Command]:
        """
        Get a command by name or alias.

        Args:
            name: Name or alias of the command

        Returns:
            The command, or None if not found
        """
        name_lower = name.lower()
        return self._commands.get(name_lower)

    def has(self, name: str) -> bool:
        """
        Check if a command exists.

        Args:
            name: Name or alias to check

        Returns:
            True if the command exists
        """
        return self.get(name) is not None

    def list_commands(
        self,
        include_hidden: bool = False,
        include_disabled: bool = False,
    ) -> list[Command]:
        """
        List all registered commands, each with its primary name only.

        Aliases are NOT returned as separate entries here — they are resolved
        at lookup time by :meth:`get`.  Callers that need to display aliases
        should read the ``aliases`` field on each returned command, or use
        :meth:`list_all_names` to get all invocable names.

        Args:
            include_hidden: Include hidden commands
            include_disabled: Include disabled commands

        Returns:
            List of commands
        """
        seen: set[int] = set()
        commands: list[Command] = []
        for cmd in self._commands.values():
            cmd_id = id(cmd)
            if cmd_id in seen:
                continue
            seen.add(cmd_id)
            if not include_hidden and cmd.is_hidden:
                continue
            if not include_disabled and not cmd.is_enabled():
                continue
            commands.append(cmd)

        return sorted(commands, key=lambda c: c.name.lower())

    def list_invocable_commands(
        self,
        include_hidden: bool = False,
        include_disabled: bool = False,
    ) -> list[Command]:
        """Return all invocable command entries, one per primary name AND alias.

        Unlike :meth:`list_commands`, this returns a separate entry for each
        alias so the slash-command display and autocomplete can surface
        ``/orch`` alongside ``/orchestrator``.  Each alias entry is a shallow
        copy with the alias name set; the underlying command object is shared.

        These entries are safe for display/autocomplete use.  Command execution
        should always go through :meth:`get` which returns the canonical entry.
        """
        seen_objs: set[int] = set()
        entries: list[Command] = []
        for key, cmd in self._commands.items():
            cmd_id = id(cmd)
            if cmd_id in seen_objs:
                continue
            seen_objs.add(cmd_id)

            # Primary name entry
            if not include_hidden and cmd.is_hidden:
                continue
            if not include_disabled and not cmd.is_enabled():
                continue
            entries.append(cmd)

            # Alias entries — shallow copy with alias name for display
            for alias in cmd.aliases:
                alias_lower = alias.lower()
                if alias_lower == key:
                    continue
                alias_entry = replace(cmd, name=alias_lower, aliases=[])
                entries.append(alias_entry)

        return sorted(entries, key=lambda c: c.name.lower())

    def list_all_names(self) -> list[str]:
        """Return all invocable names (primary names + aliases), lowercased.

        Useful for autocomplete and slash-command listing where aliases
        should appear as separate entries.
        """
        seen_objs: set[int] = set()
        names: list[str] = []
        for key, cmd in self._commands.items():
            cmd_id = id(cmd)
            if cmd_id not in seen_objs:
                seen_objs.add(cmd_id)
                names.append(key)
                for alias in cmd.aliases:
                    names.append(alias.lower())
        return sorted(names)

    def find_commands(self, query: str, limit: int = 20) -> list[Command]:
        """
        Find commands matching a query, deduplicated by object identity.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching commands
        """
        query_lower = query.lower()
        matches: list[tuple[int, Command]] = []
        seen: set[int] = set()

        for command in self._commands.values():
            cmd_id = id(command)
            if cmd_id in seen:
                continue
            seen.add(cmd_id)
            score = 0

            # Exact name match
            if query_lower == command.name.lower():
                score = 1000
            # Name starts with query
            elif command.name.lower().startswith(query_lower):
                score = 100
            # Query in name
            elif query_lower in command.name.lower():
                score = 50
            # Query in description
            elif query_lower in command.description.lower():
                score = 25
            # Query in aliases
            elif any(query_lower in alias.lower() for alias in command.aliases):
                score = 30

            if score > 0:
                matches.append((-score, command.name, command))  # Negative for ascending sort, name for tiebreaker

        # Sort by score (highest first), then name
        matches.sort()
        return [cmd for _, _, cmd in matches[:limit]]

    def clear(self) -> None:
        """Clear all registered commands."""
        self._commands.clear()


# Global registry instance
_REGISTRY = CommandRegistry()


def get_command_registry() -> CommandRegistry:
    """Get the global command registry."""
    return _REGISTRY


def register_command(command: Command) -> None:
    """Register a command in the global registry."""
    _REGISTRY.register(command)


def get_command(name: str) -> Optional[Command]:
    """Get a command from the global registry."""
    return _REGISTRY.get(name)


def has_command(name: str) -> bool:
    """Check if a command exists in the global registry."""
    return _REGISTRY.has(name)


def list_commands(
    include_hidden: bool = False,
    include_disabled: bool = False,
) -> list[Command]:
    """List commands from the global registry."""
    return _REGISTRY.list_commands(include_hidden, include_disabled)


def find_commands(query: str, limit: int = 20) -> list[Command]:
    """Find commands in the global registry."""
    return _REGISTRY.find_commands(query, limit)
