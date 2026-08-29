#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import inspect
import logging
import os
import re
import secrets
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from .model import Skill

logger = logging.getLogger(__name__)

LoadedFrom = str

VALID_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_:-]{0,63}$")
VALID_CONTEXTS = frozenset({"inline", "fork"})


@dataclass
class SkillValidationError:
    field: str
    message: str


@dataclass
class BundledSkillDefinition:
    name: str
    description: str
    get_prompt_for_command: Callable[..., str]
    aliases: list[str] = field(default_factory=list)
    when_to_use: str | None = None
    argument_hint: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    effort: str | int | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    is_enabled: Callable[[], bool] | None = None
    context: str = "inline"
    agent: str | None = None
    hooks: dict[str, Any] | None = None
    files: dict[str, str] | None = None
    requires_resources: bool = False


_bundled_skills: list[Skill] = []

# One unpredictable, owner-private directory per process. A skill's path is
# assigned at registration, while its directory and files remain lazy.
_bundled_skills_root: Path | None = None
_bundled_skills_root_lock = threading.Lock()

# Core initialization calls back into register_bundled_skill, hence RLock.
_registry_lock = threading.RLock()
_registry_condition = threading.Condition(_registry_lock)
_LAZY_INITIALIZED: bool = False
_LAZY_INITIALIZING: bool = False

_BUNDLED_VERSION = "dev"
for _distribution_name in ("clawcodex-dev-mind", "clawcodex"):
    try:
        _BUNDLED_VERSION = metadata.version(_distribution_name)
        break
    except metadata.PackageNotFoundError:
        continue
_BUNDLED_VERSION = re.sub(r"[^a-zA-Z0-9_.-]", "_", _BUNDLED_VERSION)
_PROCESS_NONCE = secrets.token_hex(16)


def _chmod_private(path: Path, mode: int) -> None:
    """Apply POSIX owner-only modes, tolerating Windows ACL semantics."""

    if os.name == "nt":
        return
    os.chmod(path, mode)


def _directory_create_mode() -> int:
    return 0o777 if os.name == "nt" else 0o700


def _file_create_mode() -> int:
    return 0o666 if os.name == "nt" else 0o600


def _create_bundled_skills_root() -> Path:
    """Create ``clawcodex/bundled-skills/<version>/<nonce>`` safely."""
    parent = Path(tempfile.gettempdir())
    for component in ("clawcodex", "bundled-skills", _BUNDLED_VERSION):
        parent = parent / component
        _ensure_private_directory(parent)
    process_root = parent / _PROCESS_NONCE
    _ensure_private_directory(process_root)
    return process_root


def get_bundled_skills_root() -> str:
    """Return the process-private root used for lazily extracted files."""

    global _bundled_skills_root
    if _bundled_skills_root is not None:
        return str(_bundled_skills_root)
    with _bundled_skills_root_lock:
        if _bundled_skills_root is None:
            root = _create_bundled_skills_root()
            _chmod_private(root, 0o700)
            _bundled_skills_root = root
    return str(_bundled_skills_root)


def get_bundled_skill_extract_dir(
    skill_name: str,
    *,
    registration_nonce: str | None = None,
) -> str:
    """Allocate a private path for one registered skill definition."""

    identity = skill_name if registration_nonce is None else f"{skill_name}:{registration_nonce}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    readable = re.sub(r"[^a-zA-Z0-9_.-]", "_", skill_name).strip("._-")
    readable = (readable or "skill")[:32]
    return str(Path(get_bundled_skills_root()) / f"{readable}-{digest}")


def is_bundled_skill_path(path: str | os.PathLike[str]) -> bool:
    """Return whether *path* resolves inside the private bundled root."""

    try:
        root = Path(get_bundled_skills_root()).resolve(strict=True)
        candidate = Path(path).expanduser().resolve(strict=False)
        candidate.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _lazy_init() -> None:
    """Seed the core catalogue once without inverting init/registry locks."""

    global _LAZY_INITIALIZED, _LAZY_INITIALIZING
    with _registry_condition:
        while _LAZY_INITIALIZING:
            _registry_condition.wait()
        if _LAZY_INITIALIZED:
            return
        _LAZY_INITIALIZING = True

    succeeded = False
    try:
        bundled = importlib.import_module(f"{__package__}.bundled")
        init_bundled_skills = getattr(bundled, "init_bundled_skills")
        succeeded = init_bundled_skills()
    except Exception:
        # Bundled skills must not make every SkillTool call unusable. Keep
        # any extension/core entries registered before the failure.
        logger.exception("failed to initialize bundled skills")
    finally:
        with _registry_condition:
            _LAZY_INITIALIZING = False
            _LAZY_INITIALIZED = succeeded
            _registry_condition.notify_all()


def validate_skill_definition(
    definition: BundledSkillDefinition,
) -> list[SkillValidationError]:
    errors: list[SkillValidationError] = []
    if not definition.name or not definition.name.strip():
        errors.append(SkillValidationError("name", "Skill name is required"))
    elif not VALID_NAME_RE.match(definition.name):
        errors.append(
            SkillValidationError(
                "name",
                f"Skill name '{definition.name}' must match pattern: "
                f"start with letter, contain only [a-zA-Z0-9_:-], max 64 chars",
            )
        )

    if not definition.description or not definition.description.strip():
        errors.append(SkillValidationError("description", "Skill description is required"))
    if not callable(definition.get_prompt_for_command):
        errors.append(SkillValidationError("get_prompt_for_command", "Prompt builder is required"))
    if definition.requires_resources and not definition.files:
        errors.append(
            SkillValidationError(
                "files",
                "Resource-dependent bundled skills must declare at least one file",
            )
        )

    if definition.context not in VALID_CONTEXTS:
        errors.append(
            SkillValidationError(
                "context",
                f"Invalid context '{definition.context}', must be one of: {', '.join(sorted(VALID_CONTEXTS))}",
            )
        )

    seen_aliases: set[str] = set()
    for alias in definition.aliases:
        if not alias or not alias.strip():
            errors.append(SkillValidationError("aliases", "Alias cannot be empty"))
            continue
        if not VALID_NAME_RE.match(alias):
            errors.append(SkillValidationError("aliases", f"Invalid alias: {alias}"))
            continue
        if alias == definition.name:
            errors.append(SkillValidationError("aliases", "Alias duplicates canonical name"))
            continue
        if alias in seen_aliases:
            errors.append(SkillValidationError("aliases", f"Duplicate alias: {alias}"))
            continue
        seen_aliases.add(alias)

    return errors


def validate_skill(skill: Skill) -> list[SkillValidationError]:
    errors: list[SkillValidationError] = []
    if not skill.name or not skill.name.strip():
        errors.append(SkillValidationError("name", "Skill name is required"))
    if not skill.description or not skill.description.strip():
        errors.append(SkillValidationError("description", "Skill description is required"))
    if skill.context not in VALID_CONTEXTS:
        errors.append(SkillValidationError("context", f"Invalid context '{skill.context}'"))
    return errors


def skill_from_mcp_tool(
    server_name: str,
    tool_name: str,
    tool_description: str,
    *,
    input_schema: dict[str, Any] | None = None,
) -> Skill:
    skill_name = f"mcp:{server_name}:{tool_name}"

    argument_hint = None
    argument_names: list[str] = []
    if input_schema and "properties" in input_schema:
        props = input_schema["properties"]
        required = set(input_schema.get("required", []))
        hints: list[str] = []
        for prop_name in props:
            if prop_name in required:
                hints.append(f"<{prop_name}>")
            else:
                hints.append(f"[{prop_name}]")
            argument_names.append(prop_name)
        argument_hint = " ".join(hints)

    def get_prompt(args: str) -> str:
        parts = [f"Use the MCP tool '{tool_name}' from server '{server_name}'."]
        if tool_description:
            parts.append(f"Tool description: {tool_description}")
        if args:
            parts.append(f"Arguments: {args}")
        return "\n".join(parts)

    return Skill(
        name=skill_name,
        description=tool_description or f"MCP tool {tool_name} from {server_name}",
        content="",
        source=f"mcp:{server_name}",
        loaded_from="mcp",
        user_invocable=True,
        allowed_tools=[f"mcp__{server_name}__{tool_name}"],
        argument_hint=argument_hint,
        argument_names=argument_names,
        get_prompt_for_command=get_prompt,
    )


def _builder_accepts_context(builder: Callable[..., str]) -> bool:
    try:
        signature = inspect.signature(builder)
        signature.bind("", None)
    except (TypeError, ValueError):
        return False
    return True


def _invoke_prompt_builder(
    builder: Callable[..., str],
    accepts_context: bool,
    args: str,
    context: Any | None,
) -> str:
    if accepts_context:
        return builder(args, context)
    return builder(args)


def _resolve_skill_file_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve a bundled relative path, rejecting every escape spelling."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("bundled skill file path must be a non-empty string")

    windows_path = PureWindowsPath(relative_path)
    posix_path = PurePosixPath(relative_path.replace("\\", "/"))
    if any(
        (
            windows_path.is_absolute(),
            bool(windows_path.drive),
            posix_path.is_absolute(),
            ".." in posix_path.parts,
            not posix_path.parts,
            posix_path == PurePosixPath("."),
        )
    ):
        raise ValueError(f"bundled skill file path escapes skill dir: {relative_path}")

    target = base_dir.joinpath(*posix_path.parts)
    try:
        target.resolve(strict=False).relative_to(base_dir.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"bundled skill file path escapes skill dir: {relative_path}") from exc
    return target


def _ensure_private_directory(path: Path) -> None:
    try:
        os.mkdir(path, _directory_create_mode())
    except FileExistsError:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OSError(f"unsafe bundled skill directory: {path}")
    _chmod_private(path, 0o700)


def _write_exclusive_private_file(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow:
        flags |= no_follow
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    descriptor = os.open(path, flags, _file_create_mode())
    try:
        payload = content.encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
        _chmod_private(path, 0o600)
    finally:
        os.close(descriptor)


def _extract_bundled_skill_files(
    skill_name: str,
    files: dict[str, str],
    skill_root: str,
) -> str | None:
    """Safely extract bundled files once; return None on every failure."""

    skill_dir = Path(skill_root)
    try:
        if not is_bundled_skill_path(skill_dir):
            raise ValueError(f"bundled skill root escaped process root: {skill_dir}")

        resolved_files: list[tuple[Path, str]] = []
        identities: set[str] = set()
        for relative_path, content in files.items():
            if not isinstance(content, str):
                raise TypeError(f"bundled skill file content must be text: {relative_path}")
            target = _resolve_skill_file_path(skill_dir, relative_path)
            identity = os.path.normcase(str(target.resolve(strict=False)))
            if identity in identities:
                raise ValueError(f"duplicate bundled skill file path: {relative_path}")
            identities.add(identity)
            resolved_files.append((target, content))

        try:
            os.mkdir(skill_dir, _directory_create_mode())
            _chmod_private(skill_dir, 0o700)
        except FileExistsError as exc:
            raise OSError(f"bundled skill extraction directory already exists: {skill_dir}") from exc

        for target, content in resolved_files:
            current = skill_dir
            relative = target.relative_to(skill_dir)
            for part in relative.parts[:-1]:
                current = current / part
                _ensure_private_directory(current)
            _write_exclusive_private_file(target, content)
        return str(skill_dir)
    except Exception as exc:  # noqa: BLE001 - bundled files are fail-open
        logger.warning("failed to extract files for bundled skill %s: %s", skill_name, exc)
        return None


def _build_prompt_wrapper(
    definition: BundledSkillDefinition,
    skill_root: str | None,
) -> Callable[..., str]:
    builder = definition.get_prompt_for_command
    accepts_context = _builder_accepts_context(builder)
    files = dict(definition.files or {})
    extraction_lock = threading.Lock()
    extraction_attempted = False
    extracted_dir: str | None = None

    def get_prompt(args: str, context: Any | None = None) -> str:
        nonlocal extraction_attempted, extracted_dir
        prompt = _invoke_prompt_builder(builder, accepts_context, args, context)
        if not files or skill_root is None:
            return prompt

        if not extraction_attempted:
            with extraction_lock:
                if not extraction_attempted:
                    extracted_dir = _extract_bundled_skill_files(
                        definition.name,
                        files,
                        skill_root,
                    )
                    extraction_attempted = True
        setattr(get_prompt, "_bundled_resource_root", extracted_dir)
        diagnostic = (
            None
            if extracted_dir is not None
            else (
                f"Bundled skill {definition.name!r} resources could not be extracted; "
                "continuing without a base directory"
            )
        )
        setattr(get_prompt, "_bundled_resource_diagnostic", diagnostic)
        if extracted_dir is None:
            if definition.requires_resources:
                raise RuntimeError(f"Required bundled resources for skill {definition.name!r} could not be extracted")
            return prompt
        return f"Base directory for this skill: {extracted_dir}\n\n{prompt}"

    setattr(get_prompt, "_bundled_resource_root", None)
    setattr(get_prompt, "_bundled_resource_diagnostic", None)
    return get_prompt


def register_bundled_skill(definition: BundledSkillDefinition) -> bool:
    """Register or replace one bundled skill without changing core init state."""
    errors = validate_skill_definition(definition)
    if errors:
        for error in errors:
            logger.warning("bundled skill %r rejected: %s: %s", definition.name, error.field, error.message)
        return False

    skill_root = (
        get_bundled_skill_extract_dir(
            definition.name,
            registration_nonce=secrets.token_hex(16),
        )
        if definition.files
        else None
    )
    skill = Skill(
        name=definition.name,
        description=definition.description,
        content="",
        source="bundled",
        loaded_from="bundled",
        aliases=list(definition.aliases),
        allowed_tools=list(definition.allowed_tools),
        argument_hint=definition.argument_hint,
        when_to_use=definition.when_to_use,
        model=definition.model,
        effort=definition.effort,
        disable_model_invocation=definition.disable_model_invocation,
        user_invocable=definition.user_invocable,
        context=definition.context,
        agent=definition.agent,
        hooks=dict(definition.hooks) if definition.hooks is not None else None,
        get_prompt_for_command=_build_prompt_wrapper(definition, skill_root),
        is_enabled_fn=definition.is_enabled,
        is_hidden=not definition.user_invocable,
        has_user_specified_description=True,
        skill_root=skill_root,
    )

    global _LAZY_INITIALIZED
    with _registry_lock:
        for index, existing in enumerate(_bundled_skills):
            if existing.name == skill.name:
                _bundled_skills[index] = skill
                break
        else:
            _bundled_skills.append(skill)
        # An explicit registration is a deliberate catalogue seed. Suppress
        # automatic core expansion until clear_bundled_skills() re-arms it;
        # registrations performed by _lazy_init itself keep its in-progress
        # state and are finalized by _lazy_init after all registrars run.
        if not _LAZY_INITIALIZING:
            _LAZY_INITIALIZED = True

    _invalidate_catalog_cache()
    return True


def get_registered_bundled_skills() -> list[Skill]:
    """Return current registrations without triggering core initialization."""

    with _registry_lock:
        return list(_bundled_skills)


def get_bundled_skills() -> list[Skill]:
    _lazy_init()
    return get_registered_bundled_skills()


def get_bundled_skill_by_name(name: str) -> Skill | None:
    """Resolve a bundled name with canonical names ahead of every alias."""

    _lazy_init()
    with _registry_lock:
        for skill in _bundled_skills:
            if skill.name == name:
                return skill
        for skill in _bundled_skills:
            if name in skill.aliases:
                return skill
    return None


def clear_bundled_skills() -> None:
    """Wipe registrations and re-arm core lazy initialization."""

    global _LAZY_INITIALIZED, _LAZY_INITIALIZING
    with _registry_lock:
        _bundled_skills.clear()
        try:
            bundled = importlib.import_module(f"{__package__}.bundled")
            reset_bundled_skills_init_flag = getattr(bundled, "reset_bundled_skills_init_flag")
            reset_bundled_skills_init_flag()
        except Exception:
            logger.debug("Failed to reset bundled skill init flag", exc_info=True)
        _LAZY_INITIALIZED = False
        _LAZY_INITIALIZING = False

    _invalidate_catalog_cache()


def _invalidate_catalog_cache() -> None:
    """Invalidate the optional catalog without creating an import-graph cycle."""

    try:
        catalog = importlib.import_module(f"{__package__}.catalog")
    except ImportError:  # pragma: no cover - package import boundary
        return
    invalidate = getattr(catalog, "_invalidate_catalog_cache_only", None)
    if callable(invalidate):
        invalidate()
