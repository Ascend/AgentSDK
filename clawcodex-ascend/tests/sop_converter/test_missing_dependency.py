#!/usr/bin/env python3
# coding=utf-8

"""Runtime repair for SDK imports that sop convert did not install."""

from __future__ import annotations

from pathlib import Path

import pytest

from extensions.sop_converter.missing_dependency import (
    MissingDependencyRepair,
    parse_missing_module,
    top_level_imported_modules,
)


def test_parse_missing_module_from_traceback() -> None:
    text = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 8, in <module>\n'
        "    from datasets import tqdm\n"
        "ModuleNotFoundError: No module named 'datasets'\n"
    )
    assert parse_missing_module(text) == "datasets"


def test_parse_missing_module_keeps_dotted_namespace() -> None:
    from extensions.sop_converter.missing_dependency import pip_name_for_missing_module

    text = "ModuleNotFoundError: No module named 'opentelemetry.sdk'\n"
    assert parse_missing_module(text) == "opentelemetry.sdk"
    assert pip_name_for_missing_module("opentelemetry.sdk") == "opentelemetry-sdk"
    assert pip_name_for_missing_module("opentelemetry.sdk.trace") == "opentelemetry-sdk"
    assert pip_name_for_missing_module("sklearn.metrics") == "scikit-learn"


def test_parse_missing_module_from_wrapper_json_error() -> None:
    text = '{"error": "No module named \'opentelemetry.sdk\'"}'
    assert parse_missing_module(text) == "opentelemetry.sdk"


def test_wrapper_bundle_dir_reads_bootstrap_assignment(tmp_path: Path) -> None:
    from extensions.sop_converter.missing_dependency import wrapper_bundle_dir

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    wrapper = tmp_path / "app_fn.py"
    wrapper.write_text(
        f'_BUNDLE_DIR = _normalize_bootstrap_path(r"{bundle}")\n',
        encoding="utf-8",
    )
    assert wrapper_bundle_dir(wrapper) == bundle.resolve()


def test_top_level_imported_modules_reads_from_import_and_from(tmp_path: Path) -> None:
    source = tmp_path / "app.py"
    source.write_text(
        "import os\n"
        "from datasets import tqdm\n"
        "from mineru.cli.common import read_fn\n"
        "from mx_rag.document import LoaderMng\n",
        encoding="utf-8",
    )
    names = top_level_imported_modules(source)
    assert "datasets" in names
    assert "mineru" in names
    assert "mx_rag" in names
    assert "os" in names


def test_repair_installs_pip_package_into_bundle_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source = tmp_path / "app.py"
    source.write_text("from datasets import tqdm\n", encoding="utf-8")
    installed: list[str] = []

    monkeypatch.setattr(
        "extensions.sop_converter.missing_dependency.install_bundle_packages",
        lambda _bundle, packages: installed.extend(packages),
    )
    monkeypatch.setattr(
        "extensions.sop_converter.missing_dependency.append_sdk_requirement",
        lambda _bundle, package: None,
    )

    outcome = MissingDependencyRepair(bundle_dir).repair(
        "datasets",
        source_file=source,
    )

    assert outcome.installed is True
    assert outcome.error_code is None
    assert installed == ["datasets"]


def test_repair_installs_opentelemetry_sdk_for_dotted_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source = tmp_path / "span.py"
    source.write_text("from opentelemetry.sdk.trace import ReadableSpan\n", encoding="utf-8")
    installed: list[str] = []

    monkeypatch.setattr(
        "extensions.sop_converter.missing_dependency.install_bundle_packages",
        lambda _bundle, packages: installed.extend(packages),
    )
    monkeypatch.setattr(
        "extensions.sop_converter.missing_dependency.append_sdk_requirement",
        lambda _bundle, package: None,
    )

    outcome = MissingDependencyRepair(bundle_dir).repair(
        "opentelemetry.sdk",
        source_file=source,
    )

    assert outcome.installed is True
    assert installed == ["opentelemetry-sdk"]


def test_repair_installs_transitive_import_not_in_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source = tmp_path / "loop_coordinator.py"
    source.write_text(
        "from openjiuwen.harness.schema.stop_condition import StopConditionEvaluator\n",
        encoding="utf-8",
    )
    installed: list[str] = []

    monkeypatch.setattr(
        "extensions.sop_converter.missing_dependency.install_bundle_packages",
        lambda _bundle, packages: installed.extend(packages),
    )
    monkeypatch.setattr(
        "extensions.sop_converter.missing_dependency.append_sdk_requirement",
        lambda _bundle, package: None,
    )

    outcome = MissingDependencyRepair(bundle_dir).repair(
        "opentelemetry.sdk",
        source_file=source,
    )

    assert outcome.installed is True
    assert installed == ["opentelemetry-sdk"]


def test_append_sdk_requirement_keeps_convert_sdk_requirements(tmp_path: Path) -> None:
    from extensions.sop_converter.core.bundle_manifest import (
        read_bundle_manifest,
        write_bundle_manifest,
    )
    from extensions.sop_converter.missing_dependency import append_sdk_requirement

    bundle_dir = tmp_path / "bundle"
    sdk_dir = tmp_path / "sdk"
    bundle_dir.mkdir()
    sdk_dir.mkdir()
    convert_reqs = ("aiohttp>=3.11.12", "pyoxigraph>=0.5.8")
    write_bundle_manifest(
        bundle_dir,
        sdk_source_dir=sdk_dir,
        sdk_requirements=convert_reqs,
    )

    append_sdk_requirement(bundle_dir, "opentelemetry-sdk")
    append_sdk_requirement(bundle_dir, "opentelemetry-sdk")

    manifest = read_bundle_manifest(bundle_dir)
    assert manifest is not None
    assert manifest.sdk_requirements == convert_reqs
    assert manifest.runtime_extra_requirements == ("opentelemetry-sdk",)


def test_repair_returns_install_failed_when_pip_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source = tmp_path / "app.py"
    source.write_text("from mx_rag.document import LoaderMng\n", encoding="utf-8")

    def _fail(_bundle, _packages):
        raise RuntimeError("No matching distribution found for mx_rag")

    monkeypatch.setattr(
        "extensions.sop_converter.missing_dependency.install_bundle_packages",
        _fail,
    )

    outcome = MissingDependencyRepair(bundle_dir).repair(
        "mx_rag",
        source_file=source,
    )

    assert outcome.installed is False
    assert outcome.error_code == "sdk_dependency_install_failed"
    assert "mx_rag" in outcome.message
