"""Regression coverage for the canonical bundle-venv exports."""


def test_core_bundle_venv_exports_reuse_top_level_implementation() -> None:
    from extensions.sop_converter import bundle_venv as top_level_bundle_venv
    from extensions.sop_converter.core import bundle_venv_dir, bundle_venv_python

    assert bundle_venv_dir is top_level_bundle_venv.bundle_venv_dir
    assert bundle_venv_python is top_level_bundle_venv.bundle_venv_python
