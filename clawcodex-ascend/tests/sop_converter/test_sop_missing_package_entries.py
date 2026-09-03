"""Regression tests for migrated SOP converter package entry points."""


def test_package_entry_initializes_default_adapters():
    from extensions import sop_converter

    assert sop_converter._DEFAULTS is not None
    assert all(
        getattr(sop_converter._DEFAULTS, name) is not None
        for name in (
            "agent_definition_factory",
            "skill_factory",
            "frontmatter_parser",
            "tool_authoring",
            "permission_context_factory",
            "agent_loader",
            "clear_sop_caches",
        )
    )


def test_package_and_runtime_exports_are_available():
    import extensions.sop_converter as package
    import extensions.sop_converter.runtime as runtime

    assert package.SdkParser is not None
    assert package.AgentBuilder is runtime.AgentBuilder
    assert package.SkillGrouper is runtime.SkillGrouper


def test_handwritten_macro_template_is_packaged():
    from extensions.sop_converter.runtime.macros.templates import (
        HANDWRITTEN_MACRO_TEMPLATE,
    )

    assert HANDWRITTEN_MACRO_TEMPLATE.is_file()
    content = HANDWRITTEN_MACRO_TEMPLATE.read_text(encoding="utf-8")
    assert "version: 1" in content
    assert "workflow:" in content
