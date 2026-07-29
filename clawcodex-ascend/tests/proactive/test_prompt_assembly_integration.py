from __future__ import annotations

from clawcodex_ext.context_system.prompt_assembly import (
    build_full_system_prompt,
    build_full_system_prompt_blocks,
)
from clawcodex_ext.services.proactive import reset_default_controller_for_tests


def test_prompt_assembly_omits_proactive_section_when_inactive() -> None:
    reset_default_controller_for_tests()

    prompt = build_full_system_prompt(cwd=".")

    assert "<proactive-mode" not in prompt


def test_prompt_assembly_injects_proactive_section_when_active() -> None:
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test", focus="minimal")

    prompt = build_full_system_prompt(cwd=".")
    blocks = build_full_system_prompt_blocks(cwd=".")
    block_text = "\n".join(str(block.get("text", "")) for block in blocks)

    assert '<proactive-mode phase="active" focus="minimal">' in prompt
    assert '<proactive-mode phase="active" focus="minimal">' in block_text
