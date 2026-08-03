# pylint: disable=E0611
from clawcodex_ext.agent.tool_authoring.call_handlers.bash import parse_sop_wrapper_stdout


def test_parse_sop_wrapper_stdout_last_json_line():
    raw = "2026-06-23 | common | INFO | Registered connector pool type: default\n\"/root/.openjiuwen\"\n"
    assert parse_sop_wrapper_stdout(raw) == "/root/.openjiuwen"


def test_parse_sop_wrapper_stdout_object():
    raw = 'log line\n{"ok": true}\n'
    assert parse_sop_wrapper_stdout(raw) == {"ok": True}
