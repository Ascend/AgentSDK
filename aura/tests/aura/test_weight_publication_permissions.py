"""Exercise the real publication method without importing Ray or NPU packages."""
import ast
import os
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "aura/controllers/rollout_controller/rollout_weight_manager.py"


def publication_method():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "_move_weights"]
    assert len(methods) == 1
    namespace = dict(os=os, shutil=shutil, traceback=traceback)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_move_weights"]


@pytest.mark.parametrize("case", ["success", "chmod_failure", "empty"])
def test_publication_permissions(tmp_path, monkeypatch, case):
    src = tmp_path / "source"
    src.mkdir()
    manager = SimpleNamespace(inference_save_path=str(tmp_path / "published"),
                              weights_version=0, logger=Mock())
    if case != "empty":
        weight = src / "model.safetensors"
        weight.write_bytes(b"test payload")
        weight.chmod(0o600)
    real_chmod = os.chmod
    calls = []

    def chmod(path, mode):
        assert manager.weights_version == 0
        calls.append(mode)
        if case == "chmod_failure":
            raise PermissionError("chmod denied")
        return real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", chmod)
    if case == "success":
        publication_method()(manager, str(src), 1)
        assert calls == [0o644]
        assert manager.weights_version == 1
        assert (tmp_path / "published/weights_1/model.safetensors").read_bytes() == b"test payload"
    else:
        with pytest.raises(PermissionError if case == "chmod_failure" else RuntimeError):
            publication_method()(manager, str(src), 1)
        assert manager.weights_version == 0
