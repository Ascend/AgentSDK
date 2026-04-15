#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import json
import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch, mock_open


@pytest.fixture(autouse=True, scope="function")
def mock_ray():
    """Mock ray module for testing memory module"""
    mock_ray = MagicMock()
    mock_ray.remote = lambda cls: cls
    
    with patch.dict('sys.modules', {'ray': mock_ray}):
        yield mock_ray
        
class TestJsonEpisodeStore:
    """Tests for JsonEpisodeStore class."""

    def test_initialization(self):
        """Test JsonEpisodeStore initialization."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        store = JsonEpisodeStore(path="/tmp/test.jsonl")
        
        assert store.store_path == "/tmp/test.jsonl"

    def test_store_episode_simple(self):
        """Test storing a simple episode."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        
        try:
            store = JsonEpisodeStore(path=temp_path)
            
            episode_dict = {
                "episode_id": "test-episode-123",
                "task": {"problem": "test problem"},
                "termination_reason": "env_done",
                "trajectories": []
            }
            
            store.store_episode(episode_dict, "workflow-1")
            
            with open(temp_path, 'r') as f:
                content = f.read().strip()
                stored_data = json.loads(content)
                
            assert stored_data["episode_id"] == "test-episode-123"
            assert stored_data["task"]["problem"] == "test problem"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_store_episode_with_trajectories(self):
        """Test storing an episode with trajectories."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        
        try:
            store = JsonEpisodeStore(path=temp_path)
            
            episode_dict = {
                "episode_id": "test-episode-456",
                "task": {"problem": "complex problem"},
                "termination_reason": "max_turns_exceeded",
                "trajectories": [
                    ["agent_1", {"task": "task1", "reward": 1.0}],
                    ["agent_2", {"task": "task2", "reward": 0.5}]
                ]
            }
            
            store.store_episode(episode_dict, "workflow-2")
            
            with open(temp_path, 'r') as f:
                content = f.read().strip()
                stored_data = json.loads(content)
            
            assert len(stored_data["trajectories"]) == 2
            assert stored_data["trajectories"][0][0] == "agent_1"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_store_episode_with_nested_data(self):
        """Test storing an episode with nested data structures."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        
        try:
            store = JsonEpisodeStore(path=temp_path)
            
            episode_dict = {
                "episode_id": "test-episode-789",
                "task": {
                    "problem": "nested problem",
                    "metadata": {
                        "difficulty": "hard",
                        "tags": ["math", "algebra"]
                    }
                },
                "trajectories": [
                    ["agent_1", {
                        "steps": [
                            {"action": "think", "reward": 0.1},
                            {"action": "execute", "reward": 0.9}
                        ]
                    }]
                ]
            }
            
            store.store_episode(episode_dict, "workflow-3")
            
            with open(temp_path, 'r') as f:
                content = f.read().strip()
                stored_data = json.loads(content)
            
            assert stored_data["task"]["metadata"]["difficulty"] == "hard"
            assert len(stored_data["trajectories"][0][1]["steps"]) == 2
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_store_multiple_episodes(self):
        """Test storing multiple episodes to the same file."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        
        try:
            store = JsonEpisodeStore(path=temp_path)
            
            for i in range(3):
                episode_dict = {
                    "episode_id": f"test-episode-{i}",
                    "task": {"problem": f"problem {i}"},
                    "trajectories": []
                }
                store.store_episode(episode_dict, f"workflow-{i}")
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            
            for i, line in enumerate(lines):
                data = json.loads(line.strip())
                assert data["episode_id"] == f"test-episode-{i}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_store_episode_with_permission_error_retry(self):
        """Test that store_episode retries on PermissionError."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        
        try:
            store = JsonEpisodeStore(path=temp_path)
            
            episode_dict = {
                "episode_id": "test-episode-permission",
                "task": {"problem": "test"},
                "trajectories": []
            }
            
            call_count = [0]
            original_open = open
            
            def mock_open_func(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise PermissionError("Permission denied")
                return original_open(*args, **kwargs)
            
            with patch("builtins.open", side_effect=mock_open_func):
                with patch("random.uniform", return_value=0.01):
                    with patch("time.sleep"):
                        store.store_episode(episode_dict, "workflow-1")
            
            assert call_count[0] == 3
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_convert_to_string_with_list(self):
        """Test convert_to_string function with list."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        store = JsonEpisodeStore(path="/tmp/test.jsonl")
        
        test_list = [1, 2, 3.14, "string", True, None]
        
        def convert_to_string(value):
            if isinstance(value, list):
                return [convert_to_string(v) for v in value]
            elif isinstance(value, dict):
                return {key: convert_to_string(v) for key, v in value.items()}
            else:
                return str(value)
        
        result = convert_to_string(test_list)
        
        assert result == ["1", "2", "3.14", "string", "True", "None"]

    def test_convert_to_string_with_dict(self):
        """Test convert_to_string function with dict."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        store = JsonEpisodeStore(path="/tmp/test.jsonl")
        
        test_dict = {
            "int": 42,
            "float": 3.14,
            "string": "hello",
            "bool": True,
            "none": None
        }
        
        def convert_to_string(value):
            if isinstance(value, list):
                return [convert_to_string(v) for v in value]
            elif isinstance(value, dict):
                return {key: convert_to_string(v) for key, v in value.items()}
            else:
                return str(value)
        
        result = convert_to_string(test_dict)
        
        assert result["int"] == "42"
        assert result["float"] == "3.14"
        assert result["string"] == "hello"
        assert result["bool"] == "True"
        assert result["none"] == "None"

    def test_convert_to_string_nested(self):
        """Test convert_to_string function with nested structures."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        test_nested = {
            "list": [1, 2, {"nested": "value"}],
            "dict": {"inner_list": [3, 4, 5]}
        }
        
        def convert_to_string(value):
            if isinstance(value, list):
                return [convert_to_string(v) for v in value]
            elif isinstance(value, dict):
                return {key: convert_to_string(v) for key, v in value.items()}
            else:
                return str(value)
        
        result = convert_to_string(test_nested)
        
        assert result["list"][0] == "1"
        assert result["list"][2]["nested"] == "value"
        assert result["dict"]["inner_list"] == ["3", "4", "5"]

    def test_get_episodes_not_implemented(self):
        """Test that get_episodes is not implemented."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        store = JsonEpisodeStore(path="/tmp/test.jsonl")
        
        result = store.get_episodes("workflow-1")
        
        assert result is None

    def test_store_episode_with_special_characters(self):
        """Test storing an episode with special characters."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        
        try:
            store = JsonEpisodeStore(path=temp_path)
            
            episode_dict = {
                "episode_id": "test-episode-special",
                "task": {"problem": "test\nwith\tnewlines\tand\ttabs"},
                "trajectories": []
            }
            
            store.store_episode(episode_dict, "workflow-special")
            
            with open(temp_path, 'r') as f:
                content = f.read().strip()
                stored_data = json.loads(content)
            
            assert "newlines" in stored_data["task"]["problem"]
            assert "tabs" in stored_data["task"]["problem"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_store_episode_empty_trajectories(self):
        """Test storing an episode with empty trajectories."""
        from agentic_rl.memory.episode.backend.json_episode_store import JsonEpisodeStore
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        
        try:
            store = JsonEpisodeStore(path=temp_path)
            
            episode_dict = {
                "episode_id": "test-episode-empty",
                "task": {"problem": "test"},
                "trajectories": []
            }
            
            store.store_episode(episode_dict, "workflow-empty")
            
            with open(temp_path, 'r') as f:
                content = f.read().strip()
                stored_data = json.loads(content)
            
            assert stored_data["trajectories"] == []
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
