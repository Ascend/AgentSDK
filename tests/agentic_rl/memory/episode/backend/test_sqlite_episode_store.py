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
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any, Dict

@pytest.fixture(autouse=True, scope="function")
def mock_ray():
    """Mock ray module for testing memory module"""
    mock_ray = MagicMock()
    mock_ray.remote = lambda cls: cls
    
    with patch.dict('sys.modules', {'ray': mock_ray}):
        yield mock_ray

class MockEpisode:
    """Mock Episode class for testing."""
    def __init__(self, episode_id=None, task=None, is_correct=False, trajectories=None):
        self.id = episode_id or "test-id"
        self.task = task
        self.is_correct = is_correct
        self.termination_reason = None
        self.trajectories = trajectories or {}


class MockTerminationReason:
    """Mock TerminationReason enum for testing."""
    MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    ENV_DONE = "env_done"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    TIMEOUT = "timeout"
    
    def __init__(self, value):
        self.value = value


class TestSQLiteEpisodeStore:
    """Tests for SQLiteEpisodeStore class."""

    def test_initialization(self):
        """Test SQLiteEpisodeStore initialization."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            assert store.db_path == temp_db
            store.close()
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_init_db_creates_tables(self):
        """Test that _init_db creates necessary tables."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            
            import sqlite3
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
            assert cursor.fetchone() is not None
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workflows'")
            assert cursor.fetchone() is not None
            
            conn.close()
            store.close()
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_store_episode(self):
        """Test storing an episode."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            
            mock_episode = MockEpisode(
                episode_id="test-episode-123",
                task={"problem": "test problem"},
                is_correct=True
            )
            mock_episode.trajectories = {
                "agent_1": Mock(to_dict=lambda: {"task": "task1", "reward": 1.0})
            }
            
            store.store_episode(mock_episode, "workflow-1")
            
            import sqlite3
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE workflow_id='workflow-1'")
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 1
            store.close()
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_get_episodes_empty_result(self):
        """Test retrieving episodes when no results."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            
            with patch("agentic_rl.memory.episode.backend.sqlite_episode_store.Episode", MockEpisode):
                episodes = store.get_episodes("workflow-nonexistent")
            
            assert len(episodes) == 0
            store.close()
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_get_statistics_zero_episodes(self):
        """Test getting statistics when no episodes."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            stats = store.get_statistics("workflow-empty")
            
            assert stats["total_episodes"] == 0
            assert stats["accuracy"] == 0.0
        finally:
            store.close()
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_close(self):
        """Test closing database connection."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            store.close()
            
            assert store.conn is None
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_store_episode_with_null_task(self):
        """Test storing an episode with null task."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            
            mock_episode = MockEpisode(episode_id="test-episode-null", task=None)
            mock_episode.trajectories = {}
            
            store.store_episode(mock_episode, "workflow-null")
            
            import sqlite3
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE id='test-episode-null'")
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 1
            store.close()
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_store_episode_with_complex_trajectories(self):
        """Test storing an episode with complex trajectories."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            
            mock_trajectory = Mock()
            mock_trajectory.to_dict.return_value = {
                "task": "complex task",
                "steps": [
                    {"action": "step1", "reward": 0.5},
                    {"action": "step2", "reward": 0.5}
                ],
                "reward": 1.0
            }
            
            mock_episode = MockEpisode(episode_id="test-episode-complex")
            mock_episode.trajectories = {
                "agent_1": mock_trajectory,
                "agent_2": mock_trajectory
            }
            
            store.store_episode(mock_episode, "workflow-complex")
            
            import sqlite3
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE id='test-episode-complex'")
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 1
            store.close()
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_multiple_store_operations(self):
        """Test multiple store operations."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            
            for i in range(3):
                mock_episode = MockEpisode(episode_id=f"test-episode-{i}")
                mock_episode.trajectories = {}
                store.store_episode(mock_episode, f"workflow-{i}")
            
            import sqlite3
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodes")
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 3
            store.close()
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)

    def test_get_statistics_with_data(self):
        """Test getting statistics with actual data."""
        from agentic_rl.memory.episode.backend.sqlite_episode_store import SQLiteEpisodeStore
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name
        
        try:
            store = SQLiteEpisodeStore(db_path=temp_db)
            
            for i in range(10):
                mock_episode = MockEpisode(
                    episode_id=f"test-episode-{i}",
                    is_correct=(i < 7)
                )
                mock_episode.trajectories = {}
                store.store_episode(mock_episode, "workflow-1")
            
            stats = store.get_statistics("workflow-1")
            
            assert stats["total_episodes"] == 10
            assert stats["correct_episodes"] == 7
            assert abs(stats["accuracy"] - 0.7) < 0.01
            store.close()
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)
