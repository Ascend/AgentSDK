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
import pytest
from unittest.mock import Mock, MagicMock, patch
from abc import ABC, abstractmethod

@pytest.fixture(autouse=True, scope="function")
def mock_ray():
    """Mock ray module for testing memory module"""
    mock_ray = MagicMock()
    mock_ray.remote = lambda cls: cls
    
    with patch.dict('sys.modules', {'ray': mock_ray}):
        yield mock_ray

class TestEpisodeStoreInterface:
    """Tests for EpisodeStore abstract interface."""

    def test_episode_store_is_abstract(self):
        """Test that EpisodeStore is an abstract class."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        assert ABC in EpisodeStore.__bases__
        
        with pytest.raises(TypeError):
            EpisodeStore()

    def test_episode_store_has_store_episode_method(self):
        """Test that EpisodeStore has store_episode abstract method."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        assert hasattr(EpisodeStore, 'store_episode')
        assert getattr(EpisodeStore.store_episode, '__isabstractmethod__', False)

    def test_episode_store_has_get_episodes_method(self):
        """Test that EpisodeStore has get_episodes abstract method."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        assert hasattr(EpisodeStore, 'get_episodes')
        assert getattr(EpisodeStore.get_episodes, '__isabstractmethod__', False)

    def test_episode_store_store_episode_signature(self):
        """Test store_episode method signature."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        import inspect
        
        sig = inspect.signature(EpisodeStore.store_episode)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'episode' in params
        assert 'workflow_id' in params

    def test_episode_store_get_episodes_signature(self):
        """Test get_episodes method signature."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        import inspect
        
        sig = inspect.signature(EpisodeStore.get_episodes)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'workflow_id' in params
        assert 'limit' in params


class TestConcreteEpisodeStore:
    """Tests for concrete implementation of EpisodeStore."""

    def test_concrete_implementation_must_implement_store_episode(self):
        """Test that concrete class must implement store_episode."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        class IncompleteStore(EpisodeStore):
            def get_episodes(self, workflow_id, limit=None):
                return []
        
        with pytest.raises(TypeError):
            IncompleteStore()

    def test_concrete_implementation_must_implement_get_episodes(self):
        """Test that concrete class must implement get_episodes."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        class IncompleteStore(EpisodeStore):
            def store_episode(self, episode, workflow_id):
                pass
        
        with pytest.raises(TypeError):
            IncompleteStore()

    def test_complete_concrete_implementation(self):
        """Test that complete implementation can be instantiated."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        class CompleteStore(EpisodeStore):
            def store_episode(self, episode, workflow_id):
                pass
            
            def get_episodes(self, workflow_id, limit=None):
                return []
        
        store = CompleteStore()
        assert isinstance(store, EpisodeStore)

    def test_concrete_store_store_episode(self):
        """Test store_episode in concrete implementation."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        class TestStore(EpisodeStore):
            def __init__(self):
                self.episodes = {}
            
            def store_episode(self, episode, workflow_id):
                if workflow_id not in self.episodes:
                    self.episodes[workflow_id] = []
                self.episodes[workflow_id].append(episode)
            
            def get_episodes(self, workflow_id, limit=None):
                episodes = self.episodes.get(workflow_id, [])
                return episodes[:limit] if limit else episodes
        
        store = TestStore()
        mock_episode = Mock()
        mock_episode.episode_id = "test-episode-123"
        
        store.store_episode(mock_episode, "workflow-1")
        
        assert "workflow-1" in store.episodes
        assert len(store.episodes["workflow-1"]) == 1
        assert store.episodes["workflow-1"][0] == mock_episode

    def test_concrete_store_get_episodes(self):
        """Test get_episodes in concrete implementation."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        class TestStore(EpisodeStore):
            def __init__(self):
                self.episodes = {}
            
            def store_episode(self, episode, workflow_id):
                if workflow_id not in self.episodes:
                    self.episodes[workflow_id] = []
                self.episodes[workflow_id].append(episode)
            
            def get_episodes(self, workflow_id, limit=None):
                episodes = self.episodes.get(workflow_id, [])
                return episodes[:limit] if limit else episodes
        
        store = TestStore()
        mock_episode1 = Mock()
        mock_episode2 = Mock()
        mock_episode3 = Mock()
        
        store.store_episode(mock_episode1, "workflow-1")
        store.store_episode(mock_episode2, "workflow-1")
        store.store_episode(mock_episode3, "workflow-2")
        
        episodes = store.get_episodes("workflow-1")
        assert len(episodes) == 2
        
        episodes_limited = store.get_episodes("workflow-1", limit=1)
        assert len(episodes_limited) == 1
        
        episodes_empty = store.get_episodes("workflow-3")
        assert len(episodes_empty) == 0

    def test_concrete_store_with_multiple_workflows(self):
        """Test store with multiple workflows."""
        from agentic_rl.memory.episode.episode_store import EpisodeStore
        
        class TestStore(EpisodeStore):
            def __init__(self):
                self.episodes = {}
            
            def store_episode(self, episode, workflow_id):
                if workflow_id not in self.episodes:
                    self.episodes[workflow_id] = []
                self.episodes[workflow_id].append(episode)
            
            def get_episodes(self, workflow_id, limit=None):
                episodes = self.episodes.get(workflow_id, [])
                return episodes[:limit] if limit else episodes
        
        store = TestStore()
        
        for i in range(3):
            episode = Mock()
            episode.episode_id = f"episode-w1-{i}"
            store.store_episode(episode, "workflow-1")
        
        for i in range(2):
            episode = Mock()
            episode.episode_id = f"episode-w2-{i}"
            store.store_episode(episode, "workflow-2")
        
        assert len(store.get_episodes("workflow-1")) == 3
        assert len(store.get_episodes("workflow-2")) == 2
        assert len(store.get_episodes("workflow-3")) == 0
