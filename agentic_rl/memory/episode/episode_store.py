#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Optional

from agentic_rl.memory.episode.episode import Episode


class EpisodeStore(ABC):
    """Abstract interface for storing episodes."""

    @abstractmethod
    def store_episode(self, episode: Episode, workflow_id: str) -> None:
        """Store an episode with associated workflow_id."""
        pass

    @abstractmethod
    def get_episodes(self, workflow_id: str, limit: Optional[int] = None):
        """Retrieve episodes for a given workflow_id."""
        pass
