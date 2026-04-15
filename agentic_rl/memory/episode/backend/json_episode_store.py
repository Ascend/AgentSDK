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

import json
import logging
import random
import time
from typing import Optional

from agentic_rl.memory.episode.episode import Episode
from agentic_rl.memory.episode.episode_store import EpisodeStore

logger = logging.getLogger(__name__)


class JsonEpisodeStore(EpisodeStore):
    """Json-backed episode store implementation."""

    def __init__(self, path: str):
        self.store_path = path

    def store_episode(self, episode_dict: dict, workflow_id: str) -> None:

        def convert_to_string(value):
            if isinstance(value, list):
                return [convert_to_string(v) for v in value]
            elif isinstance(value, dict):
                return {key: convert_to_string(v) for key, v in value.items()}
            else:
                return str(value)

        # Concurrent write lock
        while True:
            try:
                with open(self.store_path, 'a') as f:
                    episode_dict_str = convert_to_string(episode_dict)
                    episode_dict_str = json.dumps(episode_dict_str, ensure_ascii=False)
                    f.write(episode_dict_str + '\n')
                logger.info(f'write_file {self.store_path} done')
                break
            except PermissionError:
                wait_time = random.uniform(1, 10)
                time.sleep(wait_time)

    def get_episodes(self, workflow_id: str, limit: Optional[int] = None) -> list[Episode]:
        pass
