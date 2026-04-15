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
import sqlite3
from typing import Optional, List, Dict, Any
from agentic_rl.memory.episode.episode_store import Episode, EpisodeStore
from agentic_rl.memory.episode.episode import TerminationReason


class SQLiteEpisodeStore(EpisodeStore):
    """SQLite-backed episode store implementation."""

    def __init__(self, db_path: str):
        """Initialize SQLite episode store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS episodes
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           workflow_id
                           TEXT
                           NOT
                           NULL,
                           task_data
                           TEXT,
                           is_correct
                           BOOLEAN,
                           termination_reason
                           TEXT,
                           trajectories_data
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           workflow_id
                       ) REFERENCES workflows
                       (
                           id
                       )
                           )
                       """)

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS workflows
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       """)

        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_episodes_workflow_id
                           ON episodes(workflow_id)
                       """)

        self.conn.commit()

    def store_episode(self, episode: Episode, workflow_id: str) -> None:
        """Store an episode with associated workflow_id."""
        cursor = self.conn.cursor()

        # Insert workflow if it doesn't exist
        cursor.execute("""
                       INSERT
                       OR IGNORE INTO workflows (id) VALUES (?)
                       """, (workflow_id,))

        # Serialize trajectories to JSON
        trajectories_json = {}
        for name, trajectory in episode.trajectories.items():
            trajectories_json[name] = trajectory.to_dict()

        # Insert episode
        cursor.execute("""
            INSERT OR REPLACE INTO episodes 
            (id, workflow_id, task_data, is_correct, termination_reason, trajectories_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            episode.id,
            workflow_id,
            json.dumps(episode.task) if episode.task else None,
            episode.is_correct,
            episode.termination_reason.value if episode.termination_reason else None,
            json.dumps(trajectories_json)
        ))

        self.conn.commit()

    def get_episodes(self, workflow_id: str, limit: Optional[int] = None) -> List[Episode]:
        """Retrieve episodes for a given workflow_id."""
        cursor = self.conn.cursor()

        query = """
                SELECT id, task_data, is_correct, termination_reason, trajectories_data
                FROM episodes
                WHERE workflow_id = ?
                ORDER BY created_at DESC \
                """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, (workflow_id,))
        rows = cursor.fetchall()

        episodes = []
        for row in rows:
            episode_id, task_data, is_correct, termination_reason, trajectories_data = row

            # Deserialize data
            task = json.loads(task_data) if task_data else None
            trajectories_dict = json.loads(trajectories_data) if trajectories_data else {}

            # Reconstruct episode (note: this is a simplified reconstruction)
            # For full reconstruction, you'd need to properly deserialize trajectories
            episode = Episode(
                id=episode_id,
                task=task,
                is_correct=bool(is_correct),
                trajectories={}  # Simplified - would need proper trajectory reconstruction
            )

            # Set termination reason if available
            if termination_reason:
                episode.termination_reason = TerminationReason(termination_reason)

            episodes.append(episode)

        return episodes

    def get_statistics(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for episodes in the store."""
        cursor = self.conn.cursor()

        if workflow_id:
            cursor.execute("""
                           SELECT COUNT(*)                                        as total_episodes,
                                  SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_episodes
                           FROM episodes
                           WHERE workflow_id = ?
                           """, (workflow_id,))
        else:
            cursor.execute("""
                           SELECT COUNT(*)                                        as total_episodes,
                                  SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_episodes
                           FROM episodes
                           """)

        total, correct = cursor.fetchone()
        accuracy = (correct / total) if total > 0 else 0.0

        return {
            "total_episodes": total,
            "correct_episodes": correct,
            "accuracy": accuracy
        }

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
