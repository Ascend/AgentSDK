#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseEnv(ABC):
    @property
    def idx(self) -> Optional[Any]:
        """The index or identifier of the environment, often used within a batch.

        Returns:
            The assigned index or identifier, or None if not set.
        """
        return getattr(self, "_idx", None)

    @idx.setter
    def idx(self, value: Any):
        """Set the environment index or identifier.

        This allows assigning an index or identifier (e.g., its position in a batch)
        to the environment instance after it has been created.

        Example:
            env = MyEnvSubclass()  # Assuming MyEnvSubclass inherits from BaseEnv
            env.idx = 5            # Set the index externally

        Args:
            value: The index or identifier to set for this environment.
        """
        self._idx = value

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        """Standard Gym reset method. Resets the environment to an initial state.

        Returns:
            A tuple typically containing the initial observation and auxiliary info.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Standard Gym step method. Executes one time step within the environment.

        Args:
            action: An action provided by the agent.

        Returns:
            A tuple containing (observation, reward, done, info).
        """
        pass

    def close(self):
        """Standard Gym close method. Performs any necessary cleanup."""
        return

    @staticmethod
    @abstractmethod
    def from_dict(info: dict) -> "BaseEnv":
        """Creates an environment instance from a dictionary.

        This method should be implemented by concrete subclasses to handle
        environment-specific initialization from serialized data.

        Args:
            info: A dictionary containing the necessary information to initialize the environment.

        Returns:
            An instance of the specific BaseEnv subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'from_dict' static method.")

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    @property
    def application_id(self) -> Optional[Any]:
        """The identifier of the trajectory execution instance associated with the env.

        Returns:
            The identifier of the trajectory execution instance associated with the env, or None if not set.
        """
        return getattr(self, "_application_idx", None)

    @application_id.setter
    def application_id(self, value: Any):
        """Set the identifier of the trajectory execution instance associated with the env.

        This allows associating the env with the trajectory execution instance

        Example:
            env = MyEnvSubclass()  # Assuming MyEnvSubclass inherits from BaseEnv
            env.application_id = "some_id"            # Set the index externally

        Args:
            value: The identifier of the trajectory execution instance associated with the env.
        """
        self._application_idx = value