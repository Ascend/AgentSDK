#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""AWS Bedrock authentication matching TypeScript model/bedrock.ts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AwsCredentials:
    """AWS session credentials."""

    access_key_id: str
    secret_access_key: str
    session_token: str = ""
    region: str = "us-east-1"


@dataclass
class AwsAuth:
    """AWS Bedrock authentication handler."""

    region: str = ""

    def load_credentials(self) -> AwsCredentials | None:
        """Load AWS credentials from environment or shared credentials."""
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = os.environ.get("AWS_SESSION_TOKEN", "")
        region = self.region or os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

        if access_key and secret_key:
            return AwsCredentials(
                access_key_id=access_key,
                secret_access_key=secret_key,
                session_token=session_token,
                region=region,
            )

        # Try boto3 if available
        try:
            import boto3

            session = boto3.Session()
            creds = session.get_credentials()
            if creds:
                frozen = creds.get_frozen_credentials()
                return AwsCredentials(
                    access_key_id=frozen.access_key,
                    secret_access_key=frozen.secret_key,
                    session_token=frozen.token or "",
                    region=session.region_name or region,
                )
        except Exception:  # nosec B110
            pass  # Intentional best-effort path; the surrounding fallback remains valid.

        return None

    def is_configured(self) -> bool:
        """Check if AWS credentials are available."""
        return self.load_credentials() is not None

    def get_bedrock_endpoint(self, region: str | None = None) -> str:
        """Get the Bedrock endpoint URL for a region."""
        r = region or self.region or "us-east-1"
        return f"https://bedrock-runtime.{r}.amazonaws.com"
