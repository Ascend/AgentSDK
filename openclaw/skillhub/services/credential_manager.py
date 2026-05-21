"""Credential manager implementation."""

import json
from datetime import datetime
from typing import List, Optional

import keyring

from skillhub.config import Settings
from skillhub.interfaces.credential_manager import CredentialManager
from skillhub.models.credential import TokenInfo, TokenValidation


class CredentialManagerImpl(CredentialManager):
    def __init__(self, config: Settings):
        self.config = config
        self.service_name = "skillhub-cli"
        self.metadata_file = config.config_dir / "tokens.json"
        self._metadata: dict[str, dict] = {}
        self._setup_keyring()
        self._load_metadata()

    def _setup_keyring(self):
        import os
        import sys

        if sys.platform == "linux" and not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
            from keyrings.alt.file import PlaintextKeyring

            keyring.set_keyring(PlaintextKeyring())
            return
        try:
            kr = keyring.get_keyring()
            if isinstance(kr, keyring.backends.fail.Keyring):
                raise RuntimeError("no usable keyring backend")
        except Exception:
            from keyrings.alt.file import PlaintextKeyring

            keyring.set_keyring(PlaintextKeyring())

    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def _save_metadata(self):
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    async def store_token(
        self,
        platform: str,
        token: str,
        token_type: str = "pat",
        expires_at: Optional[datetime] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        keyring.set_password(self.service_name, platform, token)

        self._metadata[platform] = {
            "type": token_type,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "scopes": scopes or [],
        }
        self._save_metadata()

    async def get_token(self, platform: str) -> Optional[str]:
        return keyring.get_password(self.service_name, platform)

    async def remove_token(self, platform: str) -> None:
        try:
            keyring.delete_password(self.service_name, platform)
        except Exception:
            import logging

            logging.getLogger(__name__).debug("Failed to delete keyring password for %s", platform)

        if platform in self._metadata:
            del self._metadata[platform]
            self._save_metadata()

    async def list_tokens(self) -> List[TokenInfo]:
        tokens = []
        for platform, metadata in self._metadata.items():
            token = keyring.get_password(self.service_name, platform)
            tokens.append(
                TokenInfo(
                    platform=platform,
                    type=metadata.get("type", "pat"),
                    has_token=token is not None,
                    expires_at=datetime.fromisoformat(metadata["expires_at"]) if metadata.get("expires_at") else None,
                    scopes=metadata.get("scopes", []),
                )
            )
        return tokens

    async def validate_token(self, platform: str, token: str) -> TokenValidation:
        from skillhub.adapters.factory import AdapterFactory
        from skillhub.models.source import SourceType

        try:
            source_type = SourceType(platform)
            adapter = AdapterFactory.create(source_type, token=token)
            rate_limit = await adapter.get_rate_limit()
            await adapter.close()

            return TokenValidation(
                valid=True,
                scopes=[],
                rate_limit=rate_limit,
                message="Token is valid",
            )
        except Exception as e:
            return TokenValidation(
                valid=False,
                scopes=[],
                rate_limit=None,
                message=str(e),
            )

    def is_secure_storage_available(self) -> bool:
        try:
            keyring.get_keyring()
            return True
        except Exception:
            return False
