"""Source manager implementation."""

import json
from typing import Any, Dict, List, Optional

from skillhub.config import Settings
from skillhub.interfaces.source_manager import SourceManager, SourceTestResult
from skillhub.models.skill import DiscoveredSkill
from skillhub.models.source import Source
from skillhub.adapters.factory import AdapterFactory
from skillhub.services.credential_manager import CredentialManagerImpl


class SourceManagerImpl(SourceManager):
    def __init__(self, config: Settings):
        self.config = config
        self.sources_file = config.config_dir / "sources.json"
        self._sources: Dict[str, Source] = {}
        self._credential_manager = CredentialManagerImpl(config)
        self._discover_cache: Dict[str, List[DiscoveredSkill]] = {}
        self._load_sources()

    def _load_sources(self):
        if self.sources_file.exists():
            with open(self.sources_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for source_data in data:
                    source = Source(**source_data)
                    self._sources[source.id] = source

    async def _create_adapter(self, source: Source):
        """Create adapter with token from credential store."""
        token = await self._credential_manager.get_token(source.type.value)
        return AdapterFactory.create(source.type, token=token)

    def _save_sources(self):
        with open(self.sources_file, "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in self._sources.values()], f, indent=2, default=str)

    async def add_source(self, source: Source) -> Source:
        self._sources[source.id] = source
        self._save_sources()
        return source

    async def remove_source(self, source_id: str) -> None:
        if source_id in self._sources:
            del self._sources[source_id]
            self._save_sources()

    async def update_source(self, source_id: str, updates: Dict[str, Any]) -> Source:
        if source_id not in self._sources:
            raise ValueError(f"Source not found: {source_id}")

        source = self._sources[source_id]
        for key, value in updates.items():
            if hasattr(source, key):
                setattr(source, key, value)

        self._save_sources()
        return source

    async def get_source(self, source_id: str) -> Optional[Source]:
        return self._sources.get(source_id)

    async def list_sources(self) -> List[Source]:
        return list(self._sources.values())

    async def enable_source(self, source_id: str) -> None:
        if source_id in self._sources:
            self._sources[source_id].enabled = True
            self._save_sources()

    async def disable_source(self, source_id: str) -> None:
        if source_id in self._sources:
            self._sources[source_id].enabled = False
            self._save_sources()

    async def test_source(self, source_id: str) -> SourceTestResult:
        source = self._sources.get(source_id)
        if not source:
            return SourceTestResult(success=False, message=f"Source not found: {source_id}")

        try:
            adapter = await self._create_adapter(source)
            # Try a real API call to verify connectivity + auth
            owner, repo = self._extract_owner_repo(source.url)
            if owner and repo:
                await adapter.get_repository(owner, repo)
            else:
                await adapter.get_rate_limit()
            rate_limit = await adapter.get_rate_limit()
            await adapter.close()
            return SourceTestResult(
                success=True,
                message="Source is accessible",
                rate_limit=rate_limit,
            )
        except Exception as e:
            return SourceTestResult(success=False, message=str(e))

    @staticmethod
    def _extract_owner_repo(url: str):
        """Extract owner and repo from a URL like https://gitcode.com/Ascend/agent-skills."""
        parts = url.rstrip('/').split('/')
        if len(parts) >= 2:
            repo = parts[-1]
            owner = parts[-2]
            if owner and repo and not owner.isdigit():
                return owner, repo
        return "", ""

    async def discover_from_source(self, source_id: str, force_refresh: bool = False) -> List[DiscoveredSkill]:
        if not force_refresh and source_id in self._discover_cache:
            return self._discover_cache[source_id]

        source = self._sources.get(source_id)
        if not source:
            return []

        owner, repo = self._extract_owner_repo(source.url)
        adapter = await self._create_adapter(source)
        try:
            if owner and repo:
                skills = await self._discover_from_single_repo(adapter, source, owner, repo)
            else:
                skills = await self._discover_from_org_repos(adapter, source, owner or "")
            self._discover_cache[source_id] = skills
            return skills
        finally:
            await adapter.close()

    async def _discover_from_single_repo(self, adapter, source: Source, owner: str, repo: str) -> List[DiscoveredSkill]:
        skills = []
        subpath = source.subpath or "skills"
        contents = await adapter.get_contents(owner, repo, subpath)
        if not isinstance(contents, list):
            return skills
        for item in contents:
            if item.type != "dir":
                continue
            try:
                manifest = await adapter.get_file_content(owner, repo, f"{subpath}/{item.name}/SKILL.md")
                front = self._parse_frontmatter(manifest)
                skill = DiscoveredSkill(
                    name=front.get("name", item.name),
                    version="latest",
                    description=front.get("description", ""),
                    author=owner,
                    tags=front.get("keywords", []),
                    source={"id": source.id, "name": source.name, "type": source.type.value},
                    repository={"owner": owner, "name": repo, "url": source.url},
                    manifest_url=f"{source.url}/blob/master/{subpath}/{item.name}/SKILL.md",
                    available_versions=["latest"],
                )
                skills.append(skill)
            except Exception:
                continue
        return skills

    async def _discover_from_org_repos(self, adapter, source: Source, owner: str) -> List[DiscoveredSkill]:
        skills = []
        repos = await adapter.list_repositories(owner)
        for r in repos:
            try:
                await adapter.get_file_content(owner, r.name, "SKILL.md")
                skill = DiscoveredSkill(
                    name=r.name,
                    version="latest",
                    description=r.description,
                    author=r.owner.get("login", ""),
                    tags=r.topics,
                    source={"id": source.id, "name": source.name, "type": source.type.value},
                    repository={"owner": r.owner.get("login", ""), "name": r.name, "url": r.url},
                    manifest_url=f"{r.url}/SKILL.md",
                    available_versions=["latest"],
                )
                skills.append(skill)
            except Exception:
                continue
        return skills

    @staticmethod
    def _parse_frontmatter(content: str) -> dict:
        """Parse YAML frontmatter from SKILL.md content."""
        if not content.startswith("---"):
            return {}
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}
        try:
            import yaml

            return yaml.safe_load(parts[1]) or {}
        except Exception:
            return {}

    async def search_across_sources(
        self, query: str, sources: Optional[List[str]] = None
    ) -> Dict[str, List[DiscoveredSkill]]:
        results = {}
        source_ids = sources or list(self._sources.keys())

        for source_id in source_ids:
            source = self._sources.get(source_id)
            if not source or not source.enabled:
                continue

            try:
                # For monorepo sources, discover locally and filter by query
                all_skills = await self.discover_from_source(source_id)
                query_lower = query.lower()
                matched = []
                for s in all_skills:
                    if (
                        query_lower in s.name.lower()
                        or query_lower in (s.description or "").lower()
                        or any(query_lower in t.lower() for t in s.tags)
                    ):
                        matched.append(s)
                results[source_id] = matched
            except Exception:
                continue

        return results
