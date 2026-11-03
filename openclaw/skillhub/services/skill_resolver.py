"""Skill resolver implementation."""

from typing import Any, List, Optional, Tuple

from skillhub.interfaces.skill_resolver import SkillResolver, ValidationResult
from skillhub.models.skill import (
    DependencyConflict,
    DependencyGraph,
    ResolvedSkill,
    SkillManifest,
)
from skillhub.adapters.factory import AdapterFactory
from skillhub.models.source import Source
from skillhub.utils.semver import satisfies_version


class SkillResolverImpl(SkillResolver):
    def __init__(self, sources: List[Source], config=None):
        self.sources = sources
        self.config = config

    async def resolve_version(
        self,
        skill_name: str,
        version_spec: str,
        sources: Optional[List[str]] = None,
    ) -> Optional[ResolvedSkill]:
        source_list = [s for s in self.sources if sources is None or s.id in sources]

        for source in source_list:
            if not source.enabled:
                continue

            token = None
            try:
                if self.config:
                    from skillhub.services.credential_manager import CredentialManagerImpl

                    cred = CredentialManagerImpl(self.config)
                    token = await cred.get_token(source.type.value)
                adapter = AdapterFactory.create(source.type, token=token)

                skill_owner = ""
                skill_repo = skill_name
                if "/" in skill_name:
                    parts = skill_name.split("/", 1)
                    skill_owner = parts[0]
                    skill_repo = parts[1]

                subpath = source.subpath or "skills"
                if subpath:
                    result = await self._try_monorepo_resolution(
                        adapter, source, skill_name, skill_owner, skill_repo, version_spec
                    )
                    if result:
                        return result
                    continue

                result = await self._try_direct_repo_resolution(
                    adapter, source, skill_name, skill_owner, skill_repo, version_spec
                )
                if result:
                    return result
            except Exception:
                continue

        return None

    async def _try_monorepo_resolution(
        self, adapter, source: Source, skill_name: str, skill_owner: str, skill_repo: str, version_spec: str
    ) -> Optional[ResolvedSkill]:
        subpath = source.subpath or "skills"
        url_parts = source.url.rstrip("/").split("/")
        source_owner = url_parts[-2] if len(url_parts) >= 2 else ""
        source_repo = url_parts[-1]

        if not source_repo:
            return None

        skill_path = f"{subpath}/{skill_owner}/{skill_repo}" if skill_owner else f"{subpath}/{skill_repo}"
        try:
            manifest_content = await adapter.get_file_content(source_owner, source_repo, f"{skill_path}/SKILL.md")
            manifest = self._parse_manifest(manifest_content)
            try:
                source_repo_data = await adapter.get_repository(source_owner, source_repo)
                default_branch = source_repo_data.default_branch
            except Exception:
                default_branch = "main"

            if version_spec == "latest":
                version = manifest.version or "latest"
                ref = default_branch
            else:
                ref, version, manifest = await self._resolve_version_ref(
                    adapter, source_owner, source_repo, skill_path, version_spec, manifest, default_branch
                )
                if ref is None:
                    await adapter.close()
                    return None

            return ResolvedSkill(
                name=skill_name,
                version=version,
                repository=source.url,
                ref=ref,
                manifest=manifest,
                source={"id": source.id, "type": source.type.value},
                download_url=None,
                subpath=skill_path,
            )
        except Exception:
            await adapter.close()
            return None

    async def _resolve_version_ref(
        self,
        adapter,
        source_owner: str,
        source_repo: str,
        skill_path: str,
        version_spec: str,
        manifest: SkillManifest,
        default_branch: str,
    ) -> Tuple[Optional[str], str, SkillManifest]:
        """Resolve a git ref matching the requested version spec.

        Returns (ref, version, manifest). ref is None when the requested
        version cannot be satisfied by any tag or the default branch.
        """
        # Match tags across all pages so repositories with more tags than the
        # default page size still resolve older versions. Stops early as soon
        # as a matching tag is found.
        per_page = 100
        page = 1
        while page <= 50:
            try:
                batch = await adapter.list_tags(source_owner, source_repo, per_page=per_page, page=page)
            except Exception:
                break
            if not batch:
                break

            for tag in batch:
                # Allow both "1.0.0" and "v1.0.0" style tags.
                candidate = tag.name[1:] if tag.name.startswith("v") and tag.name[1:2].isdigit() else tag.name
                try:
                    if not satisfies_version(candidate, version_spec):
                        continue
                except ValueError:
                    continue

                try:
                    tag_manifest = self._parse_manifest(
                        await adapter.get_file_content(
                            source_owner, source_repo, f"{skill_path}/SKILL.md", ref=tag.name
                        )
                    )
                except Exception:
                    continue

                return tag.name, (tag_manifest.version or candidate), tag_manifest

            if len(batch) < per_page:
                break
            page += 1

        # No matching tag: only fall back to the default branch when its
        # manifest version satisfies the requested spec, otherwise fail
        # explicitly instead of registering the default-branch content under
        # the requested version.
        if manifest.version:
            try:
                if satisfies_version(manifest.version, version_spec):
                    return default_branch, manifest.version, manifest
            except ValueError:
                pass

        return None, version_spec, manifest

    async def _try_direct_repo_resolution(
        self, adapter, source: Source, skill_name: str, skill_owner: str, skill_repo: str, version_spec: str
    ) -> Optional[ResolvedSkill]:
        repo = await adapter.get_repository(skill_owner, skill_repo)
        manifest = await self.fetch_manifest(repo.url, repo.default_branch, source.type.value)

        if version_spec == "latest":
            version = manifest.version or "latest"
        else:
            tags = await adapter.list_tags(skill_owner, skill_repo)
            version = None
            for tag in tags:
                if satisfies_version(tag.name, version_spec):
                    version = tag.name
                    break

        if version:
            return ResolvedSkill(
                name=skill_name,
                version=version,
                repository=repo.url,
                ref=version,
                manifest=manifest,
                source={"id": source.id, "type": source.type.value},
                download_url=f"{repo.url}/archive/{version}.zip",
                subpath=source.subpath,
            )

        await adapter.close()
        return None

    async def list_available_versions(
        self,
        skill_name: str,
        source: Optional[str] = None,
    ) -> List[str]:
        source_list = [s for s in self.sources if source is None or s.id == source]

        for s in source_list:
            if not s.enabled:
                continue

            token = None
            try:
                if self.config:
                    from skillhub.services.credential_manager import CredentialManagerImpl

                    cred = CredentialManagerImpl(self.config)
                    token = await cred.get_token(s.type.value)
                adapter = AdapterFactory.create(s.type, token=token)
                tags = await adapter.list_tags("", skill_name)
                await adapter.close()
                return [tag.name for tag in tags]
            except Exception:
                continue

        return []

    async def resolve_dependencies(
        self,
        skill: ResolvedSkill,
        include_optional: bool = False,
    ) -> DependencyGraph:
        dependencies = {}
        conflicts = []
        order = [skill.name]

        if skill.manifest.dependencies:
            for dep_name, dep_spec in skill.manifest.dependencies.items():
                dep_skill = await self.resolve_version(dep_name, dep_spec)
                if dep_skill:
                    dependencies[dep_name] = dep_skill
                    order.append(dep_name)
                else:
                    conflicts.append(
                        DependencyConflict(
                            skill=dep_name,
                            required_by=[skill.name],
                            versions=[dep_spec],
                        )
                    )

        return DependencyGraph(
            root=skill,
            dependencies=dependencies,
            conflicts=conflicts,
            order=order,
        )

    async def fetch_manifest(
        self,
        repository: str,
        ref: str,
        platform: str,
    ) -> SkillManifest:
        from skillhub.models.source import SourceType
        from urllib.parse import urlparse

        # Parse owner and repo from repository URL (e.g., https://github.com/owner/repo)
        parsed_url = urlparse(repository)
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError(f"Invalid repository URL: {repository}")

        owner, repo_name = path_parts[-2], path_parts[-1]

        adapter = AdapterFactory.create(SourceType(platform), f"{parsed_url.scheme}://{parsed_url.netloc}")
        try:
            content = await adapter.get_file_content(owner, repo_name, "SKILL.md", ref)
            return self._parse_manifest(content)
        finally:
            await adapter.close()

    def validate_manifest(self, manifest: Any) -> ValidationResult:
        errors = []

        if not hasattr(manifest, "name") or not manifest.name:
            errors.append("Missing required field: name")

        if not hasattr(manifest, "description") or not manifest.description:
            errors.append("Missing required field: description")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _parse_manifest(self, content: str) -> SkillManifest:
        import yaml

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                return SkillManifest(**frontmatter)

        return SkillManifest(name="unknown", description="No description")
