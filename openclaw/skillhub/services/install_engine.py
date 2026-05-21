"""Install engine implementation."""

import json
import os
import shutil
from datetime import datetime
from typing import List, Optional

from skillhub.config import Settings
from skillhub.interfaces.install_engine import (
    InstallEngine,
    InstallOptions,
    VerificationIssue,
    VerificationResult,
)
from skillhub.models.skill import InstallResult, InstalledSkill, ResolvedSkill
from skillhub.utils.archive import extract_archive
from skillhub.utils.checksum import compute_checksum


class InstallEngineImpl(InstallEngine):
    def __init__(self, config: Settings):
        self.config = config
        self.installed_file = config.data_dir / "installed.json"
        self._installed: dict[str, InstalledSkill] = {}
        self._load_installed()

    def _load_installed(self):
        if self.installed_file.exists():
            with open(self.installed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for skill_data in data:
                    skill = InstalledSkill(**skill_data)
                    self._installed[skill.name] = skill

    def _save_installed(self):
        with open(self.installed_file, "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in self._installed.values()], f, indent=2, default=str)

    async def install(
        self,
        skill: ResolvedSkill,
        options: Optional[InstallOptions] = None,
    ) -> InstallResult:
        opts = options or InstallOptions()
        start_time = datetime.now()

        if skill.name in self._installed and not opts.force:
            return InstallResult(
                success=False,
                skill=self._installed[skill.name],
                installed_dependencies=[],
                warnings=[],
                errors=[f"Skill {skill.name} is already installed. Use --force to reinstall."],
                duration=0.0,
            )

        install_path = opts.target_path or str(self.config.skills_dir / skill.name)

        try:
            os.makedirs(install_path, exist_ok=True)

            # For monorepo sources (has subpath but no download_url), use Contents API
            if skill.subpath and not skill.download_url:
                await self._install_from_contents_api(skill, install_path)
            else:
                archive_path = os.path.join(install_path, "skill.zip")
                from skillhub.utils.http import HttpClient

                client = HttpClient(skill.repository)
                await client.download(skill.download_url, archive_path)  # type: ignore - download_url is always set when install is called
                await client.close()

                # Compute correct archive strip prefix from repository URL
                # GitHub archives extract to {repo_name}-{ref}/ regardless of skill name
                from urllib.parse import urlparse

                parsed = urlparse(skill.repository)
                repo_name = parsed.path.strip('/').split('/')[-1]
                strip_prefix = f"{repo_name}-{skill.ref}/"
                extract_archive(archive_path, install_path, strip_prefix=strip_prefix)

                if skill.subpath:
                    self._filter_subpath(install_path, skill.subpath)

                os.remove(archive_path)

            checksum = compute_checksum(install_path)

            installed_skill = InstalledSkill(
                name=skill.name,
                version=skill.version,
                source_id=skill.source.get("id", ""),
                source_type=skill.source.get("type", ""),
                repository=skill.repository,
                ref=skill.ref,
                install_path=install_path,
                checksum=checksum,
            )

            self._installed[skill.name] = installed_skill
            self._save_installed()

            duration = (datetime.now() - start_time).total_seconds()

            return InstallResult(
                success=True,
                skill=installed_skill,
                installed_dependencies=[],
                warnings=[],
                errors=[],
                duration=duration,
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return InstallResult(
                success=False,
                skill=InstalledSkill(
                    name=skill.name,
                    version=skill.version,
                    source_id="",
                    source_type="",
                    repository=skill.repository,
                    ref=skill.ref,
                    install_path=install_path,
                    checksum="",
                ),
                installed_dependencies=[],
                warnings=[],
                errors=[str(e)],
                duration=duration,
            )

    def _filter_subpath(self, install_path: str, subpath: str) -> None:
        target_dir = os.path.join(install_path, subpath)
        if not os.path.exists(target_dir):
            raise ValueError(f"Subpath '{subpath}' not found in archive")

        temp_dir = install_path + "_temp"
        os.rename(install_path, temp_dir)

        os.makedirs(install_path)
        self._copy_recursive(target_dir, install_path)
        shutil.rmtree(temp_dir)

    def _copy_recursive(self, src: str, dst: str) -> None:
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)
            if os.path.isdir(src_item):
                os.makedirs(dst_item, exist_ok=True)
                self._copy_recursive(src_item, dst_item)
            else:
                shutil.copy2(src_item, dst_item)

    async def _install_from_contents_api(self, skill: ResolvedSkill, install_path: str) -> None:
        """Install skill files via platform adapter's Contents API."""
        from skillhub.adapters.factory import AdapterFactory
        from skillhub.models.source import SourceType
        from skillhub.services.credential_manager import CredentialManagerImpl

        source_type = SourceType(skill.source.get("type", "github"))
        cred = CredentialManagerImpl(self.config)
        token = await cred.get_token(source_type.value)
        adapter = AdapterFactory.create(source_type, token=token)

        # Extract owner/repo from repository URL
        parts = skill.repository.rstrip('/').split('/')
        owner, repo = parts[-2], parts[-1]

        async def download_directory(path: str, dest: str) -> None:
            ref = skill.ref or None
            contents = await adapter.get_contents(owner, repo, path, ref=ref)
            if not isinstance(contents, list):
                contents = [contents]
            for item in contents:
                item_dest = os.path.join(dest, item.name)
                if item.type == 'file':
                    content = await adapter.get_file_content(owner, repo, item.path, ref=ref)
                    os.makedirs(os.path.dirname(item_dest), exist_ok=True)
                    with open(item_dest, 'w', encoding='utf-8') as f:
                        f.write(content)
                elif item.type == 'dir':
                    os.makedirs(item_dest, exist_ok=True)
                    await download_directory(item.path, item_dest)

        try:
            if skill.subpath:
                await download_directory(skill.subpath, install_path)
        finally:
            await adapter.close()

    async def install_from_path(
        self,
        path: str,
        options: Optional[InstallOptions] = None,
    ) -> InstallResult:
        opts = options or InstallOptions()
        start_time = datetime.now()

        skill_md = os.path.join(path, "SKILL.md")
        if not os.path.exists(skill_md):
            return InstallResult(
                success=False,
                skill=InstalledSkill(
                    name="unknown",
                    version="unknown",
                    source_id="",
                    source_type="",
                    repository="",
                    ref="",
                    install_path=path,
                    checksum="",
                ),
                installed_dependencies=[],
                warnings=[],
                errors=["No SKILL.md found in path"],
                duration=0.0,
            )

        with open(skill_md, "r", encoding="utf-8") as f:
            content = f.read()

        import yaml

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                name = frontmatter.get("name", os.path.basename(path))
                version = frontmatter.get("version", "latest")
            else:
                name = os.path.basename(path)
                version = "latest"
        else:
            name = os.path.basename(path)
            version = "latest"

        install_path = opts.target_path or str(self.config.skills_dir / name)

        try:
            if os.path.exists(install_path) and opts.force:
                shutil.rmtree(install_path)

            shutil.copytree(path, install_path)

            checksum = compute_checksum(install_path)

            installed_skill = InstalledSkill(
                name=name,
                version=version,
                source_id="local",
                source_type="local",
                repository="",
                ref="",
                install_path=install_path,
                checksum=checksum,
            )

            self._installed[name] = installed_skill
            self._save_installed()

            duration = (datetime.now() - start_time).total_seconds()

            return InstallResult(
                success=True,
                skill=installed_skill,
                installed_dependencies=[],
                warnings=[],
                errors=[],
                duration=duration,
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return InstallResult(
                success=False,
                skill=InstalledSkill(
                    name=name,
                    version=version,
                    source_id="",
                    source_type="",
                    repository="",
                    ref="",
                    install_path=install_path,
                    checksum="",
                ),
                installed_dependencies=[],
                warnings=[],
                errors=[str(e)],
                duration=duration,
            )

    async def uninstall(self, skill_name: str) -> None:
        if skill_name in self._installed:
            skill = self._installed[skill_name]
            if os.path.exists(skill.install_path):
                shutil.rmtree(skill.install_path)
            del self._installed[skill_name]
            self._save_installed()

    async def upgrade(
        self,
        skill_name: str,
        version: Optional[str] = None,
    ) -> InstallResult:
        if skill_name not in self._installed:
            return InstallResult(
                success=False,
                skill=InstalledSkill(
                    name=skill_name,
                    version="",
                    source_id="",
                    source_type="",
                    repository="",
                    ref="",
                    install_path="",
                    checksum="",
                ),
                installed_dependencies=[],
                warnings=[],
                errors=[f"Skill {skill_name} is not installed"],
                duration=0.0,
            )

        current = self._installed[skill_name]
        from skillhub.services.skill_resolver import SkillResolverImpl
        from skillhub.services.source_manager import SourceManagerImpl

        source_manager = SourceManagerImpl(self.config)
        sources = await source_manager.list_sources()
        resolver = SkillResolverImpl(sources, config=self.config)

        target_version = version or "latest"
        resolved = await resolver.resolve_version(skill_name, target_version)

        if not resolved:
            return InstallResult(
                success=False,
                skill=current,
                installed_dependencies=[],
                warnings=[],
                errors=[f"Could not resolve {skill_name}@{target_version}"],
                duration=0.0,
            )

        await self.uninstall(skill_name)
        return await self.install(resolved, InstallOptions(force=True))

    async def list_installed(self) -> List[InstalledSkill]:
        return list(self._installed.values())

    async def get_installed(self, skill_name: str) -> Optional[InstalledSkill]:
        return self._installed.get(skill_name)

    def is_installed(self, skill_name: str, version: Optional[str] = None) -> bool:
        if skill_name not in self._installed:
            return False
        if version:
            return self._installed[skill_name].version == version
        return True

    async def verify(self, skill_name: str) -> VerificationResult:
        if skill_name not in self._installed:
            return VerificationResult(
                skill=skill_name,
                valid=False,
                issues=[VerificationIssue(type="missing", path="", message="Skill not installed")],
                checksum={"expected": "", "actual": "", "match": False},
            )

        skill = self._installed[skill_name]
        issues = []

        if not os.path.exists(skill.install_path):
            issues.append(
                VerificationIssue(
                    type="missing",
                    path=skill.install_path,
                    message="Installation directory not found",
                )
            )
        else:
            actual_checksum = compute_checksum(skill.install_path)
            if actual_checksum != skill.checksum:
                issues.append(
                    VerificationIssue(
                        type="corrupt",
                        path=skill.install_path,
                        message="Checksum mismatch",
                    )
                )

        return VerificationResult(
            skill=skill_name,
            valid=len(issues) == 0,
            issues=issues,
            checksum={
                "expected": skill.checksum,
                "actual": compute_checksum(skill.install_path) if os.path.exists(skill.install_path) else "",
                "match": len(issues) == 0,
            },
        )

    async def repair(self, skill_name: str) -> InstallResult:
        if skill_name not in self._installed:
            return InstallResult(
                success=False,
                skill=InstalledSkill(
                    name=skill_name,
                    version="",
                    source_id="",
                    source_type="",
                    repository="",
                    ref="",
                    install_path="",
                    checksum="",
                ),
                installed_dependencies=[],
                warnings=[],
                errors=[f"Skill {skill_name} is not installed"],
                duration=0.0,
            )

        current = self._installed[skill_name]
        from skillhub.services.skill_resolver import SkillResolverImpl
        from skillhub.services.source_manager import SourceManagerImpl

        source_manager = SourceManagerImpl(self.config)
        sources = await source_manager.list_sources()
        resolver = SkillResolverImpl(sources, config=self.config)

        resolved = await resolver.resolve_version(skill_name, current.version)

        if not resolved:
            return InstallResult(
                success=False,
                skill=current,
                installed_dependencies=[],
                warnings=[],
                errors=[f"Could not resolve {skill_name}@{current.version}"],
                duration=0.0,
            )

        await self.uninstall(skill_name)
        return await self.install(resolved, InstallOptions(force=True))

    async def clean(self) -> int:
        removed = 0
        for name, skill in list(self._installed.items()):
            if not os.path.exists(skill.install_path):
                del self._installed[name]
                removed += 1

        if removed > 0:
            self._save_installed()

        return removed
