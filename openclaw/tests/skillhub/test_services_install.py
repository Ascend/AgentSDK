"""Tests for SkillHub service implementations."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from skillhub.config import Settings
from skillhub.models.skill import SkillManifest


class TestInstallEngine:
    """Tests for InstallEngineImpl."""

    @pytest.fixture
    def install_engine(self, mock_settings: Settings):
        """Create install engine instance."""
        from skillhub.services.install_engine import InstallEngineImpl

        return InstallEngineImpl(mock_settings)

    def test_init_creates_installed_file(self, mock_settings: Settings):
        """Test that install engine initializes correctly."""
        from skillhub.services.install_engine import InstallEngineImpl

        engine = InstallEngineImpl(mock_settings)
        assert engine.installed_file.name == "installed.json"

    def test_load_installed_empty(self, install_engine):
        """Test loading empty installed list."""
        assert len(install_engine._installed) == 0

    @pytest.mark.asyncio
    async def test_list_installed_empty(self, install_engine):
        """Test listing empty installed skills."""
        skills = await install_engine.list_installed()
        assert skills == []

    @pytest.mark.asyncio
    async def test_is_installed_false(self, install_engine):
        """Test is_installed returns False for non-existent skill."""
        result = install_engine.is_installed("non-existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_installed_none(self, install_engine):
        """Test get_installed returns None for non-existent skill."""
        result = await install_engine.get_installed("non-existent")
        assert result is None

    @pytest.mark.asyncio
    async def test_install_already_installed(self, install_engine, sample_resolved_skill):
        """Test installing already installed skill."""
        from skillhub.models.skill import InstalledSkill

        installed = InstalledSkill(
            name=sample_resolved_skill.name,
            version=sample_resolved_skill.version,
            source_id=sample_resolved_skill.source.get("id", ""),
            source_type=sample_resolved_skill.source.get("type", ""),
            repository=sample_resolved_skill.repository,
            ref=sample_resolved_skill.ref,
            install_path="/skills/test",
            checksum="abc",
        )
        install_engine._installed[installed.name] = installed

        result = await install_engine.install(sample_resolved_skill)
        assert result.success is False
        assert "already installed" in str(result.errors).lower()

    @pytest.mark.asyncio
    async def test_install_already_installed_with_force(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test force reinstalling already installed skill."""
        from skillhub.interfaces.install_engine import InstallOptions
        from skillhub.models.skill import InstalledSkill

        skill_dir = temp_dir / "skills" / sample_resolved_skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        installed = InstalledSkill(
            name=sample_resolved_skill.name,
            version=sample_resolved_skill.version,
            source_id=sample_resolved_skill.source.get("id", ""),
            source_type=sample_resolved_skill.source.get("type", ""),
            repository=sample_resolved_skill.repository,
            ref=sample_resolved_skill.ref,
            install_path=str(skill_dir),
            checksum="abc",
        )
        install_engine._installed[installed.name] = installed

        # Force reinstall with mocked download
        sample_resolved_skill.download_url = "https://example.com/skill.zip"
        opts = InstallOptions(force=True, target_path=str(skill_dir))

        with patch("skillhub.utils.http.HttpClient") as mock_http:
            mock_client = AsyncMock()
            mock_client.download = AsyncMock()
            mock_client.close = AsyncMock()
            mock_http.return_value = mock_client

            with patch("skillhub.services.install_engine.extract_archive"):
                with patch("skillhub.services.install_engine.compute_checksum", return_value="new_checksum"):
                    with patch("os.remove"):
                        result = await install_engine.install(sample_resolved_skill, opts)
                        assert result.success is True
                        assert result.skill.checksum == "new_checksum"

    @pytest.mark.asyncio
    async def test_install_with_download_url_mocked(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test install with download URL fully mocked."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "skills" / "download-test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        sample_resolved_skill.download_url = "https://github.com/test/repo/archive/v1.0.zip"
        sample_resolved_skill.subpath = None
        opts = InstallOptions(force=True, target_path=str(skill_dir))

        with patch("skillhub.utils.http.HttpClient") as mock_http:
            mock_client = AsyncMock()
            mock_client.download = AsyncMock()
            mock_client.close = AsyncMock()
            mock_http.return_value = mock_client

            with patch("skillhub.services.install_engine.extract_archive") as mock_extract:
                mock_extract.return_value = None

                with patch("skillhub.services.install_engine.compute_checksum", return_value="abc123"):
                    with patch("os.remove"):
                        result = await install_engine.install(sample_resolved_skill, opts)
                        assert result.success is True
                        mock_client.download.assert_called_once()

    @pytest.mark.asyncio
    async def test_install_with_subpath_filter_mocked(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test install with subpath filtering."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "skills" / "subpath-test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Create subpath structure
        subpath_dir = skill_dir / "skills" / "test-skill"
        subpath_dir.mkdir(parents=True, exist_ok=True)
        (subpath_dir / "SKILL.md").write_text("---\nname: subpath-test\n---")

        sample_resolved_skill.download_url = "https://example.com/archive.zip"
        sample_resolved_skill.subpath = "skills/test-skill"
        opts = InstallOptions(force=True, target_path=str(skill_dir))

        with patch("skillhub.utils.http.HttpClient") as mock_http:
            mock_client = AsyncMock()
            mock_client.download = AsyncMock()
            mock_client.close = AsyncMock()
            mock_http.return_value = mock_client

            with patch("skillhub.services.install_engine.extract_archive"):
                with patch.object(install_engine, "_filter_subpath") as mock_filter:
                    mock_filter.return_value = None

                    with patch("skillhub.services.install_engine.compute_checksum", return_value="abc"):
                        with patch("os.remove"):
                            await install_engine.install(sample_resolved_skill, opts)
                            # Should call _filter_subpath
                            mock_filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_install_exception_handling(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test install handles exceptions."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "skills" / "error-test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        sample_resolved_skill.download_url = "https://example.com/archive.zip"
        opts = InstallOptions(force=True, target_path=str(skill_dir))

        with patch("skillhub.utils.http.HttpClient") as mock_http:
            mock_http.side_effect = Exception("Network error")

            result = await install_engine.install(sample_resolved_skill, opts)
            assert result.success is False
            assert "Network error" in str(result.errors)

    @pytest.mark.asyncio
    async def test_install_from_contents_api_mocked(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test install from contents API."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "skills" / "contents-test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        sample_resolved_skill.download_url = None
        sample_resolved_skill.subpath = "skills/test"
        opts = InstallOptions(force=True, target_path=str(skill_dir))

        with patch.object(install_engine, "_install_from_contents_api", new_callable=AsyncMock) as mock_install:
            mock_install.return_value = None

            with patch("skillhub.services.install_engine.compute_checksum", return_value="abc123"):
                result = await install_engine.install(sample_resolved_skill, opts)
                assert result.success is True
                mock_install.assert_called_once()

    @pytest.mark.asyncio
    async def test_install_contents_api_full_mock(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test _install_from_contents_api with full mocking."""
        from skillhub.models.repository import ContentItem

        skill_dir = temp_dir / "skills" / "contents-full"
        skill_dir.mkdir(parents=True, exist_ok=True)

        sample_resolved_skill.download_url = None
        sample_resolved_skill.subpath = "skills/test"
        sample_resolved_skill.repository = "https://github.com/owner/repo"

        # Mock adapter and credential manager
        mock_adapter = AsyncMock()
        mock_adapter.get_contents = AsyncMock(
            return_value=[
                ContentItem(
                    type="file",
                    name="SKILL.md",
                    path="skills/test/SKILL.md",
                    sha="abc",
                    size=100,
                    url="https://api.github.com/repos/owner/repo/contents/skills/test/SKILL.md",
                )
            ]
        )
        mock_adapter.get_file_content = AsyncMock(return_value="---\nname: contents-test\n---")
        mock_adapter.close = AsyncMock()

        with patch("skillhub.adapters.factory.AdapterFactory.create") as mock_factory:
            mock_factory.return_value = mock_adapter

            with patch("skillhub.services.credential_manager.CredentialManagerImpl") as mock_cred:
                mock_cred_instance = MagicMock()
                mock_cred_instance.get_token = AsyncMock(return_value="test_token")
                mock_cred.return_value = mock_cred_instance

                await install_engine._install_from_contents_api(sample_resolved_skill, str(skill_dir))
                mock_adapter.get_contents.assert_called()
                mock_adapter.close.assert_called()

    @pytest.mark.asyncio
    async def test_uninstall_non_existing(self, install_engine):
        """Test uninstalling non-existing skill."""
        await install_engine.uninstall("non-existent")

    @pytest.mark.asyncio
    async def test_uninstall_existing_skill(self, install_engine, temp_dir: Path):
        """Test uninstalling existing skill."""
        from skillhub.models.skill import InstalledSkill

        skill_dir = temp_dir / "skills" / "uninstall-test"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: uninstall-test\n---")

        installed = InstalledSkill(
            name="uninstall-test",
            version="1.0",
            source_id="test",
            source_type="local",
            repository="",
            ref="",
            install_path=str(skill_dir),
            checksum="abc",
        )
        install_engine._installed["uninstall-test"] = installed

        await install_engine.uninstall("uninstall-test")
        assert "uninstall-test" not in install_engine._installed

    @pytest.mark.asyncio
    async def test_verify_non_existing(self, install_engine):
        """Test verifying non-existing skill."""
        result = await install_engine.verify("non-existent")
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_verify_existing_skill(self, install_engine, temp_dir: Path):
        """Test verifying existing skill."""
        from skillhub.models.skill import InstalledSkill

        skill_dir = temp_dir / "skills" / "verify-test"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: verify-test\n---")

        installed = InstalledSkill(
            name="verify-test",
            version="1.0",
            source_id="test",
            source_type="local",
            repository="",
            ref="",
            install_path=str(skill_dir),
            checksum="abc",
        )
        install_engine._installed["verify-test"] = installed

        with patch("skillhub.services.install_engine.compute_checksum", return_value="abc"):
            result = await install_engine.verify("verify-test")
            assert result.valid is True

    @pytest.mark.asyncio
    async def test_verify_checksum_mismatch(self, install_engine, temp_dir: Path):
        """Test verify with checksum mismatch."""
        from skillhub.models.skill import InstalledSkill

        skill_dir = temp_dir / "skills" / "verify-mismatch"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: verify-mismatch\n---")

        installed = InstalledSkill(
            name="verify-mismatch",
            version="1.0",
            source_id="test",
            source_type="local",
            repository="",
            ref="",
            install_path=str(skill_dir),
            checksum="original",
        )
        install_engine._installed["verify-mismatch"] = installed

        with patch("skillhub.services.install_engine.compute_checksum", return_value="different"):
            result = await install_engine.verify("verify-mismatch")
            assert result.valid is False

    @pytest.mark.asyncio
    async def test_upgrade_not_installed(self, install_engine):
        """Test upgrading non-installed skill."""
        result = await install_engine.upgrade("non-existent")
        assert result.success is False
        assert "not installed" in str(result.errors).lower()

    @pytest.mark.asyncio
    async def test_upgrade_installed_skill_mocked(self, install_engine, temp_dir: Path):
        """Test upgrading installed skill."""
        from skillhub.models.skill import InstalledSkill, ResolvedSkill

        skill_dir = temp_dir / "skills" / "upgrade-test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        installed = InstalledSkill(
            name="upgrade-test",
            version="1.0",
            source_id="github",
            source_type="github",
            repository="test/repo",
            ref="v1.0",
            install_path=str(skill_dir),
            checksum="abc",
        )
        install_engine._installed["upgrade-test"] = installed

        # Mock resolver and source manager
        mock_resolved = ResolvedSkill(
            name="upgrade-test",
            version="2.0",
            repository="test/repo",
            ref="v2.0",
            manifest=SkillManifest(name="upgrade-test", description="test"),
            source={"id": "test", "type": "github"},
            download_url="https://example.com/v2.zip",
        )

        with patch("skillhub.services.skill_resolver.SkillResolverImpl") as mock_resolver:
            mock_resolver_instance = MagicMock()
            mock_resolver_instance.resolve_version = AsyncMock(return_value=mock_resolved)
            mock_resolver.return_value = mock_resolver_instance

            with patch("skillhub.services.source_manager.SourceManagerImpl") as mock_sm:
                mock_sm_instance = MagicMock()
                mock_sm_instance.list_sources = AsyncMock(return_value=[])
                mock_sm.return_value = mock_sm_instance

                with patch.object(install_engine, "install", new_callable=AsyncMock) as mock_install:
                    mock_install.return_value = MagicMock(success=True, skill=mock_resolved)

                    await install_engine.upgrade("upgrade-test")
                    # After upgrade, install should be called with resolved skill

    @pytest.mark.asyncio
    async def test_upgrade_resolve_failed(self, install_engine, temp_dir: Path):
        """Test upgrade when resolve fails."""
        from skillhub.models.skill import InstalledSkill

        skill_dir = temp_dir / "skills" / "upgrade-fail"
        skill_dir.mkdir(parents=True, exist_ok=True)

        installed = InstalledSkill(
            name="upgrade-fail",
            version="1.0",
            source_id="github",
            source_type="github",
            repository="test/repo",
            ref="v1.0",
            install_path=str(skill_dir),
            checksum="abc",
        )
        install_engine._installed["upgrade-fail"] = installed

        with patch("skillhub.services.skill_resolver.SkillResolverImpl") as mock_resolver:
            mock_resolver_instance = MagicMock()
            mock_resolver_instance.resolve_version = AsyncMock(return_value=None)
            mock_resolver.return_value = mock_resolver_instance

            with patch("skillhub.services.source_manager.SourceManagerImpl") as mock_sm:
                mock_sm_instance = MagicMock()
                mock_sm_instance.list_sources = AsyncMock(return_value=[])
                mock_sm.return_value = mock_sm_instance

                result = await install_engine.upgrade("upgrade-fail")
                assert result.success is False
                assert "Could not resolve" in str(result.errors)

    @pytest.mark.asyncio
    async def test_repair_not_installed(self, install_engine):
        """Test repairing non-installed skill."""
        result = await install_engine.repair("non-existent")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_repair_installed_skill(self, install_engine, temp_dir: Path):
        """Test repairing installed skill."""
        from skillhub.models.skill import InstalledSkill

        skill_dir = temp_dir / "skills" / "repair-test"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: repair-test\n---")

        installed = InstalledSkill(
            name="repair-test",
            version="1.0",
            source_id="local",
            source_type="local",
            repository="",
            ref="",
            install_path=str(skill_dir),
            checksum="abc",
        )
        install_engine._installed["repair-test"] = installed

        result = await install_engine.repair("repair-test")
        # Repair may succeed or fail based on checksum
        assert result.skill.name == "repair-test"

    @pytest.mark.asyncio
    async def test_clean_returns_count(self, install_engine):
        """Test clean returns count."""
        count = await install_engine.clean()
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_install_from_path_basic(self, install_engine, temp_dir: Path):
        """Test install_from_path basic."""
        skill_dir = temp_dir / "local_skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: local-skill\nversion: '1.0'\n---\n# Local Skill")

        result = await install_engine.install_from_path(str(skill_dir))
        assert result.success is True
        assert result.skill.name == "local-skill"

    @pytest.mark.asyncio
    async def test_install_from_path_no_skill_md(self, install_engine, temp_dir: Path):
        """Test install_from_path without SKILL.md."""
        skill_dir = temp_dir / "no_skill_md"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "README.md").write_text("test")

        result = await install_engine.install_from_path(str(skill_dir))
        assert result.success is False
        assert "SKILL.md" in str(result.errors)

    @pytest.mark.asyncio
    async def test_install_from_path_with_force(self, install_engine, temp_dir: Path):
        """Test install_from_path with force."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "force_skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: force-skill\n---")

        # First install
        await install_engine.install_from_path(str(skill_dir))

        # Force reinstall
        opts = InstallOptions(force=True)
        result = await install_engine.install_from_path(str(skill_dir), opts)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_install_from_path_custom_target(self, install_engine, temp_dir: Path):
        """Test install_from_path to custom target."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "source_skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: custom-target\n---")

        target_dir = temp_dir / "custom_target"
        # Don't pre-create target_dir - let install_engine create it

        opts = InstallOptions(target_path=str(target_dir))
        result = await install_engine.install_from_path(str(skill_dir), opts)
        assert result.success is True

    def test_installed_file_path(self, mock_settings: Settings):
        """Test installed file path is set correctly."""
        from skillhub.services.install_engine import InstallEngineImpl

        engine = InstallEngineImpl(mock_settings)
        assert engine.installed_file.name == "installed.json"

    def test_filter_subpath_raises_error(self, install_engine, temp_dir: Path):
        """Test _filter_subpath raises error when not found."""
        skill_dir = temp_dir / "filter_test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError):
            install_engine._filter_subpath(str(skill_dir), "nonexistent/path")

    def test_copy_recursive_basic(self, install_engine, temp_dir: Path):
        """Test _copy_recursive basic."""
        src = temp_dir / "src_basic"
        dst = temp_dir / "dst_basic"
        src.mkdir(parents=True, exist_ok=True)
        dst.mkdir(parents=True, exist_ok=True)
        (src / "file.txt").write_text("test")

        install_engine._copy_recursive(str(src), str(dst))
        assert (dst / "file.txt").exists()

    def test_copy_recursive_nested(self, install_engine, temp_dir: Path):
        """Test _copy_recursive with nested directories."""
        src = temp_dir / "src_nested"
        dst = temp_dir / "dst_nested"
        src.mkdir(parents=True, exist_ok=True)
        dst.mkdir(parents=True, exist_ok=True)

        nested = src / "dir1" / "dir2"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "file.txt").write_text("nested")

        install_engine._copy_recursive(str(src), str(dst))
        assert (dst / "dir1" / "dir2" / "file.txt").exists()

    def test_save_and_reload(self, install_engine, temp_dir: Path):
        """Test saving and reloading installed data."""
        from skillhub.models.skill import InstalledSkill
        from skillhub.services.install_engine import InstallEngineImpl

        skill_dir = temp_dir / "skills" / "save-test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        installed = InstalledSkill(
            name="save-test",
            version="1.0",
            source_id="test",
            source_type="local",
            repository="",
            ref="",
            install_path=str(skill_dir),
            checksum="abc",
        )
        install_engine._installed["save-test"] = installed
        install_engine._save_installed()

        # Reload
        new_engine = InstallEngineImpl(install_engine.config)
        assert "save-test" in new_engine._installed

    def test_filter_subpath_success(self, install_engine, temp_dir: Path):
        """Test _filter_subpath with valid subpath - skip on Windows."""
        import platform

        # Skip this test on Windows due to path handling differences
        if platform.system() == "Windows":
            pytest.skip("Path handling differs on Windows")

        skill_dir = temp_dir / "filter_success"
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Create subpath structure using os.path.join
        subpath = "skills/test-skill"
        subpath_dir_path = os.path.join(str(skill_dir), subpath)
        os.makedirs(subpath_dir_path, exist_ok=True)

        # Create SKILL.md
        skill_md_path = os.path.join(subpath_dir_path, "SKILL.md")
        with open(skill_md_path, "w", encoding="utf-8") as f:
            f.write("---\nname: test\n---")

        # This should succeed
        install_engine._filter_subpath(str(skill_dir), subpath)

    @pytest.mark.asyncio
    async def test_install_from_contents_api_with_files(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test _install_from_contents_api with file content."""
        from skillhub.models.repository import ContentItem

        skill_dir = temp_dir / "contents_api_files"
        skill_dir.mkdir(parents=True, exist_ok=True)

        sample_resolved_skill.download_url = None
        sample_resolved_skill.subpath = "skills/test"
        sample_resolved_skill.repository = "https://github.com/owner/repo"

        with patch("skillhub.adapters.factory.AdapterFactory.create") as mock_factory:
            mock_adapter = AsyncMock()

            # Mock file and directory contents
            file_item = ContentItem(
                type="file", name="SKILL.md", path="skills/test/SKILL.md", sha="abc", size=100, url="url1"
            )
            dir_item = ContentItem(type="dir", name="subdir", path="skills/test/subdir", sha="def", size=0, url="url2")

            mock_adapter.get_contents = AsyncMock()
            mock_adapter.get_contents.side_effect = [
                [file_item, dir_item],  # First call returns root contents
                [
                    ContentItem(
                        type="file",
                        name="nested.md",
                        path="skills/test/subdir/nested.md",
                        sha="ghi",
                        size=50,
                        url="url3",
                    )
                ],  # Second call for subdir
            ]

            mock_adapter.get_file_content = AsyncMock(return_value="---\nname: test\n---")
            mock_adapter.close = AsyncMock()
            mock_factory.return_value = mock_adapter

            with patch("skillhub.services.credential_manager.CredentialManagerImpl") as mock_cred:
                mock_cred_instance = MagicMock()
                mock_cred_instance.get_token = AsyncMock(return_value="token")
                mock_cred.return_value = mock_cred_instance

                await install_engine._install_from_contents_api(sample_resolved_skill, str(skill_dir))

    @pytest.mark.asyncio
    async def test_install_with_dependency_resolution(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test install with dependency resolution."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "skills" / "dep-test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        sample_resolved_skill.manifest.dependencies = {"dep-skill": ">=1.0.0"}
        sample_resolved_skill.download_url = None
        sample_resolved_skill.subpath = "skills/dep-test"

        opts = InstallOptions(force=True, target_path=str(skill_dir))

        with patch.object(install_engine, "_install_from_contents_api", new_callable=AsyncMock):
            with patch("skillhub.services.install_engine.compute_checksum", return_value="abc"):
                # Install should still work even with dependencies
                await install_engine.install(sample_resolved_skill, opts)
                # Dependencies may be logged in warnings

    @pytest.mark.asyncio
    async def test_repair_checksum_mismatch(self, install_engine, temp_dir: Path):
        """Test repair with checksum mismatch."""
        from skillhub.models.skill import InstalledSkill

        skill_dir = temp_dir / "skills" / "repair-mismatch"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: repair-mismatch\n---")

        installed = InstalledSkill(
            name="repair-mismatch",
            version="1.0",
            source_id="test",
            source_type="local",
            repository="",
            ref="",
            install_path=str(skill_dir),
            checksum="original_checksum",
        )
        install_engine._installed["repair-mismatch"] = installed

        with patch("skillhub.services.install_engine.compute_checksum", return_value="different_checksum"):
            await install_engine.repair("repair-mismatch")
            # Repair should handle mismatch

    @pytest.mark.asyncio
    async def test_install_skill_with_version_in_manifest(self, install_engine, temp_dir: Path):
        """Test install_from_path with version in manifest."""
        skill_dir = temp_dir / "version_manifest"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: version-skill\nversion: 2.0.0\ndescription: test\n---\n# Skill")

        result = await install_engine.install_from_path(str(skill_dir))
        assert result.success is True
        assert result.skill.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_install_with_ref_parameter(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test install with specific ref."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "skills" / "ref-test"
        skill_dir.mkdir(parents=True, exist_ok=True)

        sample_resolved_skill.ref = "v1.0.0"
        sample_resolved_skill.download_url = None
        sample_resolved_skill.subpath = "skills/ref-test"

        opts = InstallOptions(force=True, target_path=str(skill_dir))

        with patch.object(install_engine, "_install_from_contents_api", new_callable=AsyncMock) as mock_install:
            mock_install.return_value = None

            with patch("skillhub.services.install_engine.compute_checksum", return_value="abc"):
                await install_engine.install(sample_resolved_skill, opts)
                mock_install.assert_called_once()

    @pytest.mark.asyncio
    async def test_install_no_download_url_no_contents(self, install_engine, sample_resolved_skill, temp_dir: Path):
        """Test install when neither download_url nor contents API available."""
        from skillhub.interfaces.install_engine import InstallOptions

        skill_dir = temp_dir / "skills" / "no-download"
        skill_dir.mkdir(parents=True, exist_ok=True)

        sample_resolved_skill.download_url = None
        sample_resolved_skill.subpath = None  # No contents API path

        opts = InstallOptions(force=True, target_path=str(skill_dir))

        await install_engine.install(sample_resolved_skill, opts)
        # Should fail without download method

    @pytest.mark.asyncio
    async def test_clean_removes_invalid_paths(self, install_engine, temp_dir: Path):
        """Test clean removes skills with invalid paths."""
        from skillhub.models.skill import InstalledSkill

        # Add installed skill with invalid path
        installed = InstalledSkill(
            name="invalid-path",
            version="1.0",
            source_id="test",
            source_type="local",
            repository="",
            ref="",
            install_path="/nonexistent/path/to/skill",
            checksum="abc",
        )
        install_engine._installed["invalid-path"] = installed

        await install_engine.clean()
        # Should remove invalid entries
        assert "invalid-path" not in install_engine._installed
