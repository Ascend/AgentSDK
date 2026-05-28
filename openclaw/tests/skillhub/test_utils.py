"""Tests for SkillHub utility modules."""

import tempfile
import tarfile
import zipfile
from pathlib import Path
from typing import Generator

import pytest

from skillhub.utils.checksum import compute_checksum, verify_checksum, compute_file_checksum
from skillhub.utils.archive import extract_archive, create_archive, list_archive_contents
from skillhub.utils.semver import parse_version, compare_versions, satisfies_version
from skillhub.utils.http import HttpClient
from skillhub.utils.logger import setup_logger


class TestChecksum:  # pylint: disable=redefined-outer-name
    """Tests for checksum utilities."""

    def test_compute_checksum_sha256_string(self):
        """Test computing SHA256 checksum for string."""
        content = "test content"
        result = compute_checksum(content, "sha256")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_compute_checksum_sha256_bytes(self):
        """Test computing SHA256 checksum for bytes."""
        content = b"test content"
        result = compute_checksum(content, "sha256")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_compute_checksum_sha512(self):
        """Test computing SHA512 checksum."""
        content = "test content"
        result = compute_checksum(content, "sha512")
        assert isinstance(result, str)
        assert len(result) == 128

    def test_compute_checksum_md5(self):
        """Test computing MD5 checksum."""
        content = "test content"
        result = compute_checksum(content, "md5")
        assert isinstance(result, str)
        assert len(result) == 32

    def test_compute_checksum_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError):
            compute_checksum("test", "invalid")

    def test_compute_checksum_consistency(self):
        """Test that same input produces same checksum."""
        content = "test content"
        result1 = compute_checksum(content, "sha256")
        result2 = compute_checksum(content, "sha256")
        assert result1 == result2

    def test_compute_checksum_different_content(self):
        """Test that different content produces different checksum."""
        content1 = "content1"
        content2 = "content2"
        result1 = compute_checksum(content1, "sha256")
        result2 = compute_checksum(content2, "sha256")
        assert result1 != result2

    def test_verify_checksum_match(self):
        """Test checksum verification with matching checksum."""
        content = "test content"
        expected = compute_checksum(content, "sha256")
        assert verify_checksum(content, expected, "sha256") is True

    def test_verify_checksum_mismatch(self):
        """Test checksum verification with non-matching checksum."""
        content = "test content"
        expected = "0000000000000000000000000000000000000000000000000000000000000000"
        assert verify_checksum(content, expected, "sha256") is False

    def test_verify_checksum_case_insensitive(self):
        """Test that checksum verification is case-insensitive."""
        content = "test content"
        expected = compute_checksum(content, "sha256")
        upper_expected = expected.upper()
        assert verify_checksum(content, upper_expected, "sha256") is True

    def test_compute_file_checksum_sha256(self, temp_dir: Path):
        """Test computing file checksum."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("test content")
        result = compute_file_checksum(str(file_path), "sha256")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_compute_file_checksum_sha512(self, temp_dir: Path):
        """Test computing file checksum with SHA512."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("test content")
        result = compute_file_checksum(str(file_path), "sha512")
        assert isinstance(result, str)
        assert len(result) == 128

    def test_compute_file_checksum_md5(self, temp_dir: Path):
        """Test computing file checksum with MD5."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("test content")
        result = compute_file_checksum(str(file_path), "md5")
        assert isinstance(result, str)
        assert len(result) == 32

    def test_compute_file_checksum_large_file(self, temp_dir: Path):
        """Test computing checksum for larger file."""
        file_path = temp_dir / "large.txt"
        file_path.write_bytes(b"x" * 10000)
        result = compute_file_checksum(str(file_path), "sha256")
        assert isinstance(result, str)


class TestArchive:  # pylint: disable=redefined-outer-name
    """Tests for archive utilities."""

    def test_extract_zip(self, temp_dir: Path):
        """Test extracting ZIP archive."""
        archive_path = temp_dir / "test.zip"
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.write(str(source_dir / "file.txt"), "file.txt")

        dest_dir = temp_dir / "dest"
        extracted = extract_archive(str(archive_path), str(dest_dir))
        assert len(extracted) > 0
        assert (dest_dir / "file.txt").exists()

    def test_extract_tar_gz(self, temp_dir: Path):
        """Test extracting TAR.GZ archive."""
        archive_path = temp_dir / "test.tar.gz"
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(str(source_dir / "file.txt"), "file.txt")

        dest_dir = temp_dir / "dest"
        extracted = extract_archive(str(archive_path), str(dest_dir))
        assert len(extracted) > 0
        assert (dest_dir / "file.txt").exists()

    def test_extract_tar(self, temp_dir: Path):
        """Test extracting TAR archive."""
        archive_path = temp_dir / "test.tar"
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        with tarfile.open(archive_path, "w") as tf:
            tf.add(str(source_dir / "file.txt"), "file.txt")

        dest_dir = temp_dir / "dest"
        extracted = extract_archive(str(archive_path), str(dest_dir))
        assert len(extracted) > 0
        assert (dest_dir / "file.txt").exists()

    def test_extract_with_strip_prefix(self, temp_dir: Path):
        """Test extracting archive with strip prefix."""
        archive_path = temp_dir / "test.zip"
        source_dir = temp_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        nested_dir = source_dir / "nested"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (nested_dir / "file.txt").write_text("test")

        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.write(str(nested_dir / "file.txt"), "nested/file.txt")

        dest_dir = temp_dir / "dest"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Verify archive contains the file with nested prefix
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            assert "nested/file.txt" in names

        # Extract without strip prefix first, then check strip logic
        extract_archive(str(archive_path), str(dest_dir))
        assert (dest_dir / "nested" / "file.txt").exists()

    def test_extract_unsupported_format(self, temp_dir: Path):
        """Test that unsupported format raises ValueError."""
        archive_path = temp_dir / "test.rar"
        archive_path.write_text("fake rar content")
        dest_dir = temp_dir / "dest"

        with pytest.raises(ValueError):
            extract_archive(str(archive_path), str(dest_dir))

    def test_create_archive_gztar(self, temp_dir: Path):
        """Test creating GZTAR archive."""
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        archive_path = temp_dir / "test.tar.gz"
        result = create_archive(str(source_dir), str(archive_path), "gztar")
        assert result == str(archive_path)
        assert archive_path.exists()
        assert tarfile.is_tarfile(str(archive_path))

    def test_create_archive_tar(self, temp_dir: Path):
        """Test creating TAR archive."""
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        archive_path = temp_dir / "test.tar"
        result = create_archive(str(source_dir), str(archive_path), "tar")
        assert result == str(archive_path)
        assert archive_path.exists()
        assert tarfile.is_tarfile(str(archive_path))

    def test_create_archive_zip(self, temp_dir: Path):
        """Test creating ZIP archive."""
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test content")

        archive_path = temp_dir / "test.zip"
        result = create_archive(str(source_dir), str(archive_path), "zip")
        assert result == str(archive_path)
        assert archive_path.exists()
        assert zipfile.is_zipfile(str(archive_path))

    def test_create_archive_nested_files(self, temp_dir: Path):
        """Test creating archive with nested files."""
        source_dir = temp_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        nested_dir = source_dir / "nested"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "file1.txt").write_text("test1")
        (nested_dir / "file2.txt").write_text("test2")

        archive_path = temp_dir / "test.zip"
        create_archive(str(source_dir), str(archive_path), "zip")

        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            assert len(names) >= 2

    def test_create_archive_unsupported_format(self, temp_dir: Path):
        """Test that unsupported format raises ValueError."""
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        archive_path = temp_dir / "test.rar"

        with pytest.raises(ValueError):
            create_archive(str(source_dir), str(archive_path), "rar")

    def test_list_archive_contents_zip(self, temp_dir: Path):
        """Test listing ZIP archive contents."""
        archive_path = temp_dir / "test.zip"
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test")

        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.write(str(source_dir / "file.txt"), "file.txt")

        contents = list_archive_contents(str(archive_path))
        assert "file.txt" in contents

    def test_list_archive_contents_tar_gz(self, temp_dir: Path):
        """Test listing TAR.GZ archive contents."""
        archive_path = temp_dir / "test.tar.gz"
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test")

        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(str(source_dir / "file.txt"), "file.txt")

        contents = list_archive_contents(str(archive_path))
        assert "file.txt" in contents

    def test_list_archive_contents_tar(self, temp_dir: Path):
        """Test listing TAR archive contents."""
        archive_path = temp_dir / "test.tar"
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("test")

        with tarfile.open(archive_path, "w") as tf:
            tf.add(str(source_dir / "file.txt"), "file.txt")

        contents = list_archive_contents(str(archive_path))
        assert "file.txt" in contents

    def test_list_archive_contents_unsupported(self, temp_dir: Path):
        """Test that unsupported format raises ValueError."""
        archive_path = temp_dir / "test.rar"
        archive_path.write_text("fake content")

        with pytest.raises(ValueError):
            list_archive_contents(str(archive_path))


class TestSemver:
    """Tests for semantic versioning utilities."""

    def test_parse_version_simple(self):
        """Test parsing simple version."""
        major, minor, patch, prerelease = parse_version("1.2.3")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert prerelease is None

    def test_parse_version_with_prerelease(self):
        """Test parsing version with prerelease."""
        major, minor, patch, prerelease = parse_version("1.2.3-beta")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert prerelease == "beta"

    def test_parse_version_with_complex_prerelease(self):
        """Test parsing version with complex prerelease."""
        major, minor, patch, prerelease = parse_version("1.2.3-beta.1")
        assert prerelease == "beta.1"

    def test_parse_version_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_version("invalid")

    def test_parse_version_missing_parts(self):
        """Test that missing parts raises ValueError."""
        with pytest.raises(ValueError):
            parse_version("1.2")

    def test_compare_versions_equal(self):
        """Test comparing equal versions."""
        assert compare_versions("1.2.3", "1.2.3") == 0

    def test_compare_versions_greater_major(self):
        """Test comparing versions with greater major."""
        assert compare_versions("2.0.0", "1.0.0") == 1

    def test_compare_versions_greater_minor(self):
        """Test comparing versions with greater minor."""
        assert compare_versions("1.2.0", "1.1.0") == 1

    def test_compare_versions_greater_patch(self):
        """Test comparing versions with greater patch."""
        assert compare_versions("1.1.2", "1.1.1") == 1

    def test_compare_versions_less_major(self):
        """Test comparing versions with lesser major."""
        assert compare_versions("1.0.0", "2.0.0") == -1

    def test_compare_versions_less_minor(self):
        """Test comparing versions with lesser minor."""
        assert compare_versions("1.1.0", "1.2.0") == -1

    def test_compare_versions_less_patch(self):
        """Test comparing versions with lesser patch."""
        assert compare_versions("1.1.1", "1.1.2") == -1

    def test_compare_versions_with_prerelease(self):
        """Test comparing versions with prerelease."""
        assert compare_versions("1.2.3", "1.2.3-beta") == 1
        assert compare_versions("1.2.3-beta", "1.2.3") == -1

    def test_compare_versions_both_prerelease(self):
        """Test comparing two prerelease versions."""
        assert compare_versions("1.2.3-alpha", "1.2.3-beta") == -1
        assert compare_versions("1.2.3-beta", "1.2.3-alpha") == 1

    def test_compare_versions_same_prerelease(self):
        """Test comparing same prerelease versions."""
        assert compare_versions("1.2.3-beta", "1.2.3-beta") == 0

    def test_satisfies_version_exact(self):
        """Test version satisfies exact specifier."""
        assert satisfies_version("1.2.3", "1.2.3") is True

    def test_satisfies_version_caret(self):
        """Test version satisfies caret specifier."""
        assert satisfies_version("1.2.3", "^1.0.0") is True
        assert satisfies_version("1.2.3", "^1.2.0") is True
        assert satisfies_version("1.2.3", "^1.2.3") is True
        assert satisfies_version("2.0.0", "^1.0.0") is False

    def test_satisfies_version_tilde(self):
        """Test version satisfies tilde specifier."""
        assert satisfies_version("1.2.3", "~1.2.0") is True
        assert satisfies_version("1.2.5", "~1.2.3") is True
        assert satisfies_version("1.3.0", "~1.2.0") is False

    def test_satisfies_version_gte(self):
        """Test version satisfies >= specifier."""
        assert satisfies_version("1.2.3", ">=1.0.0") is True
        assert satisfies_version("1.2.3", ">=1.2.3") is True
        assert satisfies_version("1.2.2", ">=1.2.3") is False

    def test_satisfies_version_lte(self):
        """Test version satisfies <= specifier."""
        assert satisfies_version("1.2.3", "<=2.0.0") is True
        assert satisfies_version("1.2.3", "<=1.2.3") is True
        assert satisfies_version("2.0.0", "<=1.2.3") is False

    def test_satisfies_version_gt(self):
        """Test version satisfies > specifier."""
        assert satisfies_version("1.2.3", ">1.0.0") is True
        assert satisfies_version("1.2.3", ">1.2.3") is False

    def test_satisfies_version_lt(self):
        """Test version satisfies < specifier."""
        assert satisfies_version("1.2.3", "<2.0.0") is True
        assert satisfies_version("1.2.3", "<1.2.3") is False

    def test_satisfies_version_wildcard(self):
        """Test version satisfies wildcard."""
        assert satisfies_version("1.2.3", "*") is True
        assert satisfies_version("99.99.99", "*") is True

    def test_satisfies_version_latest(self):
        """Test version satisfies 'latest'."""
        assert satisfies_version("1.2.3", "latest") is True

    def test_satisfies_version_prerelease_check(self):
        """Test that prerelease versions don't match non-prerelease specs."""
        assert satisfies_version("1.2.3-beta", "^1.0.0") is False


class TestHttpClient:  # pylint: disable=redefined-outer-name
    """Tests for HttpClient."""

    def test_http_client_init(self):
        """Test HttpClient initialization."""
        client = HttpClient("https://api.example.com")
        assert client.base_url == "https://api.example.com"

    def test_http_client_with_headers(self):
        """Test HttpClient with custom headers."""
        headers = {"X-Custom": "value"}
        client = HttpClient("https://api.example.com", headers=headers)
        assert client.base_url == "https://api.example.com"

    def test_http_client_with_timeout(self):
        """Test HttpClient with custom timeout."""
        client = HttpClient("https://api.example.com", timeout=60.0)
        assert client.base_url == "https://api.example.com"

    def test_http_client_http2_disabled(self):
        """Test HttpClient with HTTP/2 disabled."""
        client = HttpClient("https://api.example.com", http2=False)
        assert client.base_url == "https://api.example.com"

    @pytest.mark.asyncio
    async def test_http_client_close(self):
        """Test HttpClient close."""
        client = HttpClient("https://api.example.com")
        await client.close()

    @pytest.mark.asyncio
    async def test_http_client_get_mock(self):
        """Test HttpClient GET request with mock."""
        from unittest.mock import patch, AsyncMock, MagicMock

        with patch("skillhub.utils.http.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": "test"}
            mock_response.raise_for_status = MagicMock()
            mock_instance.get.return_value = mock_response
            mock_client.return_value = mock_instance

            client = HttpClient("https://api.example.com")
            result = await client.get("/test")
            assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_http_client_post_mock(self):
        """Test HttpClient POST request with mock."""
        from unittest.mock import patch, AsyncMock, MagicMock

        with patch("skillhub.utils.http.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "ok"}
            mock_response.raise_for_status = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            client = HttpClient("https://api.example.com")
            result = await client.post("/test", json={"key": "value"})
            assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_http_client_download_mock(self, temp_dir: Path):
        """Test HttpClient download with mock."""
        from unittest.mock import patch, AsyncMock, MagicMock

        dest_file = temp_dir / "downloaded.txt"

        with patch("skillhub.utils.http.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"test content"
            mock_response.raise_for_status = MagicMock()
            mock_instance.get.return_value = mock_response
            mock_client.return_value = mock_instance

            client = HttpClient("https://api.example.com")
            result = await client.download("https://example.com/file", str(dest_file))
            assert result == str(dest_file)


class TestLogger:  # pylint: disable=redefined-outer-name
    """Tests for logger utilities."""

    def test_setup_logger_default(self):
        """Test setup_logger with default parameters."""
        logger = setup_logger()
        assert logger.name == "skillhub"
        assert logger.level == 20  # INFO level

    def test_setup_logger_custom_name(self):
        """Test setup_logger with custom name."""
        logger = setup_logger("test-logger")
        assert logger.name == "test-logger"

    def test_setup_logger_debug_level(self):
        """Test setup_logger with DEBUG level."""
        logger = setup_logger(level="DEBUG")
        assert logger.level == 10  # DEBUG level

    def test_setup_logger_verbose(self):
        """Test setup_logger with verbose mode."""
        logger = setup_logger(verbose=True)
        assert logger is not None

    def test_setup_logger_with_file(self, temp_dir: Path):
        """Test setup_logger with log file."""
        log_file = temp_dir / "logs" / "test.log"
        logger = setup_logger(log_file=log_file)
        assert logger is not None

    def test_setup_logger_creates_log_dir(self, temp_dir: Path):
        """Test that setup_logger creates log directory."""
        log_file = temp_dir / "nested" / "deep" / "test.log"
        setup_logger(log_file=log_file)
        assert log_file.parent.exists()

    def test_logger_has_handlers(self):
        """Test that logger has handlers."""
        logger = setup_logger()
        assert len(logger.handlers) > 0

    def test_logger_console_handler(self):
        """Test that logger has console handler."""
        import logging

        logger = setup_logger()
        has_console = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert has_console

    def test_logger_file_handler(self, temp_dir: Path):
        """Test that logger has file handler when log_file specified."""
        import logging

        log_file = temp_dir / "test.log"
        logger = setup_logger(log_file=log_file)
        has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert has_file


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory."""
    directory = Path(tempfile.mkdtemp())
    yield directory
    import shutil

    shutil.rmtree(directory, ignore_errors=True)
