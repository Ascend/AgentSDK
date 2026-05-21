"""Archive extraction and creation utilities."""

import os
import tarfile
import zipfile
from typing import List, Optional


def extract_archive(archive_path: str, destination: str, strip_prefix: Optional[str] = None) -> List[str]:
    """Extract archive to destination.

    Args:
        archive_path: Path to archive file
        destination: Destination directory
        strip_prefix: Optional prefix to strip from file paths

    Returns:
        List of extracted file paths
    """
    os.makedirs(destination, exist_ok=True)
    extracted_files = []

    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                if strip_prefix and member.name.startswith(strip_prefix):
                    member.name = member.name[len(strip_prefix) :]
                if member.name:
                    tar.extract(member, destination)
                    extracted_files.append(os.path.join(destination, member.name))
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r:") as tar:
            members = tar.getmembers()
            for member in members:
                if strip_prefix and member.name.startswith(strip_prefix):
                    member.name = member.name[len(strip_prefix) :]
                if member.name:
                    tar.extract(member, destination)
                    extracted_files.append(os.path.join(destination, member.name))
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if strip_prefix and member.startswith(strip_prefix):
                    member = member[len(strip_prefix) :]
                if member:
                    zip_ref.extract(member, destination)
                    extracted_files.append(os.path.join(destination, member))
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    return extracted_files


def create_archive(source_dir: str, archive_path: str, archive_format: str = "gztar") -> str:
    """Create archive from directory.

    Args:
        source_dir: Source directory to archive
        archive_path: Output archive path
        archive_format: Archive format (gztar, tar, zip)

    Returns:
        Path to created archive
    """
    if archive_format == "gztar":
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
    elif archive_format == "tar":
        with tarfile.open(archive_path, "w:") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
    elif archive_format == "zip":
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zip_ref.write(file_path, arcname)
    else:
        raise ValueError(f"Unsupported format: {archive_format}")

    return archive_path


def list_archive_contents(archive_path: str) -> List[str]:
    """List contents of archive.

    Args:
        archive_path: Path to archive file

    Returns:
        List of file paths in archive
    """
    contents = []

    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            contents = [m.name for m in tar.getmembers() if m.name]
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r:") as tar:
            contents = [m.name for m in tar.getmembers() if m.name]
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            contents = zip_ref.namelist()
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    return contents
