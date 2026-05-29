#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""MinerU document conversion client for batch processing files to Markdown."""

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import httpx

from mineru.cli import api_client as _miner_api
from mineru.cli.common import image_suffixes, office_suffixes, pdf_suffixes
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for document conversion operations."""

    api_endpoint: str = "http://127.0.0.1:8000"
    engine_backend: str = "hybrid-auto-engine"
    parsing_strategy: str = "auto"
    ocr_lang: str = "ch"
    parse_formulas: bool = True
    parse_tables: bool = True
    page_start: int = 0
    page_end: Annotated[int | None, "inclusive"] = None


@dataclass
class FileBatch:
    """Represents a batch of input files to process."""

    sources: list[Path] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.sources)

    @property
    def description(self) -> str:
        return f"{self.size} file(s)"


@dataclass
class ProcessingResult:
    """Holds the outcome of a conversion operation."""

    output_directory: Path | None = None
    downloaded_artifact: Path | None = None


def scan_for_processable_files(root_path: str | Path) -> FileBatch:
    """Walk a path and collect all files that can be processed."""
    target = Path(root_path).expanduser().resolve()

    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    if target.is_file():
        suffix = guess_suffix_by_path(target)
        allowed = set(pdf_suffixes + image_suffixes + office_suffixes)
        if suffix not in allowed:
            raise ValueError(f"File type not supported: {target.name}")
        return FileBatch(sources=[target])

    if not target.is_dir():
        raise ValueError(f"Path must be a file or directory: {target}")

    allowed = set(pdf_suffixes + image_suffixes + office_suffixes)
    discovered = sorted(
        (item.resolve() for item in target.iterdir() if item.is_file() and guess_suffix_by_path(item) in allowed),
        key=lambda p: p.name,
    )

    if not discovered:
        raise ValueError(f"No processable files under: {target}")

    return FileBatch(sources=discovered)


def assemble_request_payload(config: ConversionConfig) -> dict:
    """Construct the API request payload from configuration."""
    return _miner_api.build_parse_request_form_data(
        lang_list=[config.ocr_lang],
        backend=config.engine_backend,
        parse_method=config.parsing_strategy,
        formula_enable=config.parse_formulas,
        table_enable=config.parse_tables,
        server_url=None,
        start_page_id=config.page_start,
        end_page_id=config.page_end,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=True,
        response_format_zip=True,
        return_original_file=False,
    )


def render_task_status(snapshot: _miner_api.TaskStatusSnapshot) -> str:
    """Convert a status snapshot into a human-readable message."""
    if snapshot.queued_ahead is None:
        return snapshot.status
    return f"{snapshot.status} (queued_ahead={snapshot.queued_ahead})"


class MinerUProcessor:
    """Handles the lifecycle of a batch document conversion."""

    def __init__(self, config: ConversionConfig):
        self.config = config
        self._state_cache: str | None = None

    async def run(self, source_batch: FileBatch, destination: Path) -> ProcessingResult:
        """Execute the full conversion pipeline."""
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        payload = assemble_request_payload(self.config)
        upload_items = [_miner_api.UploadAsset(path=f, upload_name=f.name) for f in source_batch.sources]
        result = ProcessingResult(output_directory=destination)

        async with httpx.AsyncClient(
            timeout=_miner_api.build_http_timeout(),
            follow_redirects=True,
        ) as client:
            health = await _miner_api.fetch_server_health(
                client,
                _miner_api.normalize_base_url(self.config.api_endpoint),
            )
            print(f"Using API: {health.base_url}")
            print(f"Submitting {source_batch.size} file(s)")

            submission = await _miner_api.submit_parse_task(
                base_url=health.base_url,
                upload_assets=upload_items,
                form_data=payload,
            )
            print(f"task_id: {submission.task_id}")
            if submission.queued_ahead is not None:
                print(f"status: pending (queued_ahead={submission.queued_ahead})")

            await _miner_api.wait_for_task_result(
                client=client,
                submit_response=submission,
                task_label=source_batch.description,
                status_snapshot_callback=self._emit_status,
            )
            print("status: completed")

            result.downloaded_artifact = await _miner_api.download_result_zip(
                client=client,
                submit_response=submission,
                task_label=source_batch.description,
            )

        if result.downloaded_artifact is not None:
            try:
                _miner_api.safe_extract_zip(result.downloaded_artifact, destination)
            finally:
                result.downloaded_artifact.unlink(missing_ok=True)
            print(f"Extracted result to: {destination}")

        return result

    def _emit_status(self, snapshot: _miner_api.TaskStatusSnapshot) -> None:
        """Callback to print status updates, avoiding duplicates."""
        message = render_task_status(snapshot)
        if message != self._state_cache:
            self._state_cache = message
            print(f"status: {message}")


def parse_cli_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the converter."""
    parser = argparse.ArgumentParser(
        description="Batch-convert PDF/image/Office documents to Markdown via MinerU FastAPI."
    )
    parser.add_argument("--input-path", required=True, help="Source file or directory path.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for Markdown output.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="MinerU FastAPI base URL.")
    parser.add_argument("--backend", default="hybrid-auto-engine", help="MinerU backend engine.")
    parser.add_argument(
        "--parse-method",
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="Text extraction strategy.",
    )
    parser.add_argument("--language", default="ch", help="OCR language hint.")
    parser.add_argument("--formula-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--table-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-page-id", type=int, default=0, help="Zero-based start page index.")
    parser.add_argument("--end-page-id", type=int, default=None, help="Zero-based end page index (inclusive).")
    return parser.parse_args()


def main() -> None:
    """Entry point for the conversion client."""
    args = parse_cli_arguments()

    cfg = ConversionConfig(
        api_endpoint=args.api_url,
        engine_backend=args.backend,
        parsing_strategy=args.parse_method,
        ocr_lang=args.language,
        parse_formulas=args.formula_enable,
        parse_tables=args.table_enable,
        page_start=args.start_page_id,
        page_end=args.end_page_id,
    )

    files = scan_for_processable_files(args.input_path)
    processor = MinerUProcessor(cfg)
    asyncio.run(processor.run(files, Path(args.output_dir)))


if __name__ == "__main__":
    main()
