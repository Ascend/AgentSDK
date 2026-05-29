# MinerU To Markdown Reference

A skill that converts PDF, image, and Office files to Markdown by calling an already running MinerU FastAPI service.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- MinerU environment is installed and available in current Python runtime.
- MinerU FastAPI service is already running(default: `http://127.0.0.1:8000`).
- `api_url` is reachable from the current machine.

Install dependencies:

```bash
pip install -r skills/mineru-to-markdown/requirements.txt
```

## Script Path

- `skills/mineru-to-markdown/scripts/convert.py`

## CLI Parameters

- `--input-path` (required): file path or directory path
- `--output-dir` (required): output directory for extracted result files
- `--api-url` (optional): MinerU FastAPI base URL, default `http://127.0.0.1:8000`
- `--backend` (optional): default `hybrid-auto-engine`
- `--parse-method` (optional): `auto` / `txt` / `ocr`, default `auto`
- `--language` (optional): OCR language hint, default `ch`
- `--formula-enable` / `--no-formula-enable` (optional): default enabled
- `--table-enable` / `--no-table-enable` (optional): default enabled
- `--start-page-id` (optional): zero-based start page, default `0`
- `--end-page-id` (optional): zero-based end page, default empty

## Output Contract

After completion:

- Result zip from API is downloaded temporarily.
- Zip is extracted into `output_dir`.
- Temporary zip is removed.

Typical output directory contains:

- Markdown files (`*.md`)
- image assets referenced by Markdown
- optional subdirectories grouped by file name/backend

## Examples

Single PDF:

```bash
python skills/mineru-to-markdown/scripts/convert.py \
  --input-path "/workspace/data/pdfs/demo1.pdf" \
  --output-dir "/workspace/data/MinerU-master/demo/api_output/demo1"
```

Single Office file:

```bash
python skills/mineru-to-markdown/scripts/convert.py \
  --input-path "/workspace/data/MinerU-master/demo/office_docs/docx_01.docx" \
  --output-dir "/workspace/data/MinerU-master/demo/api_output/docx_01"
```

Image file:

```bash
python skills/mineru-to-markdown/scripts/convert.py \
  --input-path "/workspace/data/MinerU-master/demo/example.png" \
  --output-dir "/workspace/data/MinerU-master/demo/api_output/image_01"
```

Directory batch:

```bash
python skills/mineru-to-markdown/scripts/convert.py \
  --input-path "/workspace/data/MinerU-master/demo/office_docs" \
  --output-dir "/workspace/data/MinerU-master/demo/api_output/office_batch"
```

## Troubleshooting

`Input path does not exist`:

- Check `--input-path`.
- Use absolute path to avoid shell cwd issues.

`Unsupported input file type`:

- Ensure file extension is MinerU-supported (PDF/image/Office).
- For uncommon types, convert to supported format first.

Connection or timeout errors:

- Confirm FastAPI service is running.
- Verify `--api-url` and port.
- Check service logs for model loading or GPU memory issues.

Task remains pending for long time:

- Retry with one small file first.
- Lower parse complexity: use `--parse-method txt` if OCR is unnecessary.
- Reduce page range using `--start-page-id` and `--end-page-id`.

Result looks incomplete:

- Switch parse strategy (`auto` -> `ocr` or `txt`).
- Enable table/formula options if needed.
- Validate source file quality (scanned image clarity, page rotation).
