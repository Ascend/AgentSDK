---
name: mineru-to-markdown
description: Convert PDF, images, and Office files (doc/docx/xls/xlsx/ppt/pptx) to Markdown through an already running MinerU FastAPI service. Use when users request document parsing, OCR extraction, or converting files to markdown/md with MinerU.
---

# MinerU To Markdown

## When To Use

Use this skill when the user asks to convert files into Markdown, especially for:

- PDF to Markdown
- image to Markdown
- Word/Excel/PPT to Markdown
- batch conversion from a directory

This skill assumes a MinerU FastAPI service is already running and reachable.

## Inputs

Collect these inputs before running conversion:

- `input_path`: required, one file or one directory
- `output_dir`: required, output directory for extracted results
- `api_url`: optional, default `http://127.0.0.1:8000`
- `backend`: optional, default `hybrid-auto-engine`
- `parse_method`: optional, default `auto` (`auto`/`txt`/`ocr`)
- `language`: optional, default `ch`
- `formula_enable`: optional, default `true`
- `table_enable`: optional, default `true`
- `start_page_id`: optional, default `0`
- `end_page_id`: optional, default empty (parse to last page)

## Run

Run:

```bash
python SKILL_DIR/skills/mineru-to-markdown/scripts/convert.py \
  --input-path "<input_path>" \
  --output-dir "<output_dir>" \
  --api-url "<api_url>"
```

For advanced options:

```bash
python SKILL_DIR/skills/mineru-to-markdown/scripts/convert.py \
  --input-path "<input_path>" \
  --output-dir "<output_dir>" \
  --api-url "http://127.0.0.1:8000" \
  --backend "hybrid-auto-engine" \
  --parse-method "auto" \
  --language "ch" \
  --formula-enable \
  --table-enable \
  --start-page-id 0
```

## Execution Rules

1. Validate input path exists and file types are supported.
2. Ensure API is reachable through `api_url`.
3. Run conversion script once with explicit paths.
4. Return:
   - submitted task id
   - conversion status summary
   - extracted output directory
5. If failure occurs, include direct action guidance.

## Supported Formats

The script supports MinerU-supported suffixes from PDF, image, and Office groups, including common formats:

- PDF: `.pdf`
- image: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` (if enabled by MinerU)
- Office: `.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`

## Failure Handling

- Input missing or unsupported: ask user to provide supported file(s).
- API unreachable: ask user to start/check MinerU FastAPI and verify `api_url`.
- Task failure: surface the error and suggest retry with:
  - smaller test file
  - different `parse_method`
  - valid page range

## Additional Resources

- Detailed usage and troubleshooting: [README.md](README.md)
