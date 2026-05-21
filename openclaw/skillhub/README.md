# SkillHub CLI

A decentralized skill management CLI for discovering, installing, and managing AI skills from Git hosting platforms (GitHub, Gitee, GitCode).

## Features

- **Multi-Platform Support**: Works with GitHub, Gitee, and GitCode
- **Skill Discovery**: Search and discover skills from multiple sources
- **Dependency Management**: Automatic dependency resolution
- **Secure Authentication**: Secure token storage using system keyring
- **Caching**: Intelligent caching for improved performance
- **Rich CLI**: Beautiful terminal output with progress bars and tables

## Installation

### From PyPI (when published)

```bash
pip install skillhub-cli
```

### From Source

```bash
git clone https://github.com/your-org/skillhub.git
cd skillhub
pip install -e .
```

## Quick Start

```bash
# Authenticate with GitHub
skillhub auth login github --token YOUR_TOKEN

# Search for skills
skillhub search "data-processing"

# Install a skill
skillhub install data-cleaner

# List installed skills
skillhub list installed

# Upgrade a skill
skillhub upgrade data-cleaner

# Uninstall a skill
skillhub uninstall data-cleaner
```

## Configuration

SkillHub uses the following directories:

- **Config**: `~/.config/skillhub/`
- **Data**: `~/.local/share/skillhub/`
- **Cache**: `~/.cache/skillhub/`

## Development

### Setup Development Environment

```bash
# Install Poetry if not already installed
pip install poetry

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .

# Run type checking
poetry run mypy .
```

### Project Structure

```text
skillhub/
├── skillhub/              # Main package
│   ├── adapters/          # Platform adapters (GitHub, Gitee, GitCode)
│   ├── commands/          # CLI commands
│   ├── interfaces/        # Abstract base classes
│   ├── models/           # Pydantic models
│   ├── services/         # Business logic (implementations)
│   ├── cli.py            # CLI entry point
│   └── config.py         # Configuration management
├── tests/                # Test suite
├── docs/                 # Documentation
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Architecture

The SkillHub CLI follows a layered architecture:

1. **CLI Layer**: Typer-based command-line interface with Rich for terminal output
2. **Service Layer**: Business logic for source management, skill resolution, installation
3. **Adapter Layer**: Platform-specific implementations for GitHub, Gitee, GitCode
4. **Model Layer**: Pydantic models for type-safe data handling

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the [Contributing Guide](https://gitcode.com/Ascend/AgentSDK/blob/master/contributing.md) for details on how to get started.

## Support

- **Issues**: [GitCode Issues](https://gitcode.com/Ascend/AgentSDK/issues)
