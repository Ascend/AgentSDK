"""Configuration commands."""

import json
import typer
from rich.console import Console
from rich.table import Table

from skillhub.config import get_config, set_config_value

app = typer.Typer()
console = Console()


@app.command("get")
def config_get(
    key: str,
) -> None:
    """Get a configuration value."""
    config = get_config()

    keys = key.split(".")
    value = config
    for k in keys:
        if hasattr(value, k):
            value = getattr(value, k)
        else:
            console.print(f"[red]Configuration key not found: {key}[/red]")
            raise typer.Exit(1)

    if isinstance(value, (dict, list)):
        console.print(json.dumps(value, indent=2, default=str))
    else:
        console.print(f"[cyan]{key}[/cyan] = [green]{value}[/green]")


@app.command("set")
def config_set(
    key: str,
    value: str,
) -> None:
    """Set a configuration value."""
    try:
        set_config_value(key, value)
        console.print("[green]Configuration updated:[/green]")
        console.print(f"  [cyan]{key}[/cyan] = [green]{value}[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to save configuration: {e}[/red]")
        raise typer.Exit(1)


@app.command("list")
def config_list() -> None:
    """List all configuration values."""
    config = get_config()

    table = Table(title="Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    config_dict = config.model_dump()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                table.add_row(f"{key}.{sub_key}", str(sub_value))
        else:
            table.add_row(key, str(value))

    console.print(table)


@app.command("reset")
def config_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Reset configuration to defaults."""
    if not force:
        confirm = typer.confirm("Are you sure you want to reset all configuration?")
        if not confirm:
            console.print("[yellow]Reset cancelled[/yellow]")
            return

    try:
        config = get_config()
        config_file = config.config_dir / "config.json"
        if config_file.exists():
            config_file.unlink()
        console.print("[green]Configuration reset to defaults[/green]")
    except Exception as e:
        console.print(f"[red]Failed to reset configuration: {e}[/red]")
        raise typer.Exit(1)
