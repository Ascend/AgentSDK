"""Uninstall command."""

import asyncio
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from skillhub.config import get_config
from skillhub.services.install_engine import InstallEngineImpl

app = typer.Typer()
console = Console()


@app.command()
def uninstall_skill(
    skill: str,
    force: bool = typer.Option(False, "--force", "-f", help="Force uninstallation"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Uninstall a skill."""

    async def _uninstall():
        config = get_config()
        engine = InstallEngineImpl(config)

        installed = await engine.get_installed(skill)
        if not installed:
            console.print(f"[red]Skill not installed: {skill}[/red]")
            raise typer.Exit(1)

        if not yes and not force:
            confirm = typer.confirm(f"Uninstall {skill}?")
            if not confirm:
                return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Uninstalling {skill}...", total=None)
            await engine.uninstall(skill)
            progress.update(task, description=f"[green]Uninstalled {skill}[/green]")

        console.print(f"[green]✓ Uninstalled {skill}[/green]")

    asyncio.run(_uninstall())
