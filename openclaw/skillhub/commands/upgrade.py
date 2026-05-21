"""Upgrade command."""

import asyncio
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from skillhub.config import get_config
from skillhub.services.install_engine import InstallEngineImpl

app = typer.Typer()
console = Console()


@app.command()
def upgrade_skill(
    skill: Optional[str] = typer.Option(None, help="Skill name (omit to upgrade all)"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Target version"),
    force: bool = typer.Option(False, "--force", "-f", help="Force upgrade"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate upgrade"),
) -> None:
    """Upgrade installed skills."""

    async def _upgrade():
        config = get_config()
        engine = InstallEngineImpl(config)

        if skill:
            skills_to_upgrade = [skill]
        else:
            installed = await engine.list_installed()
            skills_to_upgrade = [s.name for s in installed]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            for skill_name in skills_to_upgrade:
                task = progress.add_task(f"Upgrading {skill_name}...", total=None)

                result = await engine.upgrade(skill_name, version)

                if result.success:
                    progress.update(task, description=f"[green]Upgraded {skill_name}[/green]")
                else:
                    progress.update(task, description=f"[red]Failed to upgrade {skill_name}[/red]")
                    for error in result.errors:
                        console.print(f"  [red]{error}[/red]")

        if skill:
            console.print(f"[green]✓ Upgraded {skill}[/green]")
        else:
            console.print("[green]✓ All skills upgraded[/green]")

    asyncio.run(_upgrade())
