"""Install command."""

import asyncio
import json
import os
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from skillhub.config import get_config
from skillhub.interfaces.install_engine import InstallOptions
from skillhub.services.install_engine import InstallEngineImpl
from skillhub.services.skill_resolver import SkillResolverImpl
from skillhub.services.source_manager import SourceManagerImpl

app = typer.Typer()
console = Console()


def _is_local_path(skill: str) -> bool:
    return os.path.exists(skill) and os.path.isdir(skill)


@app.command()
def install_skill(
    skill: str,
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source to install from"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate installation"),
    no_deps: bool = typer.Option(False, "--no-deps", help="Skip dependencies"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Custom install path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Install a skill from a source or local path."""

    async def _install():
        config = get_config()
        source_manager = SourceManagerImpl(config)
        install_engine = InstallEngineImpl(config)

        if _is_local_path(skill):
            await _install_local(skill, install_engine, target, force, json_output)
        else:
            await _install_remote(
                skill, source, source_manager, install_engine, target, force, dry_run, no_deps, json_output
            )

    asyncio.run(_install())


async def _install_local(
    skill_path: str,
    install_engine: InstallEngineImpl,
    target: Optional[str],
    force: bool,
    json_output: bool,
):
    options = InstallOptions(
        force=force,
        skip_dependencies=True,
        target_path=target,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Installing from {skill_path}...", total=None)
        result = await install_engine.install_from_path(skill_path, options)
        progress.update(task, description="[green]Installed from local path[/green]")

    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        if result.success:
            console.print(f"[green]✓ Installed {result.skill.name}@{result.skill.version}[/green]")
        else:
            console.print("[red]✗ Installation failed[/red]")
            for error in result.errors:
                console.print(f"  [red]{error}[/red]")
            raise typer.Exit(1)


async def _install_remote(
    skill: str,
    source: Optional[str],
    source_manager: SourceManagerImpl,
    install_engine: InstallEngineImpl,
    target: Optional[str],
    force: bool,
    dry_run: bool,
    no_deps: bool,
    json_output: bool,
):
    sources = await source_manager.list_sources()
    resolver = SkillResolverImpl(sources, config=source_manager.config)

    if "@" in skill:
        name, version = skill.split("@", 1)
    else:
        name, version = skill, "latest"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Resolving {name}...", total=None)

        # Resolve source identifier (name or ID) to source ID
        if source:
            matched = next((s for s in sources if source in (s.id, s.name)), None)
            source_ids = [matched.id] if matched else None
        else:
            source_ids = None
        resolved = await resolver.resolve_version(name, version, source_ids)

        if not resolved:
            progress.update(task, description=f"[red]Failed to resolve {name}[/red]")
            console.print(f"[red]Could not find skill: {name}[/red]")
            raise typer.Exit(1)

        progress.update(task, description=f"Installing {name}@{resolved.version}...")

        options = InstallOptions(
            force=force,
            dry_run=dry_run,
            skip_dependencies=no_deps,
            target_path=target,
        )

        result = await install_engine.install(resolved, options)

        progress.update(task, description=f"[green]Installed {name}@{resolved.version}[/green]")

    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        if result.success:
            console.print(f"[green]✓ Installed {result.skill.name}@{result.skill.version}[/green]")
            if result.installed_dependencies:
                console.print(f"  Dependencies: {', '.join(result.installed_dependencies)}")
        else:
            console.print("[red]✗ Installation failed[/red]")
            for error in result.errors:
                console.print(f"  [red]{error}[/red]")
            raise typer.Exit(1)
