"""Skill command - primary interface for all skill operations."""

import asyncio
import json
import os
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from skillhub.config import get_config
from skillhub.interfaces.install_engine import InstallOptions
from skillhub.services.install_engine import InstallEngineImpl
from skillhub.services.skill_resolver import SkillResolverImpl
from skillhub.services.source_manager import SourceManagerImpl

app = typer.Typer(help="Manage skills: list, install, uninstall, upgrade, info")
console = Console()


def _is_local_path(skill: str) -> bool:
    return os.path.exists(skill) and os.path.isdir(skill)


@app.command("list", help="List installed skills")
def skill_list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all installed skills."""

    async def _list():
        config = get_config()
        engine = InstallEngineImpl(config)
        skills = await engine.list_installed()

        if json_output:
            output = [s.model_dump() for s in skills]
            console.print(json.dumps(output, indent=2, default=str))
        else:
            if not skills:
                console.print("[yellow]No skills installed[/yellow]")
                return

            table = Table(title="Installed Skills")
            table.add_column("Name", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Source", style="yellow")
            table.add_column("Installed", style="dim")

            for skill in skills:
                table.add_row(
                    skill.name,
                    skill.version,
                    skill.source_type,
                    skill.installed_at.strftime("%Y-%m-%d %H:%M"),
                )

            console.print(table)

    asyncio.run(_list())


@app.command("install", help="Install a skill from a source or local path")
def skill_install(
    skill: str,
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source to install from"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate installation"),
    no_deps: bool = typer.Option(False, "--no-deps", help="Skip dependencies"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Custom install path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Install a skill."""

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


@app.command("uninstall", help="Uninstall a skill")
def skill_uninstall(
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


@app.command("upgrade", help="Upgrade installed skills")
def skill_upgrade(
    skill: Optional[str] = typer.Option(None, "--skill", "-s", help="Skill name (omit to upgrade all)"),
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


@app.command("info", help="Show detailed information about a skill")
def skill_info(
    skill: str,
    version: str = typer.Option("latest", "--version", "-v", help="Skill version"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show detailed information about a skill."""

    async def _info():
        config = get_config()
        install_engine = InstallEngineImpl(config)
        source_manager = SourceManagerImpl(config)

        installed = await install_engine.get_installed(skill)

        sources = await source_manager.list_sources()
        resolver = SkillResolverImpl(sources, config=source_manager.config)

        resolved = await resolver.resolve_version(skill, version)

        if json_output:
            output = {
                "installed": installed.model_dump() if installed else None,
                "available": resolved.model_dump() if resolved else None,
            }
            console.print(json.dumps(output, indent=2, default=str))
        else:
            console.print(Panel.fit(f"Skill Information: {skill}"))

            table = Table()
            table.add_column("Property", style="cyan")
            table.add_column("Value")

            if installed:
                table.add_row("Name", installed.name)
                table.add_row("Installed Version", installed.version)
                table.add_row("Source", installed.source_type)
                table.add_row("Install Path", installed.install_path)
                table.add_row("Installed At", installed.installed_at.strftime("%Y-%m-%d %H:%M"))
            else:
                table.add_row("Status", "[yellow]Not installed[/yellow]")

            if resolved:
                table.add_row("Latest Version", resolved.version)
                if resolved.manifest.description:
                    table.add_row("Description", resolved.manifest.description)
                if resolved.manifest.author:
                    table.add_row("Author", resolved.manifest.author)
                if resolved.manifest.tags:
                    table.add_row("Tags", ", ".join(resolved.manifest.tags))
                if resolved.manifest.license:
                    table.add_row("License", resolved.manifest.license)

            console.print(table)

    asyncio.run(_info())
