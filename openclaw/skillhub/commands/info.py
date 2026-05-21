"""Info command."""

import asyncio
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from skillhub.config import get_config
from skillhub.services.install_engine import InstallEngineImpl
from skillhub.services.skill_resolver import SkillResolverImpl
from skillhub.services.source_manager import SourceManagerImpl

app = typer.Typer()
console = Console()


@app.command()
def info_skill(
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
        resolver = SkillResolverImpl(sources, config=config)

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
