"""List command - backward compatible alias for 'skill list'."""

import asyncio
import typer
import json
from rich.console import Console
from rich.table import Table

from skillhub.config import get_config
from skillhub.services.install_engine import InstallEngineImpl

app = typer.Typer()
console = Console()


@app.command("installed")
def list_installed(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List installed skills (alias for 'skill list')."""

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
