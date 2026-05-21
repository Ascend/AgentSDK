"""Search command."""

import asyncio
import json
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from skillhub.config import get_config
from skillhub.services.source_manager import SourceManagerImpl

app = typer.Typer()
console = Console()


@app.command()
def search_skills(
    query: str,
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source to search"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search for skills."""

    async def _search():
        config = get_config()
        manager = SourceManagerImpl(config)

        sources = [source] if source else None
        results = await manager.search_across_sources(query, sources)

        all_skills = []
        for source_id, skills in results.items():
            for skill in skills:
                all_skills.append(skill)

        all_skills = all_skills[:limit]

        if json_output:
            output = [s.model_dump() for s in all_skills]
            console.print(json.dumps(output, indent=2, default=str))
        else:
            if not all_skills:
                console.print("[yellow]No skills found[/yellow]")
                return

            table = Table(title=f"Search Results: {query}")
            table.add_column("Name", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Description")
            table.add_column("Source", style="yellow")

            for skill in all_skills:
                table.add_row(
                    skill.name,
                    skill.version,
                    skill.description or "",
                    skill.source.get("name", ""),
                )

            console.print(table)

    asyncio.run(_search())
