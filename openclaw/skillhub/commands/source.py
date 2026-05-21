"""Source management commands."""

import asyncio
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from skillhub.config import get_config
from skillhub.models.source import Source, SourceType
from skillhub.services.credential_manager import CredentialManagerImpl
from skillhub.services.source_manager import SourceManagerImpl

app = typer.Typer()
console = Console()


@app.command("list")
def list_sources(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List configured sources."""

    async def _list():
        config = get_config()
        manager = SourceManagerImpl(config)
        sources = await manager.list_sources()

        if not sources:
            console.print("[yellow]No sources configured[/yellow]")
            return

        table = Table(title="Configured Sources")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="blue")
        table.add_column("URL", style="white")
        table.add_column("Enabled", style="magenta")

        for source in sources:
            table.add_row(
                source.id,
                source.name,
                source.type.value,
                source.url,
                "Yes" if source.enabled else "No",
            )

        console.print(table)

    asyncio.run(_list())


@app.command("add")
def add_source(
    name: str,
    url: str,
    source_type: str = typer.Option("github", "--type", "-t", help="Source type"),
    priority: int = typer.Option(0, "--priority", "-p", help="Source priority"),
    subpath: Optional[str] = typer.Option(None, "--subpath", help="Subdirectory containing skills"),
    token: Optional[str] = typer.Option(None, "--token", help="API token for this platform"),
) -> None:
    """Add a new source."""

    async def _add():
        config = get_config()
        manager = SourceManagerImpl(config)

        source = Source(
            name=name,
            type=SourceType(source_type),
            url=url,
            priority=priority,
            subpath=subpath,
        )

        if token:
            cred = CredentialManagerImpl(config)
            await cred.store_token(source_type, token)

        await manager.add_source(source)
        console.print(f"[green]Added source: {name} ({source.id})[/green]")

    asyncio.run(_add())


@app.command("remove")
def remove_source(
    source_id: str,
    force: bool = typer.Option(False, "--force", "-f", help="Force removal"),
) -> None:
    """Remove a source."""

    async def _remove():
        config = get_config()
        manager = SourceManagerImpl(config)

        source = await manager.get_source(source_id)
        if not source:
            console.print(f"[red]Source not found: {source_id}[/red]")
            raise typer.Exit(1)

        if not force:
            confirm = typer.confirm(f"Remove source {source.name}?")
            if not confirm:
                return

        await manager.remove_source(source_id)
        console.print(f"[green]Removed source: {source.name}[/green]")

    asyncio.run(_remove())


@app.command("test")
def test_source(
    source_id: str,
) -> None:
    """Test source connectivity."""

    async def _test():
        config = get_config()
        manager = SourceManagerImpl(config)

        console.print(f"[blue]Testing source: {source_id}[/blue]")
        result = await manager.test_source(source_id)

        if result.success:
            console.print(f"[green]✓ {result.message}[/green]")
            if result.rate_limit:
                console.print(f"  Rate limit: {result.rate_limit.remaining}/{result.rate_limit.limit}")
        else:
            console.print(f"[red]✗ {result.message}[/red]")

    asyncio.run(_test())
