"""Cache commands."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table

from skillhub.config import get_config
from skillhub.services.cache_manager import CacheManagerImpl

app = typer.Typer()
console = Console()


@app.command("info")
def cache_info() -> None:
    """Show cache information."""

    async def _info():
        config = get_config()
        manager = CacheManagerImpl(config)

        stats = await manager.get_stats()

        table = Table(title="Cache Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Cache Size", f"{stats.total_size / 1024 / 1024:.2f} MB")
        table.add_row("Entries", str(stats.size))
        table.add_row("Hit Rate", f"{stats.hit_rate * 100:.1f}%")
        table.add_row("Location", str(config.cache_dir))

        console.print(table)

    asyncio.run(_info())


@app.command("clear")
def cache_clear(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Clear the cache."""

    async def _clear():
        config = get_config()
        manager = CacheManagerImpl(config)

        if not force:
            confirm = typer.confirm("Are you sure you want to clear the cache?")
            if not confirm:
                console.print("[yellow]Cache clear cancelled[/yellow]")
                return

        await manager.clear()
        console.print("[green]✓ Cache cleared[/green]")

    asyncio.run(_clear())


@app.command("clean")
def cache_clean() -> None:
    """Remove expired cache entries."""

    async def _clean():
        config = get_config()
        manager = CacheManagerImpl(config)

        removed = await manager.clean_expired()
        console.print(f"[green]✓ Removed {removed} expired entries[/green]")

    asyncio.run(_clean())
