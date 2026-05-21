"""Doctor command."""

import asyncio
import os
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from skillhub.config import get_config
from skillhub.services.cache_manager import CacheManagerImpl
from skillhub.services.credential_manager import CredentialManagerImpl
from skillhub.services.install_engine import InstallEngineImpl

app = typer.Typer()
console = Console()


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix issues"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Diagnose and fix issues."""

    async def _doctor():
        config = get_config()
        credential_manager = CredentialManagerImpl(config)
        cache_manager = CacheManagerImpl(config)
        install_engine = InstallEngineImpl(config)

        console.print(Panel.fit("SkillHub Doctor", title="🔧", border_style="blue"))

        table = Table(box=box.ROUNDED)
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        config_dir_exists = os.path.exists(config.config_dir)
        data_dir_exists = os.path.exists(config.data_dir)
        cache_dir_exists = os.path.exists(config.cache_dir)
        skills_dir_exists = os.path.exists(config.skills_dir)

        table.add_row(
            "Config Directory",
            "✓" if config_dir_exists else "✗",
            str(config.config_dir),
        )
        table.add_row(
            "Data Directory",
            "✓" if data_dir_exists else "✗",
            str(config.data_dir),
        )
        table.add_row(
            "Cache Directory",
            "✓" if cache_dir_exists else "✗",
            str(config.cache_dir),
        )
        table.add_row(
            "Skills Directory",
            "✓" if skills_dir_exists else "✗",
            str(config.skills_dir),
        )

        tokens = await credential_manager.list_tokens()
        platforms = ["github", "gitee", "gitcode"]
        token_map = {t.platform: t for t in tokens}

        for platform in platforms:
            token_info = token_map.get(platform)
            if token_info and token_info.has_token:
                status = "✓"
                details = "Authenticated"
            else:
                status = "✗"
                details = "Not authenticated"
            table.add_row(f"{platform.title()} Auth", status, details)

        stats = await cache_manager.get_stats()
        table.add_row(
            "Cache Size",
            "✓" if stats.total_size < config.cache.max_size_mb * 1024 * 1024 else "⚠",
            f"{stats.total_size / 1024 / 1024:.2f} MB",
        )

        installed = await install_engine.list_installed()
        table.add_row("Installed Skills", "✓", f"{len(installed)} skills")

        console.print(table)

        if fix:
            console.print("\n[green]Attempting to fix issues...[/green]")

            if not config_dir_exists:
                os.makedirs(config.config_dir, exist_ok=True)
                console.print("  [green]✓ Created config directory[/green]")

            if not data_dir_exists:
                os.makedirs(config.data_dir, exist_ok=True)
                console.print("  [green]✓ Created data directory[/green]")

            if not cache_dir_exists:
                os.makedirs(config.cache_dir, exist_ok=True)
                console.print("  [green]✓ Created cache directory[/green]")

            if not skills_dir_exists:
                os.makedirs(config.skills_dir, exist_ok=True)
                console.print("  [green]✓ Created skills directory[/green]")

            removed = await install_engine.clean()
            if removed > 0:
                console.print(f"  [green]✓ Cleaned {removed} orphaned skills[/green]")

            expired = await cache_manager.clean_expired()
            if expired > 0:
                console.print(f"  [green]✓ Removed {expired} expired cache entries[/green]")

    asyncio.run(_doctor())
