"""Authentication commands."""

import asyncio
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from skillhub.config import get_config
from skillhub.services.credential_manager import CredentialManagerImpl

app = typer.Typer()
console = Console()


@app.command("login")
def login(
    platform: str,
    token: Optional[str] = typer.Option(None, "--token", "-t", help="API token"),
) -> None:
    """Authenticate with a platform."""

    async def _login():
        config = get_config()
        manager = CredentialManagerImpl(config)

        if not token:
            console.print("[yellow]No token provided, using interactive login[/yellow]")
            token_input = typer.prompt("Enter your API token", hide_input=True)
        else:
            token_input = token

        console.print(f"[blue]Validating token for {platform}...[/blue]")
        validation = await manager.validate_token(platform, token_input)

        if validation.valid:
            await manager.store_token(platform, token_input)
            console.print(f"[green]✓ Authenticated with {platform}[/green]")
            if validation.rate_limit:
                console.print(f"  Rate limit: {validation.rate_limit.remaining}/{validation.rate_limit.limit}")
        else:
            console.print(f"[red]✗ Authentication failed: {validation.message}[/red]")
            raise typer.Exit(1)

    asyncio.run(_login())


@app.command("logout")
def logout(
    platform: str,
) -> None:
    """Remove authentication for a platform."""

    async def _logout():
        config = get_config()
        manager = CredentialManagerImpl(config)

        token = await manager.get_token(platform)
        if not token:
            console.print(f"[yellow]No authentication found for {platform}[/yellow]")
            return

        await manager.remove_token(platform)
        console.print(f"[green]✓ Removed authentication for {platform}[/green]")

    asyncio.run(_logout())


@app.command("status")
def status() -> None:
    """Show authentication status."""

    async def _status():
        config = get_config()
        manager = CredentialManagerImpl(config)

        tokens = await manager.list_tokens()

        table = Table(title="Authentication Status")
        table.add_column("Platform", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Type", style="yellow")

        platforms = ["github", "gitee", "gitcode"]
        token_map = {t.platform: t for t in tokens}

        for platform in platforms:
            token_info = token_map.get(platform)
            if token_info and token_info.has_token:
                auth_status = "✓ authenticated"
                token_type = token_info.type
            else:
                auth_status = "✗ not authenticated"
                token_type = ""  # nosec B105

            table.add_row(platform, auth_status, token_type)

        console.print(table)

    asyncio.run(_status())
