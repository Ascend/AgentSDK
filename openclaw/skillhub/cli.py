"""SkillHub CLI - Main entry point."""

import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from typing import Optional
import typer
from rich.console import Console

from skillhub.commands import auth, cache, config, doctor, info, install, skill, source, uninstall, upgrade
from skillhub.commands.list import app as list_app
from skillhub.commands.search import search_skills

app = typer.Typer(
    name="skillhub",
    help="Decentralized skill management CLI",
    no_args_is_help=True,
)

console = Console()

# Register subcommands
app.add_typer(source.app, name="source", help="Manage skill sources")
app.command("search", help="Search for skills")(search_skills)

# Primary: 'skill' group for all skill operations
app.add_typer(skill.app, name="skill", help="Manage skills (list, install, uninstall, upgrade, info)")

# Backward-compatible aliases
app.add_typer(list_app, name="list", help="List installed skills (alias for 'skill list')")
app.add_typer(install.app, name="install", help="Install skills (alias for 'skill install')")
app.add_typer(uninstall.app, name="uninstall", help="Uninstall skills (alias for 'skill uninstall')")
app.add_typer(upgrade.app, name="upgrade", help="Upgrade skills (alias for 'skill upgrade')")
app.add_typer(info.app, name="info", help="Show skill information (alias for 'skill info')")
app.add_typer(config.app, name="config", help="Manage configuration")
app.add_typer(auth.app, name="auth", help="Manage authentication")
app.add_typer(cache.app, name="cache", help="Manage cache")
app.add_typer(doctor.app, name="doctor", help="Diagnose issues")


@app.callback(invoke_without_command=True)
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """SkillHub CLI - Manage AI skills from Git repositories."""


def run() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    run()
