"""Reusable UI components for the terminal GUI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from typing import List, Optional, Dict, Any


console = Console()


def create_header(title: str, subtitle: str = "") -> Panel:
    """Create a styled header panel."""
    content = Text(title, style="bold cyan", justify="center")
    if subtitle:
        content.append(f"\n{subtitle}", style="dim")
    return Panel(content, style="cyan", box="DOUBLE")


def create_menu(title: str, options: List[str]) -> str:
    """Display a menu and get user choice."""
    console.print(Panel(title, style="bold yellow"))
    
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan", width=3)
    table.add_column("Description")
    
    for i, option in enumerate(options, 1):
        table.add_row(f"[{i}]", option)
    
    console.print(table)
    
    while True:
        choice = Prompt.ask("\nSelect option", choices=[str(i) for i in range(1, len(options) + 1)])
        return choice


def show_error(message: str):
    """Display an error message."""
    console.print(f"[bold red]✗ Error:[/bold red] {message}")


def show_success(message: str):
    """Display a success message."""
    console.print(f"[bold green]✓ Success:[/bold green] {message}")


def show_info(message: str):
    """Display an info message."""
    console.print(f"[bold blue]ℹ Info:[/bold blue] {message}")


def create_progress_spinner(description: str):
    """Create a progress spinner for long operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )


def create_data_table(title: str, data: Dict[str, Any]) -> Table:
    """Create a formatted table from dictionary data."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in data.items():
        # Format numbers nicely
        if isinstance(value, float):
            value_str = f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}"
        else:
            value_str = str(value)
        table.add_row(key, value_str)
    
    return table


def confirm_action(message: str) -> bool:
    """Ask for user confirmation."""
    return Confirm.ask(message)


def get_text_input(prompt: str, default: Optional[str] = None) -> str:
    """Get text input from user."""
    return Prompt.ask(prompt, default=default)


def clear_screen():
    """Clear the terminal screen."""
    console.clear() 