"""Reusable UI components for the terminal GUI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.tree import Tree
from rich import box
from typing import List, Optional, Dict, Any
import os
from pathlib import Path


console = Console()


def create_header(title: str, subtitle: str = "") -> Panel:
    """Create a styled header panel."""
    content = Text(title, style="bold cyan", justify="center")
    if subtitle:
        content.append(f"\n{subtitle}", style="dim")
    return Panel(content, style="cyan", box=box.DOUBLE)


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


def create_file_tree(root_path: str, title: str = "Files") -> Tree:
    """Create a file tree visualization."""
    tree = Tree(f"📁 {title}", style="bold cyan")
    
    def add_directory(parent_node, path: Path, max_depth: int = 3, current_depth: int = 0):
        """Recursively add directories and files to tree."""
        if current_depth >= max_depth:
            return
            
        try:
            items = sorted(path.iterdir())
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            # Add directories first
            for dir_item in dirs:
                if dir_item.name.startswith('.'):
                    continue
                dir_node = parent_node.add(f"📁 {dir_item.name}/", style="cyan")
                add_directory(dir_node, dir_item, max_depth, current_depth + 1)
            
            # Then add files
            for file_item in files:
                if file_item.name.startswith('.'):
                    continue
                # Choose icon based on file extension
                if file_item.suffix == '.csv':
                    icon = "📊"
                    style = "green"
                elif file_item.suffix == '.png':
                    icon = "🖼️"
                    style = "magenta"
                elif file_item.suffix == '.txt':
                    icon = "📄"
                    style = "white"
                else:
                    icon = "📄"
                    style = "dim"
                
                # Add file size
                size = file_item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
                    
                parent_node.add(f"{icon} {file_item.name} ({size_str})", style=style)
                
        except PermissionError:
            parent_node.add("⚠️ Permission denied", style="red")
    
    add_directory(tree, Path(root_path))
    return tree


def create_interactive_file_tree(root_path: str, title: str = "Files") -> Optional[str]:
    """Create an interactive file tree where users can select files."""
    from rich.table import Table
    
    # Collect all files
    files_list = []
    
    def collect_files(path: Path, prefix: str = ""):
        """Collect all files with their paths."""
        try:
            items = sorted(path.iterdir())
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            # Process directories first
            for dir_item in dirs:
                if dir_item.name.startswith('.'):
                    continue
                collect_files(dir_item, prefix + dir_item.name + "/")
            
            # Then process files
            for file_item in files:
                if file_item.name.startswith('.'):
                    continue
                    
                # Choose icon based on file extension
                if file_item.suffix == '.csv':
                    icon = "📊"
                elif file_item.suffix == '.png':
                    icon = "🖼️"
                elif file_item.suffix in ['.txt', '.md']:
                    icon = "📄"
                else:
                    icon = "📄"
                
                # Get file size
                size = file_item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
                
                files_list.append({
                    'path': str(file_item),
                    'display': f"{icon} {prefix}{file_item.name}",
                    'size': size_str,
                    'icon': icon
                })
                
        except PermissionError:
            pass
    
    collect_files(Path(root_path))
    
    if not files_list:
        show_info("No files found in directory")
        return None
    
    # Create a table with numbered files
    table = Table(title=f"📁 {title} - Select a file", show_header=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("File", style="white")
    table.add_column("Size", style="dim", width=10)
    
    for i, file_info in enumerate(files_list, 1):
        table.add_row(str(i), file_info['display'], file_info['size'])
    
    console.print(table)
    
    # Add option to go back
    console.print(f"\n[{len(files_list) + 1}] Back to menu", style="yellow")
    
    # Get user choice
    choices = [str(i) for i in range(1, len(files_list) + 2)]
    choice = Prompt.ask("\nSelect file to view", choices=choices)
    
    if choice == str(len(files_list) + 1):
        return None
    
    return files_list[int(choice) - 1]['path'] 