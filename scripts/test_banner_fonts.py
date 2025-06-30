#!/usr/bin/env python3
"""Test different pyfiglet fonts for the eToro banner."""

from pyfiglet import Figlet
from rich.console import Console
from rich.style import Style
from rich.panel import Panel

console = Console()
etoro_green = Style(color="#6ebe44")  # eToro brand green

# Test different fonts
fonts_to_test = ['slant', 'standard', 'big', 'block', 'banner', 'banner3', 'colossal', 'doom', 'epic', 'isometric1', 'larry3d', 'lean', 'small', 'smslant', 'speed', 'starwars']

# Test with full text
full_text = "eToro - AlphaPod Challenge"
console.print("\n[bold cyan]Testing different fonts with full text:[/bold cyan]")
console.print(f"[dim]Text: '{full_text}'[/dim]\n")

for font in fonts_to_test:
    try:
        figlet = Figlet(font=font)
        ascii_banner = figlet.renderText(full_text)
        
        console.print(Panel(f"Font: {font}", style="yellow"))
        console.print(ascii_banner, style=etoro_green)
        console.print("-" * 80)
    except:
        console.print(f"[red]Font '{font}' not available[/red]")

# Test with shorter text options
console.print("\n[bold cyan]Testing with shorter text options:[/bold cyan]")

shorter_options = [
    "eToro AlphaPod",
    "eToro - AlphaPod",
    "eToro",
    "AlphaPod Challenge"
]

best_fonts = ['slant', 'standard', 'small', 'smslant', 'lean']

for text in shorter_options:
    console.print(f"\n[bold magenta]Text: '{text}'[/bold magenta]")
    for font in best_fonts:
        try:
            figlet = Figlet(font=font)
            ascii_banner = figlet.renderText(text)
            
            console.print(f"\n[yellow]Font: {font}[/yellow]")
            console.print(ascii_banner, style=etoro_green)
        except:
            pass 