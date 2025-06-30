#!/usr/bin/env python3
"""
CVaR/CLEIR Portfolio Optimization GUI

Run this script to launch the terminal GUI.
It will automatically install dependencies if needed.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_and_install_dependencies():
    """Check and install required dependencies."""
    try:
        import rich
        import click
        import pyfiglet
        print("✓ Dependencies already installed")
        return True
    except ImportError:
        print("Installing required dependencies...")
        
        # Try pip install
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "rich>=13.0.0", "click>=8.0.0", "pyfiglet>=0.8.0", "-q"
            ])
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install dependencies")
            print("Please install manually: pip install rich>=13.0.0 click>=8.0.0 pyfiglet>=0.8.0")
            return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("CVaR/CLEIR Portfolio Optimization Terminal GUI")
    print("=" * 60)
    
    # Check dependencies
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # Add src to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Import and run GUI
    try:
        from src.gui.app import main as run_gui
        run_gui()
    except ImportError as e:
        print(f"Error importing GUI: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main() 