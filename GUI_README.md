# CVaR/CLEIR Terminal GUI

A Rich-based terminal GUI for the CVaR/CLEIR portfolio optimization system.

## Quick Start

```bash
# Run the GUI (installs dependencies automatically)
python3 run_gui.py
```

## Features

### 📊 Data Management
- Download S&P 100 stock data
- Download custom ticker lists
- View cached data
- Clear cache

### 🎯 Optimization
- **CVaR Optimization**: Standard Conditional Value at Risk optimization
- **CLEIR Optimization**: CVaR-LASSO Enhanced Index Replication

### 📈 Results & Deliverables
- View performance metrics
- Generate Task A deliverables:
  - Daily index values CSV
  - Performance metrics table
  - Comparison plots

### ℹ️ About
- Project information
- GitHub repository link
- Task requirements reference

## Navigation

- Use number keys to select menu options
- Press Enter to confirm selections
- Follow on-screen prompts for inputs

## Requirements

The GUI will automatically install:
- `rich>=13.0.0` - Terminal UI framework
- `click>=8.0.0` - CLI utilities

## Project Structure

```
src/gui/
├── __init__.py      # Module initialization
├── app.py           # Main GUI application
├── components.py    # Reusable UI components
└── controllers.py   # Business logic controllers
```

## GitHub Repository

https://github.com/Campervans/Alpha-Pods---Simple 