# Performance Graph Links in GUI

## Overview
The CVaR/CLEIR GUI now includes clickable links to performance analysis graphs after optimization completes.

## Features

### 1. Automatic Graph Generation
When you run CVaR or CLEIR optimization through the GUI, the system will automatically:
- Generate a performance comparison plot showing the index vs benchmarks
- Save it as a PNG file in the results directory

### 2. Clickable Links
After optimization completes, you'll see:
- A results summary table with key metrics
- A clickable link to the performance graph (if it exists)
- Instructions on how to generate the graph if it doesn't exist

### 3. Graph Contents
The performance analysis graphs include:
- **Your optimized index** (CVaR or CLEIR)
- **Cap-weighted benchmark** (S&P 500/SPY)
- **Equal-weighted benchmark** (simulated)
- **Performance summary box** showing:
  - Total return
  - Annualized return
  - Volatility
  - Sharpe ratio

## File Locations
The graphs are saved at:
- CVaR: `/Users/james/Alpha-Pods---Simple/results/cvar_index_performance_analysis.png`
- CLEIR: `/Users/james/Alpha-Pods---Simple/results/cleir_index_performance_analysis.png`

## How to Use

1. **Run Optimization**: Use menu option 2 (CVaR) or 3 (CLEIR)
2. **View Results**: After optimization completes, you'll see the results summary
3. **Click the Link**: Click on the file path link to open the graph in your default image viewer
4. **Alternative**: If the graph doesn't exist, go to Results menu and select "Generate Missing Deliverables"

## Terminal Compatibility
The links use the `file://` protocol and are formatted as Rich terminal links. They should be clickable in most modern terminals including:
- iTerm2
- Terminal.app (macOS)
- VS Code integrated terminal
- Most Linux terminals with link support

## Example Output
```
✓ Optimization completed!

CVaR Optimization Results:
Annual Return      12.45%
Sharpe Ratio       0.823
Max Drawdown       -18.32%
Final Index Value  245.67
Total Return       145.67%

📊 Performance Analysis Graph:
file:///Users/james/Alpha-Pods---Simple/results/cvar_index_performance_analysis.png
Click the link above to view the CVaR index vs benchmark comparison graph
``` 