# Terminal GUI Task List

## Overview
Build a terminal-based GUI using Rich (https://github.com/Textualize/rich) to interact with the CVaR/CLEIR solver system.

## Phase 1: Setup & Dependencies

### 1.1 Update Requirements
- [ ] Add `rich>=13.0.0` to `pyproject.toml`
- [ ] Add `click>=8.0.0` for CLI framework
- [ ] Update environment.yml if needed

### 1.2 Create GUI Structure
- [ ] Create `src/gui/` directory
- [ ] Create `src/gui/__init__.py`
- [ ] Create `src/gui/app.py` - main GUI application
- [ ] Create `src/gui/components.py` - reusable UI components

## Phase 2: Core Components

### 2.1 Main Menu
- [ ] Create main menu with options:
  - Data Management
  - Run CVaR Optimization
  - Run CLEIR Optimization
  - View Results
  - Settings
  - Exit

### 2.2 Data Management Screen
- [ ] Download new data (ticker selection)
- [ ] View cached data
- [ ] Clear cache
- [ ] Show download progress with Rich progress bars

### 2.3 Optimization Screen
- [ ] Parameter input form (confidence level, lookback, etc.)
- [ ] Universe selection (predefined lists or custom)
- [ ] Run optimization with live progress
- [ ] Display results summary

### 2.4 Results Viewer
- [ ] Show performance metrics in Rich tables
- [ ] Display portfolio weights
- [ ] Show rebalancing history
- [ ] Export options

### 2.5 Deliverables Screen
- [ ] Generate all Task A deliverables with one click:
  - Daily index values CSV
  - Performance metrics table (annual return, volatility, Sharpe, CVaR, max DD, turnover)
  - Comparison plot (CVaR vs equal-weight vs cap-weight)
- [ ] Show deliverable status checklist
- [ ] Export to specified directory
- [ ] Add GitHub repo link: https://github.com/Campervans/Alpha-Pods---Simple

## Phase 3: Implementation Details

### 3.1 Entry Point
- [ ] Create `gui.py` in project root
- [ ] Add shebang and make executable
- [ ] Handle dependency checking on startup
- [ ] Graceful error handling

### 3.2 UI Components
- [ ] Progress spinner for long operations
- [ ] Input validation with error messages
- [ ] Confirmation dialogs for destructive actions
- [ ] Status bar with current operation

### 3.3 Data Flow
- [ ] Create `src/gui/controllers.py` for business logic
- [ ] Keep UI separate from solver logic
- [ ] Add result caching for quick viewing

## Phase 4: Polish

### 4.1 User Experience
- [ ] Add keyboard shortcuts
- [ ] Help text for each screen
- [ ] Color coding for different states
- [ ] Clear navigation breadcrumbs
- [ ] About screen with:
  - Project description
  - GitHub link: https://github.com/Campervans/Alpha-Pods---Simple
  - Task requirements summary

### 4.2 Error Handling
- [ ] Catch and display solver errors nicely
- [ ] Network error handling for downloads
- [ ] Validation for all inputs
- [ ] Recovery options

## Phase 5: Testing & Documentation

### 5.1 Testing
- [ ] Manual test each screen
- [ ] Test error scenarios
- [ ] Test with different terminal sizes

### 5.2 Documentation
- [ ] Add GUI section to README
- [ ] Screenshot examples
- [ ] Quick start guide

## Implementation Order
1. Start with `gui.py` and basic menu
2. Add data management (most independent)
3. Add CVaR optimization screen
4. Add CLEIR optimization screen
5. Add results viewer
6. Polish and error handling

## Notes
- Keep it simple - terminal UI, not web
- Use Rich's Layout for responsive design
- Show live logs during operations
- Make common tasks easy (e.g., "Run standard backtest")

## Task A Requirements Reference
- **Universe**: 60 liquid U.S. stocks from S&P 100
- **Optimization**: 95% daily CVaR
- **Constraints**: Fully invested, long-only, max 5% per stock
- **Rebalancing**: Quarterly
- **Transaction costs**: 10 bps per side
- **Data period**: Jan 1, 2010 to Dec 31, 2024
- **Deliverables**:
  1. CSV with daily index values
  2. Performance metrics table
  3. Comparison plot vs benchmarks