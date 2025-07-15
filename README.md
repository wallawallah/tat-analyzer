# TAT-Analyzer

A comprehensive trading performance dashboard for analyzing Trade Automation Toolbox (TAT) CSV exports.

## Features

- **📊 Performance Overview**: Key metrics, P&L charts, and summary statistics
- **📈 Strategy Analysis**: Compare performance across different strategies
- **📋 Trade Details**: Search, filter, and analyze individual trades
- **⚠️ Risk Analysis**: Drawdown analysis and comprehensive risk metrics

## Quick Start

### Installation

1. Clone or download this repository
2. Install dependencies using uv:

```bash
uv sync
```

### Running the Dashboard

```bash
uv run streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage

### Loading Data

1. **Upload CSV**: Use the sidebar to upload your TAT CSV export file

### Navigation

- **Overview**: Get a high-level view of your trading performance
- **Strategy Analysis**: Dive deep into strategy-specific performance
- **Trade Details**: Explore individual trades with filtering and search
- **Risk Analysis**: Understand your risk profile and drawdown patterns

### Features

#### Performance Metrics
- Total P&L and ROI
- Win rate and trade statistics
- Best/worst trades
- Profit factor and Sharpe ratio

#### Charts and Visualizations
- Cumulative P&L curves
- Strategy performance comparisons
- Risk-return scatter plots
- Drawdown analysis
- Trade duration vs P&L

#### Filtering and Search
- Date range filtering
- Strategy and trade type filters
- P&L range filtering
- Text search across all fields

#### Risk Analysis
- Maximum drawdown calculation
- Value at Risk (VaR) metrics
- Volatility and risk-adjusted returns
- Strategy risk comparison

## Data Format

The dashboard expects CSV files exported from Trade Automation Toolbox with the following key columns:

- `Account`: Trading account identifier
- `Date`: Trade date
- `TradeType`: Type of trade (e.g., CallSpread, PutSpread)
- `Strategy`: Trading strategy name
- `ProfitLoss`: Gross profit/loss
- `Commission`: Commission fees
- `Status`: Trade status (Stopped, Expired, etc.)
- `OpenDateTime`/`CloseDateTime`: Trade timing
- Additional columns for prices, premiums, etc.

## Project Structure

```
tat-analyzer/
├── app.py                    # Main Streamlit application
├── pages/                    # Multi-page app pages
│   ├── 01_overview.py
│   ├── 02_strategy_analysis.py
│   ├── 03_trade_details.py
│   └── 04_risk_analysis.py
├── utils/                    # Utility modules
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py        # CSV loading and preprocessing
│   ├── calculations.py       # Trade analysis functions
│   ├── charts.py            # Plotly chart functions
│   ├── constants.py         # Constants and configuration
│   └── filters.py           # Data filtering utilities
├── pyproject.toml           # Dependencies and project config
├── uv.lock                  # Dependency lock file
├── CLAUDE.md                # AI development guidelines
└── README.md                # This file
```

## Dependencies

- **streamlit**: Web framework for the dashboard
- **plotly**: Interactive charts and visualizations
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **python-dateutil**: Date parsing and manipulation
- **loguru**: Logging

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd tat-analyzer

# Install dependencies
uv sync

# Install development dependencies
uv sync --extra dev
```

### Code Style

This project uses `ruff` for code formatting and linting:

```bash
# Format code
uv run ruff format .

# Check for linting issues
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .
```

### Adding New Features

1. **Calculations**: Add new analysis functions to `utils/calculations.py`
2. **Charts**: Add new chart types to `utils/charts.py`
3. **Pages**: Create new pages in the `pages/` directory
4. **Configuration**: Update constants in `utils/constants.py`

## Technical Details

### Data Processing

- CSV files are loaded and cached using Streamlit's caching mechanism
- Date columns are parsed and standardized
- Derived metrics (ROI, trade duration, etc.) are calculated
- Data validation ensures required columns are present

### Performance Optimization

- Streamlit caching for data loading and calculations
- Efficient pandas operations for large datasets
- Lazy loading of charts and analyses
- Configurable pagination for large trade tables

### Error Handling

- Comprehensive error handling for file loading
- Graceful degradation for missing or invalid data
- User-friendly error messages and warnings
- Logging for debugging and monitoring

## Troubleshooting

### Common Issues

1. **"No data available"**: Ensure your CSV file has the required columns
2. **Import errors**: Make sure all dependencies are installed with `uv sync`
3. **Performance issues**: Try filtering data by date range for large datasets
4. **Chart not displaying**: Check browser console for JavaScript errors

### File Format Issues

If your CSV file doesn't load correctly:

1. Check that it has the required columns
2. Ensure dates are in a recognizable format
3. Verify numeric columns don't contain text
4. Try opening the file in a text editor to check for formatting issues

## Contributing

1. Follow the existing code style and structure
2. Add docstrings to new functions
3. Update tests if adding new calculations
4. Document any new features in this README

## License

This project is open source. See the license file for details.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the code documentation
3. Create an issue in the project repository