# TAT-Analyzer

A comprehensive trading performance dashboard for analyzing Trade Automation Toolbox (TAT) CSV exports.
** TAT-Analyzer is currently in preview beta mode. Development is ongoing. Please report any bugs / feature requests via github issues  **

## Features

- **📊 Performance Overview**: Key metrics, P&L charts, and summary statistics
- **📈 Strategy Analysis**: Compare performance across different strategies
- **📋 Trade Details**: Search, filter, and analyze individual trades

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


## Data Format

The dashboard expects CSV files exported from Trade Automation Toolbox.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/wallawallah/tat-analyzer.git
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

### Common Issues

1. **"No data available"**: Ensure your CSV file has the required columns
2. **Import errors**: Make sure all dependencies are installed with `uv sync`
3. **Performance issues**: Try filtering data by date range for large datasets
4. **Chart not displaying**: Check browser console for JavaScript errors


## Contributing

1. Follow the existing code style and structure
2. Add docstrings to new functions
3. Update tests if adding new calculations
4. Document any new features in this README

## Disclaimer

**IMPORTANT: This software is provided "AS IS" with no warranty and is for informational and educational purposes only.**

This analysis tool should **NOT** be construed as any form of investment, financial, or trading advice. The information provided is not intended to be used for the purpose of making or refraining from making any investment decisions.

**Trading stocks, options, and other financial instruments involves a high level of risk and may not be suitable for all investors.** There is no guarantee of future performance and you could lose some or all of your invested capital. Past performance is not necessarily indicative of future results.

**Limitation of Liability:** In no event shall the authors, contributors, or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software. This includes, but is not limited to, any direct, indirect, incidental, special, exemplary, or consequential damages.

By using this software, you acknowledge and agree that you are solely responsible for your investment decisions. We strongly recommend that you conduct your own research and seek the advice of a qualified professional before making any investment decisions.

## License

This project is licensed under the GNU Affero General Public License v3.0 with Commons Clause.

See the [LICENSE](LICENSE) file for full license details.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the code documentation
3. Create an issue in the project repository