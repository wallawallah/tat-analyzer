"""
Constants and configuration for TAT-Analyzer.

This module contains constants, mappings, and configuration values used throughout the application.
"""

# Chart colors and styling
CHART_COLORS = {
    'primary': '#2E86AB',
    'primary_alpha': 'rgba(46, 134, 171, 0.3)',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'positive': '#28A745',
    'positive_alpha': 'rgba(40, 167, 69, 0.3)',
    'negative': '#DC3545',
    'negative_alpha': 'rgba(220, 53, 69, 0.3)',
    'neutral': '#6C757D',
    'background': '#F8F9FA'
}

# Chart theme configuration
CHART_THEME = {
    'template': 'plotly_white',
    'height': 500,
    'showlegend': True,
    'legend': {
        'orientation': "h",
        'yanchor': "bottom",
        'y': 1.02,
        'xanchor': "right",
        'x': 1
    },
    'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
    'font': {'size': 12},
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white'
}

# Trade types mapping
TRADE_TYPES = {
    'CallSpread': 'Call Spread',
    'PutSpread': 'Put Spread',
    'IronCondor': 'Iron Condor',
    'IronButterfly': 'Iron Butterfly',
    'Straddle': 'Straddle',
    'Strangle': 'Strangle'
}

# Stop types mapping
STOP_TYPES = {
    'Vertical': 'Vertical',
    'Short': 'Short',
    'Long': 'Long',
    'Manual': 'Manual'
}

# Status mapping
STATUS_TYPES = {
    'Stopped': 'Stopped',
    'Expired': 'Expired',
    'Manual Closed': 'Manual Closed',
    'Assigned': 'Assigned'
}

# Strategy mapping (based on common patterns in the data)
STRATEGIES = {
    'MEIC - Morning': 'MEIC Morning',
    'MEIC - afternoon': 'MEIC Afternoon',
    'MEIC My way (morning)': 'MEIC Custom Morning',
    'MEIC My way (afternoon)': 'MEIC Custom Afternoon',
    'METF Call side': 'METF Call',
    'METF Put side': 'METF Put',
    'METF': 'METF'
}

# Streamlit page configuration
PAGE_CONFIG = {
    'page_title': 'TAT-Analyzer',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Metric formatting
METRIC_FORMATS = {
    'currency': '${:,.2f}',
    'percentage': '{:.2f}%',
    'integer': '{:,}',
    'float': '{:.2f}',
    'ratio': '{:.3f}'
}

# Default date ranges for filtering
DATE_RANGES = {
    'Today': 0,
    'Yesterday': 1,
    'This Week': 'this_week',
    'Last Week': 'last_week',
    'This Month': 'this_month',
    'Last Month': 'last_month',
    'Last 7 days': 7,
    'Last 30 days': 30,
    'Last 3 months': 90,
    'Last 6 months': 180,
    'Last year': 365,
    'All time': None,
    'Custom': 'custom'
}

# Column mappings for display
COLUMN_DISPLAY_NAMES = {
    'TradeID': 'Trade ID',
    'OpenDateTime': 'Open Date/Time',
    'CloseDateTime': 'Close Date/Time',
    'TradeType': 'Trade Type',
    'StopType': 'Stop Type',
    'StopMultiple': 'Stop Multiple',
    'PriceOpen': 'Open Price',
    'PriceClose': 'Close Price',
    'PriceStopTarget': 'Stop Target',
    'TotalPremium': 'Total Premium',
    'ProfitLoss': 'Gross P&L',
    'NetPnL': 'Net P&L',
    'Commission': 'Commission',
    'IsWinner': 'Winner',
    'ROI': 'ROI %',
    'PCR': 'PCR %',
    'TradeDurationMinutes': 'Duration (min)',
    'BuyingPower': 'Buying Power',
    'StopMultipleResult': 'Stop Multiple Result',
    'Slippage': 'Slippage'
}

# Risk metrics descriptions
RISK_METRICS_DESCRIPTIONS = {
    'max_drawdown': 'Maximum peak-to-trough decline in account value',
    'max_drawdown_pct': 'Maximum drawdown as percentage of peak value',
    'volatility': 'Standard deviation of trade returns',
    'sharpe_ratio': 'Risk-adjusted return measure',
    'sortino_ratio': 'Downside risk-adjusted return measure',
    'calmar_ratio': 'Annual return divided by maximum drawdown',
    'profit_factor': 'Gross profits divided by gross losses',
    'expectancy': 'Expected profit per trade'
}

# Chart type options
CHART_TYPES = {
    'pnl_charts': ['cumulative', 'daily', 'monthly'],
    'distribution_charts': ['histogram', 'box', 'violin'],
    'time_periods': ['daily', 'weekly', 'monthly']
}

# Default number of bins for histograms
DEFAULT_HISTOGRAM_BINS = 50

# Color palette for multi-series charts
MULTI_SERIES_COLORS = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
    '#6A994E', '#BC6C25', '#8E44AD', '#16A085',
    '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C'
]

# Default figure size
DEFAULT_FIGURE_SIZE = (12, 8)

# Streamlit styling
STREAMLIT_STYLE = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    .positive {
        color: #28A745;
    }

    .negative {
        color: #DC3545;
    }

    .neutral {
        color: #6C757D;
    }

    .sidebar-info {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
"""

# File validation settings
ALLOWED_FILE_EXTENSIONS = ['.csv']
MAX_FILE_SIZE_MB = 100

# Data validation settings
REQUIRED_COLUMNS = [
    'Account', 'Date', 'TradeType', 'ProfitLoss', 'Status', 'Strategy', 'TradeID'
]

# Cache settings
CACHE_TTL = 3600  # 1 hour in seconds
