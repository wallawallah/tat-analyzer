"""
Main Streamlit application for TAT-Analyzer.

This is the entry point for the trade analysis dashboard that loads CSV data
and provides multi-page navigation for analyzing trading performance.
"""

import sys
from pathlib import Path

import streamlit as st

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from loguru import logger

from utils.constants import PAGE_CONFIG, STREAMLIT_STYLE
from utils.data_loader import get_data_summary, load_trades_data, validate_csv_format

# Configure page
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)


def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">TAT-Analyzer Dashboard</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'trade_data' not in st.session_state:
        st.session_state.trade_data = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None

    # Sidebar for data loading and navigation
    with st.sidebar:
        st.header("Data Loading")

        # Data loading options
        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV file", "Use sample data"]
        )

        if data_source == "Upload CSV file":
            load_uploaded_data()
        else:
            load_sample_data()

        # Show data summary if loaded
        if st.session_state.data_loaded:
            display_data_summary()

        # Navigation info
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown("""
        Use the pages in the sidebar to explore different aspects of your trading data:

        - **Overview**: Key metrics and performance summary
        - **Strategy Analysis**: Performance by strategy
        - **Trade Details**: Individual trade analysis
        - **Risk Analysis**: Risk metrics and drawdown
        """)

    # Main content area
    if not st.session_state.data_loaded:
        display_welcome_message()
    else:
        display_data_info()


def load_uploaded_data():
    """Handle uploaded CSV file."""
    uploaded_file = st.file_uploader(
        "Upload your TAT CSV file",
        type=['csv'],
        help="Upload a CSV file exported from Trade Automation Toolbox"
    )

    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Validate file format
            is_valid, error_message = validate_csv_format(temp_path)

            if not is_valid:
                st.error(f"Invalid file format: {error_message}")
                return

            # Load data
            with st.spinner("Loading trade data..."):
                df = load_trades_data(temp_path)
                st.session_state.trade_data = df
                st.session_state.data_summary = get_data_summary(df)
                st.session_state.data_loaded = True

            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

            st.success(f"Successfully loaded {len(df)} trades!")
            logger.info(f"Loaded {len(df)} trades from uploaded file")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Error loading uploaded data: {str(e)}")
            # Clean up temp file on error
            Path(temp_path).unlink(missing_ok=True)


def load_sample_data():
    """Load sample data from the existing CSV file."""
    sample_file_path = "export-2025-7-9.csv"

    if st.button("Load Sample Data"):
        if not Path(sample_file_path).exists():
            st.error(f"Sample file not found: {sample_file_path}")
            return

        try:
            with st.spinner("Loading sample trade data..."):
                df = load_trades_data(sample_file_path)
                st.session_state.trade_data = df
                st.session_state.data_summary = get_data_summary(df)
                st.session_state.data_loaded = True

            st.success(f"Successfully loaded {len(df)} trades from sample data!")
            logger.info(f"Loaded {len(df)} trades from sample file")

        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            logger.error(f"Error loading sample data: {str(e)}")


def display_data_summary():
    """Display data summary in sidebar."""
    if st.session_state.data_summary:
        summary = st.session_state.data_summary

        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### Data Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Trades", f"{summary['total_trades']:,}")
            st.metric("Strategies", summary['strategies'])
            st.metric("Accounts", summary['accounts'])

        with col2:
            # Color the P&L metric based on positive/negative
            "positive" if summary['total_pnl'] >= 0 else "negative"
            st.metric(
                "Total P&L",
                f"${summary['total_pnl']:,.2f}",
                delta=None
            )
            st.metric("Win Rate", f"{summary['win_rate']:.1f}%")
            st.metric("Avg Duration", f"{summary['avg_trade_duration']:.0f}m")

        st.markdown("**Date Range:**")
        st.write(f"{summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}")

        st.markdown('</div>', unsafe_allow_html=True)


def display_welcome_message():
    """Display welcome message when no data is loaded."""
    st.markdown("## Welcome to TAT-Analyzer!")

    st.markdown("""
    This dashboard helps you analyze your trading performance using data exported from Trade Automation Toolbox.

    ### Getting Started
    1. **Upload your CSV file** using the sidebar, or
    2. **Use the sample data** to explore the dashboard features

    ### Features
    - **Performance Overview**: Key metrics, P&L charts, and win rate analysis
    - **Strategy Analysis**: Compare performance across different strategies
    - **Trade Details**: Search and filter individual trades
    - **Risk Analysis**: Drawdown analysis and risk metrics

    ### About Your Data
    The dashboard expects CSV files exported from Trade Automation Toolbox with the following columns:
    - Account, Date, TradeType, ProfitLoss, Status, Strategy, and others

    All data is processed locally - nothing is sent to external servers.
    """)


def display_data_info():
    """Display information about the loaded data."""
    if st.session_state.data_summary:
        summary = st.session_state.data_summary

        st.markdown("## Data Loaded Successfully")

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Trades",
                f"{summary['total_trades']:,}",
                help="Total number of trades in the dataset"
            )

        with col2:
            pnl_delta = summary['total_pnl']
            st.metric(
                "Total P&L",
                f"${summary['total_pnl']:,.2f}",
                delta=f"${pnl_delta:,.2f}",
                help="Total profit/loss across all trades"
            )

        with col3:
            st.metric(
                "Win Rate",
                f"{summary['win_rate']:.1f}%",
                help="Percentage of profitable trades"
            )

        with col4:
            st.metric(
                "Strategies",
                summary['strategies'],
                help="Number of different strategies used"
            )

        st.markdown("---")
        st.markdown("### Navigate to explore your data:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📊 Overview**")
            st.markdown("Key performance metrics and charts")

        with col2:
            st.markdown("**📈 Strategy Analysis**")
            st.markdown("Compare strategy performance")

        with col3:
            st.markdown("**📋 Trade Details**")
            st.markdown("Search and filter individual trades")

        st.markdown("---")
        st.info("💡 **Tip:** Use the pages in the sidebar to navigate between different analysis views.")


if __name__ == "__main__":
    main()
