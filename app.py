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


@st.dialog("Disclaimer")
def show_disclaimer():
    """Display disclaimer modal for production usage."""
    try:
        with open("disclaimer", "r") as f:
            disclaimer_content = f.read()
    except FileNotFoundError:
        disclaimer_content = """
        <div class="modal-body">
        <h5>Educational Purposes Only</h5>
        <p>This application is for <strong>informational and educational purposes only</strong>. 
        By using this application, you acknowledge and agree that you are solely responsible for your trading decisions.</p>
        </div>
        """
    
    st.markdown(disclaimer_content, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Decline", use_container_width=True):
            st.error("You must accept the terms to use this application.")
            st.stop()
    
    with col2:
        if st.button("Accept", use_container_width=True, type="primary"):
            st.session_state.disclaimer_accepted = True
            st.rerun()


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
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state.disclaimer_accepted = False

    # Check if running in production mode and show disclaimer if needed
    is_development_mode = st.get_option("global.developmentMode")
    if not is_development_mode and not st.session_state.disclaimer_accepted:
        show_disclaimer()
        return

    # Sidebar for data loading and navigation
    with st.sidebar:
        st.header("Data Loading")

        # Data loading
        load_uploaded_data()

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
    1. **Upload your CSV file** using the sidebar to get started

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
