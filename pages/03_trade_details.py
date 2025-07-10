"""
Trade Details page for TAT-Analyzer dashboard.

This page provides detailed view of individual trades with search, filtering,
and sorting capabilities.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.charts import create_trade_duration_chart, create_trade_timing_chart
from utils.constants import COLUMN_DISPLAY_NAMES, STREAMLIT_STYLE
from utils.filters import FilterManager

# Apply custom CSS
st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

st.title("📋 Trade Details")


def main():
    """Main trade details page function."""
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load your trading data first using the main page.")
        return

    df = st.session_state.trade_data

    # Initialize filter manager
    filter_manager = FilterManager()
    filter_manager.set_sidebar_title("Filters & Search")

    # Add filters
    filter_manager.add_date_range("trade_date_range", "Date Range", "Date", True)
    filter_manager.add_category("strategy", "Strategy", "Strategy",
                               sorted(df['Strategy'].unique().tolist()), multi_select=True, default_all=True)
    filter_manager.add_category("trade_type", "Trade Type", "TradeType",
                               sorted(df['TradeType'].unique().tolist()), multi_select=True, default_all=True)
    filter_manager.add_category("status", "Status", "Status",
                               sorted(df['Status'].unique().tolist()), multi_select=True, default_all=True)
    filter_manager.add_category("template", "Template", "Template",
                               sorted(df['Template'].unique().tolist()))

    # Add custom filters
    st.sidebar.subheader("Outcome")
    outcome_filter = st.sidebar.selectbox(
        "Filter by outcome:",
        options=['All', 'Winners Only', 'Losers Only'],
        index=0,
        key="outcome_filter"
    )

    # Add P&L range filter
    min_pnl = float(df['NetPnL'].min())
    max_pnl = float(df['NetPnL'].max())
    filter_manager.add_numeric_range("pnl_range", "P&L Range ($)", "NetPnL", min_pnl, max_pnl, 1.0, "${:,.2f}")

    # Add search filter
    search_columns = ['TradeID', 'Strategy', 'TradeType', 'Status', 'StopType']
    filter_manager.add_text_search("search", "Search", search_columns, "Enter trade ID, strategy, or any text...")

    # Render filters and apply to data
    filter_manager.render_sidebar()
    filtered_df = filter_manager.apply_filters(df)

    # Apply outcome filter manually (since it's a custom filter)
    if outcome_filter == 'Winners Only':
        filtered_df = filtered_df[filtered_df['IsWinner']]
    elif outcome_filter == 'Losers Only':
        filtered_df = filtered_df[~filtered_df['IsWinner']]

    if filtered_df.empty:
        st.warning("No trades match the selected filters.")
        return

    # Display trade summary
    display_trade_summary(filtered_df)

    # Display trade analysis
    display_trade_analysis(filtered_df)

    # Display trade table
    display_trade_table(filtered_df)




def display_trade_summary(df: pd.DataFrame):
    """Display summary statistics for filtered trades."""
    st.markdown("### Trade Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", f"{len(df):,}")
        st.metric("Winners", f"{df['IsWinner'].sum():,}")

    with col2:
        win_rate = (df['IsWinner'].sum() / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.metric("Losers", f"{(~df['IsWinner']).sum():,}")

    with col3:
        total_pnl = df['NetPnL'].sum()
        st.metric("Total P&L", f"${total_pnl:,.2f}")
        st.metric("Average P&L", f"${df['NetPnL'].mean():.2f}")

    with col4:
        best_trade = df['NetPnL'].max()
        worst_trade = df['NetPnL'].min()
        st.metric("Best Trade", f"${best_trade:.2f}")
        st.metric("Worst Trade", f"${worst_trade:.2f}")


def display_trade_table(df: pd.DataFrame):
    """Display the main trade table with sorting and pagination."""
    st.markdown("### Trade Table")

    # Column selector
    available_columns = [
        'TradeID', 'OpenDateTime', 'CloseDateTime', 'TradeType', 'Strategy',
        'NetPnL', 'IsWinner', 'Status', 'TradeDurationMinutes', 'PCR', 'ROI',
        'PriceOpen', 'PriceClose', 'TotalPremium', 'Commission'
    ]

    # Filter available columns based on what exists in the DataFrame
    available_columns = [col for col in available_columns if col in df.columns]

    default_columns = [
        'TradeID', 'OpenDateTime', 'TradeType', 'Strategy', 'NetPnL', 'IsWinner'
    ]
    default_columns = [col for col in default_columns if col in available_columns]

    selected_columns = st.multiselect(
        "Select columns to display:",
        options=available_columns,
        default=default_columns
    )

    if not selected_columns:
        st.warning("Please select at least one column to display.")
        return

    # Sort options
    col1, col2 = st.columns(2)

    with col1:
        sort_column = st.selectbox(
            "Sort by:",
            options=selected_columns,
            index=0 if 'OpenDateTime' not in selected_columns else selected_columns.index('OpenDateTime')
        )

    with col2:
        sort_order = st.selectbox(
            "Sort order:",
            options=['Descending', 'Ascending'],
            index=0
        )

    # Prepare display DataFrame
    display_df = df[selected_columns].copy()

    # Format columns for display
    if 'OpenDateTime' in display_df.columns:
        display_df['OpenDateTime'] = display_df['OpenDateTime'].dt.strftime('%Y-%m-%d %H:%M')

    if 'CloseDateTime' in display_df.columns:
        display_df['CloseDateTime'] = display_df['CloseDateTime'].dt.strftime('%Y-%m-%d %H:%M')

    if 'IsWinner' in display_df.columns:
        display_df['IsWinner'] = display_df['IsWinner'].map({True: '✅', False: '❌'})

    if 'TradeDurationMinutes' in display_df.columns:
        display_df['TradeDurationMinutes'] = display_df['TradeDurationMinutes'].apply(lambda x: f"{x:.0f}m")

    # Format numeric columns
    numeric_cols = ['NetPnL', 'PriceOpen', 'PriceClose', 'TotalPremium', 'Commission']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")

    if 'ROI' in display_df.columns:
        display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.1f}%")

    if 'PCR' in display_df.columns:
        display_df['PCR'] = display_df['PCR'].apply(lambda x: f"{x:.1f}%")

    # Rename columns for better display
    display_df.columns = [COLUMN_DISPLAY_NAMES.get(col, col) for col in display_df.columns]

    # Sort the DataFrame
    ascending = sort_order == 'Ascending'

    # For sorting, we need to use the original column name
    sorted_indices = df[sort_column].sort_values(ascending=ascending).index
    display_df = display_df.loc[sorted_indices]

    # Pagination
    page_size = st.selectbox("Rows per page:", options=[10, 25, 50, 100], index=1)

    total_rows = len(display_df)
    total_pages = (total_rows - 1) // page_size + 1

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        page = st.number_input(
            f"Page (1-{total_pages}):",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )

    # Calculate start and end indices
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Display the table
    st.dataframe(
        display_df.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True
    )

    # Display pagination info
    st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_rows)} of {total_rows} trades")


def display_trade_analysis(df: pd.DataFrame):
    """Display additional trade analysis charts."""
    st.markdown("### Trade Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### P&L Distribution")

        fig = px.histogram(
            df,
            x='NetPnL',
            nbins=30,
            title='P&L Distribution',
            color_discrete_sequence=['#2E86AB']
        )

        fig.update_layout(
            xaxis_title="Net P&L ($)",
            yaxis_title="Count",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Trade Duration vs P&L")

        try:
            fig = create_trade_duration_chart(df)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating duration chart: {str(e)}")

    # Additional analysis
    st.markdown("#### Performance by Time")

    # Time granularity selector
    time_granularity = st.selectbox(
        "Time granularity:",
        options=["Hour", "30min", "15min"],
        index=0,
        key="trade_details_time_granularity"
    )

    # Create timing performance chart
    try:
        fig = create_trade_timing_chart(df, granularity=time_granularity)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating timing chart: {str(e)}")

    # Trade statistics by strategy
    st.markdown("#### Performance by Strategy")

    strategy_stats = df.groupby('Strategy').agg({
        'NetPnL': ['sum', 'mean', 'count'],
        'IsWinner': 'sum',
        'TradeDurationMinutes': 'mean'
    }).round(2)

    strategy_stats.columns = ['Total_PnL', 'Avg_PnL', 'Count', 'Winners', 'Avg_Duration']
    strategy_stats['Win_Rate'] = (strategy_stats['Winners'] / strategy_stats['Count']) * 100
    strategy_stats = strategy_stats.reset_index()

    # Format for display
    strategy_display = strategy_stats.copy()
    strategy_display['Total_PnL'] = strategy_display['Total_PnL'].apply(lambda x: f"${x:,.2f}")
    strategy_display['Avg_PnL'] = strategy_display['Avg_PnL'].apply(lambda x: f"${x:.2f}")
    strategy_display['Win_Rate'] = strategy_display['Win_Rate'].apply(lambda x: f"{x:.1f}%")
    strategy_display['Avg_Duration'] = strategy_display['Avg_Duration'].apply(lambda x: f"{x:.0f}m")

    strategy_display.columns = [
        'Strategy', 'Total P&L', 'Avg P&L', 'Trade Count',
        'Winners', 'Avg Duration', 'Win Rate'
    ]

    st.dataframe(strategy_display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
