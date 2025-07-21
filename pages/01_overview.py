"""
Overview page for TAT-Analyzer dashboard.

This page displays key performance metrics, P&L charts, and summary statistics
for the loaded trading data.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.calculations import (
    calculate_pnl_metrics,
    calculate_win_rate,
)
from utils.charts import create_pnl_chart, create_strategy_performance_chart
from utils.constants import STREAMLIT_STYLE
from utils.filters import FilterManager

# Apply custom CSS
st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

st.title("📊 Performance Overview")


def main():
    """Main overview page function."""
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load your trading data first using the main page.")
        return

    df = st.session_state.trade_data

    # Initialize filter manager
    filter_manager = FilterManager()
    filter_manager.set_sidebar_title("Filters")

    # Add filters
    filter_manager.add_date_range("date_range", "Date Range", "Date", True)
    filter_manager.add_category("strategy", "Strategy", "Strategy",
                               sorted(df['Strategy'].unique().tolist()))
    filter_manager.add_category("trade_type", "Trade Type", "TradeType",
                               sorted(df['TradeType'].unique().tolist()))
    filter_manager.add_category("template", "Template", "Template",
                               sorted(df['Template'].unique().tolist()))

    # Render filters and apply to data
    filter_manager.render_sidebar()
    filtered_df = filter_manager.apply_filters(df)

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # Display metrics and charts
    display_key_metrics(filtered_df)
    display_performance_summary(filtered_df)
    display_pnl_charts(filtered_df)
    display_monthly_heatmap(filtered_df)




def display_key_metrics(df: pd.DataFrame):
    """Display key performance metrics."""
    st.markdown("### Key Metrics")

    # Calculate metrics
    pnl_metrics = calculate_pnl_metrics(df)
    win_metrics = calculate_win_rate(df)

    # Row 1: Premium sold, Total P&L, Average P&L, PCR
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Premium Sold",
            f"${pnl_metrics.get('premium_sold', 0):,.2f}",
            help="Total premium received from credit trades"
        )

    with col2:
        total_pnl = pnl_metrics.get('total_pnl', 0)
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric(
            "Total P&L",
            f"${total_pnl:,.2f}",
            delta=None,
            delta_color=pnl_color
        )

    with col3:
        st.metric(
            "Average P&L",
            f"${pnl_metrics.get('average_pnl', 0):.2f}",
            help="Average profit/loss per trade"
        )

    with col4:
        st.metric(
            "PCR",
            f"{pnl_metrics.get('pcr_percent', 0):.1f}%",
            help="Premium Capture Rate (credit trades only)"
        )

    # Row 2: Total trades, Expired trades, Stopped trades, Manual closed trades
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Trades",
            f"{win_metrics.get('total_trades', 0):,}",
            help="Total number of trades"
        )

    with col2:
        st.metric(
            "Expired Trades",
            f"{win_metrics.get('expired_trades', 0):,}",
            help="Number of trades that expired"
        )

    with col3:
        st.metric(
            "Stopped Trades",
            f"{win_metrics.get('stopped_trades', 0):,}",
            help="Number of trades stopped by stop-loss"
        )

    with col4:
        st.metric(
            "Manual Closed Trades",
            f"{win_metrics.get('manual_closed_trades', 0):,}",
            help="Number of manually closed trades"
        )

    # Row 3: Winning trades, Losing trades, Profit factor, Avg slippage
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Winning Trades",
            f"{win_metrics.get('winning_trades', 0):,}",
            help="Number of profitable trades"
        )

    with col2:
        st.metric(
            "Losing Trades",
            f"{win_metrics.get('losing_trades', 0):,}",
            help="Number of losing trades"
        )

    with col3:
        st.metric(
            "Profit Factor",
            f"{pnl_metrics.get('profit_factor', 0):.2f}",
            help="Gross profits / Gross losses"
        )

    with col4:
        st.metric(
            "Average Slippage",
            f"${pnl_metrics.get('avg_slippage', 0):.2f}",
            help="Average slippage per trade"
        )

    # Additional metrics in expandable section
    with st.expander("Additional Metrics"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Trade", f"${pnl_metrics.get('best_trade', 0):.2f}", help="Most profitable single trade")
            st.metric("Worst Trade", f"${pnl_metrics.get('worst_trade', 0):.2f}", help="Most losing single trade")

        with col2:
            st.metric("Total Commissions", f"${pnl_metrics.get('total_commissions', 0):,.2f}", help="Total commission costs")
            st.metric("Total Slippage", f"${pnl_metrics.get('total_slippage', 0):.2f}")

        with col3:
            st.metric("Average Winner", f"${win_metrics.get('avg_winner', 0):.2f}")
            st.metric("Average Loser", f"${win_metrics.get('avg_loser', 0):.2f}")


def display_pnl_charts(df: pd.DataFrame):
    """Display P&L charts."""
    st.markdown("### P&L Analysis")

    # Chart type selector
    chart_type = st.selectbox(
        "Chart Type",
        options=['cumulative', 'daily', 'monthly'],
        format_func=lambda x: x.title(),
        index=0
    )

    # Create and display chart
    try:
        fig = create_pnl_chart(df, chart_type)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")


def display_performance_summary(df: pd.DataFrame):
    """Display performance summary and strategy comparison."""
    st.markdown("### Performance Summary")

    col1, col2 = st.columns(2)

    with col1:
        # Trade type distribution pie chart
        trade_type_counts = df['TradeType'].value_counts()

        import plotly.express as px
        fig_pie = px.pie(
            values=trade_type_counts.values,
            names=trade_type_counts.index,
            title="Trade Type Distribution"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Strategy performance chart
        try:
            strategy_fig = create_strategy_performance_chart(df)
            st.plotly_chart(strategy_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating strategy chart: {str(e)}")



def display_monthly_heatmap(df: pd.DataFrame):
    """Display monthly performance heatmap."""
    # Monthly performance heatmap
    with st.expander("Monthly Performance Heatmap"):
        try:
            monthly_data = create_monthly_heatmap_data(df)
            if not monthly_data.empty:
                # Create a Plotly heatmap instead of using pandas styling
                import plotly.graph_objects as go

                # Prepare data for heatmap
                years = monthly_data.index.tolist()
                months = monthly_data.columns.tolist()
                values = monthly_data.values

                # Create heatmap
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=values,
                    x=months,
                    y=years,
                    colorscale='RdYlGn',
                    zmid=0,  # Center colorscale at zero
                    text=[[f"${val:,.0f}" if not pd.isna(val) else "" for val in row] for row in values],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar={"title": "P&L ($)"}
                ))

                fig_heatmap.update_layout(
                    title="Monthly P&L Performance",
                    xaxis_title="Month",
                    yaxis_title="Year",
                    height=400
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Insufficient data for monthly heatmap.")
        except Exception as e:
            st.error(f"Error creating monthly heatmap: {str(e)}")


def create_monthly_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create data for monthly performance heatmap."""
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    monthly_pnl = df.groupby(['Year', 'Month'])['NetPnL'].sum().unstack(fill_value=0)

    # Replace month numbers with names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    monthly_pnl.columns = [month_names.get(col, col) for col in monthly_pnl.columns]

    return monthly_pnl


if __name__ == "__main__":
    main()
