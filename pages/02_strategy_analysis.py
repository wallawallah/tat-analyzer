"""
Strategy Analysis page for TAT-Analyzer dashboard.

This page provides detailed analysis of trading performance by strategy,
including comparisons, metrics, and visualizations.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))


from utils.calculations import (
    calculate_pnl_metrics,
    calculate_strategy_performance,
    calculate_win_rate,
)
from utils.charts import (
    create_pnl_chart,
    create_strategy_performance_chart,
    create_trade_timing_chart,
)
from utils.constants import CHART_COLORS, STREAMLIT_STYLE
from utils.filters import FilterManager

# Apply custom CSS
st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

st.title("📈 Strategy Analysis")


def main():
    """Main strategy analysis page function."""
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load your trading data first using the main page.")
        return

    df = st.session_state.trade_data

    # Initialize filter manager
    filter_manager = FilterManager()
    filter_manager.set_sidebar_title("Filters")

    # Add filters
    filter_manager.add_date_range("strategy_date_range", "Date Range", "Date", True)
    filter_manager.add_category("strategy_template", "Template", "Template",
                               sorted(df['Template'].unique().tolist()))

    # Render filters and apply to data
    filter_manager.render_sidebar()
    filtered_df = filter_manager.apply_filters(df)

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # Display strategy overview
    display_strategy_overview(filtered_df)

    # Strategy comparison
    display_strategy_comparison(filtered_df)

    # Individual strategy analysis
    display_individual_strategy_analysis(filtered_df)




def display_strategy_overview(df: pd.DataFrame):
    """Display overview of all strategies."""
    st.markdown("### Strategy Overview")

    # Calculate strategy performance
    strategy_stats = calculate_strategy_performance(df)

    if strategy_stats.empty:
        st.warning("No strategy data available.")
        return

    # Display strategy performance table
    st.dataframe(
        strategy_stats.style.format({
            'Total_PnL': '${:,.2f}',
            'Avg_PnL': '${:,.2f}',
            'Win_Rate': '{:.1f}%',
            'ROI_Percent': '{:.1f}%',
            'PCR_Percent': '{:.1f}%',
            'Profit_Per_Trade': '${:,.2f}',
            'Total_Premium': '${:,.2f}',
            'Total_Commission': '${:,.2f}',
            'Avg_Duration_Min': '{:.0f}m'
        }),
        use_container_width=True
    )

    # Strategy performance chart
    try:
        fig = create_strategy_performance_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating strategy chart: {str(e)}")


def display_strategy_comparison(df: pd.DataFrame):
    """Display strategy comparison charts."""
    st.markdown("### Strategy Comparison")

    # Metrics selector
    metric_options = {
        'Total P&L': 'Total_PnL',
        'Win Rate': 'Win_Rate',
        'PCR': 'PCR_Percent',
        'ROI': 'ROI_Percent',
        'Average P&L': 'Avg_PnL',
        'Trade Count': 'Trade_Count'
    }

    selected_metric = st.selectbox(
        "Select metric to compare:",
        options=list(metric_options.keys()),
        index=0
    )

    # Create comparison chart
    strategy_stats = calculate_strategy_performance(df)

    if not strategy_stats.empty:
        metric_col = metric_options[selected_metric]

        fig = go.Figure()

        # Color bars based on positive/negative for P&L metrics
        if 'PnL' in metric_col or 'ROI' in metric_col or 'PCR' in metric_col:
            colors = [CHART_COLORS['positive'] if x >= 0 else CHART_COLORS['negative']
                     for x in strategy_stats[metric_col]]
        else:
            colors = CHART_COLORS['primary']

        fig.add_trace(go.Bar(
            x=strategy_stats['Strategy'],
            y=strategy_stats[metric_col],
            name=selected_metric,
            marker_color=colors,
            text=strategy_stats[metric_col].round(2),
            textposition='auto'
        ))

        # Format y-axis based on metric type
        if selected_metric in ['Total P&L', 'Average P&L']:
            fig.update_yaxes(title_text="Amount ($)")
        elif selected_metric in ['Win Rate', 'ROI', 'PCR']:
            fig.update_yaxes(title_text="Percentage (%)")
        else:
            fig.update_yaxes(title_text="Count")

        fig.update_layout(
            title=f"Strategy Comparison: {selected_metric}",
            xaxis_title="Strategy",
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # Risk-Return scatter plot
    st.markdown("#### Risk-Return Analysis")

    if not strategy_stats.empty and len(strategy_stats) > 1:
        # Calculate volatility for each strategy
        strategy_volatility = df.groupby('Strategy')['NetPnL'].std().reset_index()
        strategy_volatility.columns = ['Strategy', 'Volatility']

        # Merge with strategy stats
        risk_return_data = strategy_stats.merge(strategy_volatility, on='Strategy')

        fig = px.scatter(
            risk_return_data,
            x='Volatility',
            y='Avg_PnL',
            size='Trade_Count',
            color='Win_Rate',
            hover_name='Strategy',
            title='Risk-Return Analysis by Strategy',
            labels={
                'Volatility': 'Risk (Volatility)',
                'Avg_PnL': 'Return (Avg P&L)',
                'Win_Rate': 'Win Rate (%)'
            },
            color_continuous_scale='RdYlGn'
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need multiple strategies for risk-return analysis.")


def display_individual_strategy_analysis(df: pd.DataFrame):
    """Display detailed analysis for individual strategies."""
    st.markdown("### Individual Strategy Analysis")

    # Strategy selector
    strategies = sorted(df['Strategy'].unique())
    selected_strategy = st.selectbox(
        "Select strategy for detailed analysis:",
        options=strategies,
        index=0
    )

    # Filter data for selected strategy
    strategy_df = df[df['Strategy'] == selected_strategy]

    if strategy_df.empty:
        st.warning(f"No data available for strategy: {selected_strategy}")
        return

    # Strategy metrics
    col1, col2, col3, col4 = st.columns(4)

    pnl_metrics = calculate_pnl_metrics(strategy_df)
    win_metrics = calculate_win_rate(strategy_df)

    with col1:
        st.metric("Total P&L", f"${pnl_metrics.get('total_pnl', 0):,.2f}")
        st.metric("Average P&L", f"${pnl_metrics.get('average_pnl', 0):.2f}")

    with col2:
        st.metric("Win Rate", f"{win_metrics.get('win_rate', 0):.1f}%")
        st.metric("Total Trades", f"{len(strategy_df):,}")

    with col3:
        st.metric("Best Trade", f"${pnl_metrics.get('best_trade', 0):.2f}")
        st.metric("Worst Trade", f"${pnl_metrics.get('worst_trade', 0):.2f}")

    with col4:
        st.metric("PCR", f"{pnl_metrics.get('pcr_percent', 0):.1f}%", help="Premium Capture Rate (credit trades only)")
        st.metric("Profit Factor", f"{pnl_metrics.get('profit_factor', 0):.2f}")

    # Strategy performance over time
    st.markdown("#### Performance Over Time")

    chart_type = st.selectbox(
        "Chart type:",
        options=['cumulative', 'daily', 'monthly'],
        key='strategy_chart_type'
    )

    try:
        fig = create_pnl_chart(strategy_df, chart_type)
        fig.update_layout(title=f"{selected_strategy} - {chart_type.title()} P&L")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

    # Trade distribution analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Trade Type Distribution")
        trade_type_counts = strategy_df['TradeType'].value_counts()
        fig = px.pie(
            values=trade_type_counts.values,
            names=trade_type_counts.index,
            title=f"{selected_strategy} - Trade Types"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### P&L Distribution")
        fig = px.histogram(
            strategy_df,
            x='NetPnL',
            nbins=20,
            title=f"{selected_strategy} - P&L Distribution"
        )
        fig.update_layout(xaxis_title="Net P&L ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Time-based analysis
    st.markdown("#### Time-based Analysis")

    # Time granularity selector
    time_granularity = st.selectbox(
        "Time granularity:",
        options=["Hour", "30min", "15min"],
        index=0,
        key="strategy_analysis_time_granularity"
    )

    # Create timing performance chart for selected strategy
    try:
        fig = create_trade_timing_chart(strategy_df, granularity=time_granularity)
        fig.update_layout(title=f"{selected_strategy} - Performance by Time")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating timing chart: {str(e)}")

    # Recent trades table
    st.markdown("#### Recent Trades")

    recent_trades = strategy_df.nlargest(10, 'OpenDateTime')[
        ['OpenDateTime', 'TradeType', 'NetPnL', 'IsWinner', 'TradeDurationMinutes']
    ].copy()

    recent_trades['OpenDateTime'] = recent_trades['OpenDateTime'].dt.strftime('%Y-%m-%d %H:%M')
    recent_trades['NetPnL'] = recent_trades['NetPnL'].apply(lambda x: f"${x:.2f}")
    recent_trades['IsWinner'] = recent_trades['IsWinner'].map({True: '✅', False: '❌'})
    recent_trades['TradeDurationMinutes'] = recent_trades['TradeDurationMinutes'].apply(lambda x: f"{x:.0f}m")

    recent_trades.columns = ['Open Time', 'Trade Type', 'Net P&L', 'Winner', 'Duration']

    st.dataframe(recent_trades, use_container_width=True)


if __name__ == "__main__":
    main()
