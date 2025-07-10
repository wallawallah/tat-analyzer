"""
Risk Analysis page for TAT-Analyzer dashboard.

This page provides comprehensive risk analysis including drawdown analysis,
risk metrics, and various risk-related visualizations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.calculations import (
    calculate_drawdown,
    calculate_pnl_metrics,
    calculate_risk_metrics,
)
from utils.charts import create_drawdown_chart
from utils.constants import CHART_COLORS, RISK_METRICS_DESCRIPTIONS, STREAMLIT_STYLE

# Apply custom CSS
st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

st.title("⚠️ Risk Analysis")


def main():
    """Main risk analysis page function."""
    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load your trading data first using the main page.")
        return

    df = st.session_state.trade_data

    # Display risk metrics overview
    display_risk_metrics(df)

    # Display drawdown analysis
    display_drawdown_analysis(df)

    # Display risk distribution analysis
    display_risk_distribution(df)

    # Display risk by strategy
    display_risk_by_strategy(df)


def display_risk_metrics(df: pd.DataFrame):
    """Display key risk metrics."""
    st.markdown("### Risk Metrics Overview")

    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(df)
    pnl_metrics = calculate_pnl_metrics(df)
    drawdown_metrics = calculate_drawdown(df)

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Max Drawdown",
            f"${drawdown_metrics.get('max_drawdown', 0):.2f}",
            help=RISK_METRICS_DESCRIPTIONS.get('max_drawdown', '')
        )

        st.metric(
            "Max Drawdown %",
            f"{drawdown_metrics.get('max_drawdown_pct', 0):.1f}%",
            help=RISK_METRICS_DESCRIPTIONS.get('max_drawdown_pct', '')
        )

    with col2:
        st.metric(
            "Volatility",
            f"${risk_metrics.get('volatility', 0):.2f}",
            help=RISK_METRICS_DESCRIPTIONS.get('volatility', '')
        )

        st.metric(
            "VaR (95%)",
            f"${risk_metrics.get('var_95', 0):.2f}",
            help="Value at Risk at 95% confidence level"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{risk_metrics.get('sharpe_ratio', 0):.3f}",
            help=RISK_METRICS_DESCRIPTIONS.get('sharpe_ratio', '')
        )

        st.metric(
            "Sortino Ratio",
            f"{risk_metrics.get('sortino_ratio', 0):.3f}",
            help=RISK_METRICS_DESCRIPTIONS.get('sortino_ratio', '')
        )

    with col4:
        st.metric(
            "Profit Factor",
            f"{pnl_metrics.get('profit_factor', 0):.2f}",
            help=RISK_METRICS_DESCRIPTIONS.get('profit_factor', '')
        )

        st.metric(
            "Calmar Ratio",
            f"{risk_metrics.get('calmar_ratio', 0):.3f}",
            help=RISK_METRICS_DESCRIPTIONS.get('calmar_ratio', '')
        )

    # Additional risk metrics in expandable section
    with st.expander("Additional Risk Metrics"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Skewness", f"{risk_metrics.get('skewness', 0):.3f}")
            st.metric("Kurtosis", f"{risk_metrics.get('kurtosis', 0):.3f}")

        with col2:
            st.metric("VaR (99%)", f"${risk_metrics.get('var_99', 0):.2f}")
            st.metric("Current Drawdown", f"${drawdown_metrics.get('current_drawdown', 0):.2f}")

        with col3:
            st.metric("Downside Deviation", f"${risk_metrics.get('downside_deviation', 0):.2f}")
            st.metric("Longest Drawdown", f"{drawdown_metrics.get('longest_drawdown_days', 0)} trades")


def display_drawdown_analysis(df: pd.DataFrame):
    """Display drawdown analysis charts."""
    st.markdown("### Drawdown Analysis")

    # Create drawdown chart
    try:
        fig = create_drawdown_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating drawdown chart: {str(e)}")

    # Underwater equity curve with cumulative P&L
    st.markdown("#### Equity Curve with Drawdown")

    df_sorted = df.sort_values('OpenDateTime')
    cumulative_pnl = df_sorted['NetPnL'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Cumulative P&L', 'Drawdown'),
        row_heights=[0.7, 0.3]
    )

    # Cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=df_sorted['OpenDateTime'],
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color=CHART_COLORS['primary'], width=2)
        ),
        row=1, col=1
    )

    # Running maximum
    fig.add_trace(
        go.Scatter(
            x=df_sorted['OpenDateTime'],
            y=running_max,
            mode='lines',
            name='Running Maximum',
            line=dict(color=CHART_COLORS['secondary'], width=1, dash='dash')
        ),
        row=1, col=1
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=df_sorted['OpenDateTime'],
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color=CHART_COLORS['negative'], width=2),
            fill='tonexty',
            fillcolor=CHART_COLORS['negative_alpha']
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="Equity Curve and Drawdown Analysis",
        height=600,
        showlegend=True
    )

    fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def display_risk_distribution(df: pd.DataFrame):
    """Display risk distribution analysis."""
    st.markdown("### Risk Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### P&L Distribution")

        # Create histogram with normal distribution overlay
        fig = go.Figure()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=df['NetPnL'],
                nbinsx=30,
                name='P&L Distribution',
                opacity=0.7,
                marker_color=CHART_COLORS['primary']
            )
        )

        # Add vertical lines for key statistics
        mean_pnl = df['NetPnL'].mean()
        std_pnl = df['NetPnL'].std()

        fig.add_vline(
            x=mean_pnl,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_pnl:.2f}"
        )

        fig.add_vline(
            x=mean_pnl - std_pnl,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"-1σ: ${mean_pnl - std_pnl:.2f}"
        )

        fig.add_vline(
            x=mean_pnl + std_pnl,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"+1σ: ${mean_pnl + std_pnl:.2f}"
        )

        fig.update_layout(
            title="P&L Distribution with Statistics",
            xaxis_title="Net P&L ($)",
            yaxis_title="Frequency",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Risk-Return Scatter")

        # Calculate rolling statistics
        window_size = min(10, len(df) // 4)  # Adaptive window size

        df_sorted = df.sort_values('OpenDateTime')
        rolling_return = df_sorted['NetPnL'].rolling(window=window_size).mean()
        rolling_risk = df_sorted['NetPnL'].rolling(window=window_size).std()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=rolling_risk,
                y=rolling_return,
                mode='markers',
                name='Risk-Return Points',
                marker=dict(
                    color=range(len(rolling_risk)),
                    colorscale='Viridis',
                    size=8,
                    showscale=True,
                    colorbar=dict(title="Time")
                ),
                text=df_sorted['OpenDateTime'].dt.strftime('%Y-%m-%d'),
                hovertemplate='Risk: %{x:.2f}<br>Return: %{y:.2f}<br>Date: %{text}<extra></extra>'
            )
        )

        fig.update_layout(
            title=f"Rolling Risk-Return Analysis ({window_size}-trade window)",
            xaxis_title="Risk (Standard Deviation)",
            yaxis_title="Return (Average P&L)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Risk metrics table
    st.markdown("#### Risk Metrics Summary")

    # Calculate percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pnl_percentiles = np.percentile(df['NetPnL'], percentiles)

    risk_summary = pd.DataFrame({
        'Percentile': [f"{p}%" for p in percentiles],
        'P&L Value': [f"${v:.2f}" for v in pnl_percentiles]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(risk_summary, use_container_width=True, hide_index=True)

    with col2:
        # Additional risk statistics
        risk_stats = pd.DataFrame({
            'Metric': ['Best Trade', 'Worst Trade', 'Average Trade', 'Median Trade', 'Standard Deviation'],
            'Value': [
                f"${df['NetPnL'].max():.2f}",
                f"${df['NetPnL'].min():.2f}",
                f"${df['NetPnL'].mean():.2f}",
                f"${df['NetPnL'].median():.2f}",
                f"${df['NetPnL'].std():.2f}"
            ]
        })

        st.dataframe(risk_stats, use_container_width=True, hide_index=True)


def display_risk_by_strategy(df: pd.DataFrame):
    """Display risk analysis by strategy."""
    st.markdown("### Risk Analysis by Strategy")

    # Calculate risk metrics by strategy
    strategy_risk = []

    for strategy in df['Strategy'].unique():
        strategy_df = df[df['Strategy'] == strategy]

        if len(strategy_df) < 2:  # Need at least 2 trades for meaningful risk metrics
            continue

        risk_metrics = calculate_risk_metrics(strategy_df)
        drawdown_metrics = calculate_drawdown(strategy_df)
        pnl_metrics = calculate_pnl_metrics(strategy_df)

        strategy_risk.append({
            'Strategy': strategy,
            'Total_Trades': len(strategy_df),
            'Total_PnL': strategy_df['NetPnL'].sum(),
            'Avg_PnL': strategy_df['NetPnL'].mean(),
            'Volatility': risk_metrics.get('volatility', 0),
            'Max_Drawdown': drawdown_metrics.get('max_drawdown', 0),
            'Sharpe_Ratio': risk_metrics.get('sharpe_ratio', 0),
            'Sortino_Ratio': risk_metrics.get('sortino_ratio', 0),
            'VaR_95': risk_metrics.get('var_95', 0),
            'Profit_Factor': pnl_metrics.get('profit_factor', 0)
        })

    if not strategy_risk:
        st.warning("Insufficient data for strategy risk analysis.")
        return

    strategy_risk_df = pd.DataFrame(strategy_risk)

    # Strategy risk comparison table
    st.markdown("#### Strategy Risk Metrics")

    display_risk_df = strategy_risk_df.copy()
    display_risk_df['Total_PnL'] = display_risk_df['Total_PnL'].apply(lambda x: f"${x:,.2f}")
    display_risk_df['Avg_PnL'] = display_risk_df['Avg_PnL'].apply(lambda x: f"${x:.2f}")
    display_risk_df['Volatility'] = display_risk_df['Volatility'].apply(lambda x: f"${x:.2f}")
    display_risk_df['Max_Drawdown'] = display_risk_df['Max_Drawdown'].apply(lambda x: f"${x:.2f}")
    display_risk_df['VaR_95'] = display_risk_df['VaR_95'].apply(lambda x: f"${x:.2f}")
    display_risk_df['Sharpe_Ratio'] = display_risk_df['Sharpe_Ratio'].apply(lambda x: f"{x:.3f}")
    display_risk_df['Sortino_Ratio'] = display_risk_df['Sortino_Ratio'].apply(lambda x: f"{x:.3f}")
    display_risk_df['Profit_Factor'] = display_risk_df['Profit_Factor'].apply(lambda x: f"{x:.2f}")

    display_risk_df.columns = [
        'Strategy', 'Trades', 'Total P&L', 'Avg P&L', 'Volatility',
        'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'VaR 95%', 'Profit Factor'
    ]

    st.dataframe(display_risk_df, use_container_width=True, hide_index=True)

    # Risk-return scatter plot by strategy
    st.markdown("#### Strategy Risk-Return Profile")

    if len(strategy_risk_df) > 1:
        fig = px.scatter(
            strategy_risk_df,
            x='Volatility',
            y='Avg_PnL',
            size='Total_Trades',
            color='Sharpe_Ratio',
            hover_name='Strategy',
            title='Strategy Risk-Return Profile',
            labels={
                'Volatility': 'Risk (Volatility)',
                'Avg_PnL': 'Return (Average P&L)',
                'Sharpe_Ratio': 'Sharpe Ratio'
            },
            color_continuous_scale='RdYlGn'
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need multiple strategies for risk-return comparison.")

    # Strategy drawdown comparison
    st.markdown("#### Drawdown Comparison by Strategy")

    drawdown_data = []
    for strategy in df['Strategy'].unique():
        strategy_df = df[df['Strategy'] == strategy].sort_values('OpenDateTime')

        if len(strategy_df) < 2:
            continue

        cumulative_pnl = strategy_df['NetPnL'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max

        drawdown_data.append({
            'Strategy': strategy,
            'Max_Drawdown': drawdown.min(),
            'Current_Drawdown': drawdown.iloc[-1],
            'Drawdown_Duration': len(drawdown[drawdown < 0]) if (drawdown < 0).any() else 0
        })

    if drawdown_data:
        drawdown_df = pd.DataFrame(drawdown_data)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=drawdown_df['Strategy'],
                y=drawdown_df['Max_Drawdown'],
                name='Max Drawdown',
                marker_color=CHART_COLORS['negative']
            )
        )

        fig.update_layout(
            title="Maximum Drawdown by Strategy",
            xaxis_title="Strategy",
            yaxis_title="Drawdown ($)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
