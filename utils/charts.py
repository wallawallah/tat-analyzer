"""
Chart creation utilities for TAT-Analyzer.

This module contains functions for creating various Plotly charts for trade analysis.
"""


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .constants import CHART_COLORS, CHART_THEME
from .calculations import _calculate_pcr


def create_pnl_chart(df: pd.DataFrame, chart_type: str = 'cumulative') -> go.Figure:
    """
    Create P&L chart showing cumulative or daily performance.
    
    Args:
        df: DataFrame containing trade data
        chart_type: Type of chart ('cumulative', 'daily', 'monthly')
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return _create_empty_chart("No data available")

    df_sorted = df.sort_values('OpenDateTime')

    if chart_type == 'cumulative':
        return _create_cumulative_pnl_chart(df_sorted)
    elif chart_type == 'daily':
        return _create_daily_pnl_chart(df_sorted)
    elif chart_type == 'monthly':
        return _create_monthly_pnl_chart(df_sorted)
    else:
        raise ValueError("chart_type must be 'cumulative', 'daily', or 'monthly'")


def create_distribution_chart(df: pd.DataFrame, column: str, chart_type: str = 'histogram') -> go.Figure:
    """
    Create distribution chart for specified column.
    
    Args:
        df: DataFrame containing trade data
        column: Column to analyze
        chart_type: Type of chart ('histogram', 'box', 'violin')
        
    Returns:
        Plotly figure object
    """
    if df.empty or column not in df.columns:
        return _create_empty_chart("No data available")

    if chart_type == 'histogram':
        return _create_histogram(df, column)
    elif chart_type == 'box':
        return _create_box_plot(df, column)
    elif chart_type == 'violin':
        return _create_violin_plot(df, column)
    else:
        raise ValueError("chart_type must be 'histogram', 'box', or 'violin'")


def create_strategy_performance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create strategy performance comparison chart.
    
    Args:
        df: DataFrame containing trade data
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return _create_empty_chart("No data available")

    strategy_stats = df.groupby('Strategy').agg({
        'NetPnL': ['sum', 'count'],
        'IsWinner': 'sum'
    }).round(2)

    strategy_stats.columns = ['Total_PnL', 'Trade_Count', 'Winning_Trades']
    strategy_stats['Win_Rate'] = (strategy_stats['Winning_Trades'] / strategy_stats['Trade_Count']) * 100
    strategy_stats = strategy_stats.reset_index()

    # Calculate PCR for each strategy
    strategy_stats['PCR_Percent'] = 0.0
    for idx, row in strategy_stats.iterrows():
        strategy_name = row['Strategy']
        strategy_data = df[df['Strategy'] == strategy_name]
        if not strategy_data.empty:
            strategy_stats.loc[idx, 'PCR_Percent'] = _calculate_pcr(strategy_data)

    # Color bars based on positive/negative P&L
    bar_colors = [CHART_COLORS['positive'] if pnl >= 0 else CHART_COLORS['negative'] 
                  for pnl in strategy_stats['Total_PnL']]

    # Create figure with single y-axis
    fig = go.Figure()

    # Add P&L bars with PnL, WR, and PCR labels
    fig.add_trace(
        go.Bar(
            x=strategy_stats['Strategy'],
            y=strategy_stats['Total_PnL'],
            name='Total P&L',
            marker_color=bar_colors,
            text=[f"PnL: ${pnl:,.0f}<br>PCR: {pcr:.1f}%<br>WR: {wr:.1f}%" for pnl, wr, pcr in 
                  zip(strategy_stats['Total_PnL'], strategy_stats['Win_Rate'], strategy_stats['PCR_Percent'])],
            textposition='auto',
            textfont=dict(size=10, color='white'),
        )
    )

    # Update layout
    fig.update_xaxes(title_text="Strategy")
    fig.update_yaxes(title_text="Total P&L ($)")
    
    # Create layout config and override showlegend
    layout_config = CHART_THEME.copy()
    layout_config['title'] = "Strategy Performance Comparison"
    layout_config['showlegend'] = False
    
    fig.update_layout(**layout_config)

    return fig


def create_trade_timing_chart(df: pd.DataFrame, granularity: str = 'Hour') -> go.Figure:
    """
    Create chart showing trade timing patterns.
    
    Args:
        df: DataFrame containing trade data
        granularity: Time granularity - 'Hour', '30min', or '15min'
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return _create_empty_chart("No data available")

    # Create time grouping based on granularity
    if granularity == 'Hour':
        df_copy = df.copy()
        df_copy['TimeGroup'] = df_copy['OpenDateTime'].dt.hour
        df_copy['TimeStart'] = df_copy['TimeGroup'].astype(str) + ':00'
        df_copy['TimeEnd'] = ((df_copy['TimeGroup'] + 1) % 24).astype(str) + ':00'
        df_copy['TimeRange'] = df_copy['TimeStart'] + ' - ' + df_copy['TimeEnd']
        chart_title = "P&L by time of the day (Hourly)"
        count_title = "Trade Count by time of the day (Hourly)"
        x_title = "Hour of Day"
    elif granularity == '30min':
        df_copy = df.copy()
        df_copy['TimeFloor'] = df_copy['OpenDateTime'].dt.floor('30min')
        df_copy['TimeGroup'] = df_copy['TimeFloor'].dt.strftime('%H:%M')
        df_copy['TimeStart'] = df_copy['TimeFloor'].dt.strftime('%H:%M')
        df_copy['TimeEnd'] = (df_copy['TimeFloor'] + pd.Timedelta(minutes=30)).dt.strftime('%H:%M')
        df_copy['TimeRange'] = df_copy['TimeStart'] + ' - ' + df_copy['TimeEnd']
        chart_title = "P&L by time of the day (30-minute)"
        count_title = "Trade Count by time of the day (30-minute)"
        x_title = "Time of Day"
    elif granularity == '15min':
        df_copy = df.copy()
        df_copy['TimeFloor'] = df_copy['OpenDateTime'].dt.floor('15min')
        df_copy['TimeGroup'] = df_copy['TimeFloor'].dt.strftime('%H:%M')
        df_copy['TimeStart'] = df_copy['TimeFloor'].dt.strftime('%H:%M')
        df_copy['TimeEnd'] = (df_copy['TimeFloor'] + pd.Timedelta(minutes=15)).dt.strftime('%H:%M')
        df_copy['TimeRange'] = df_copy['TimeStart'] + ' - ' + df_copy['TimeEnd']
        chart_title = "P&L by time of the day (15-minute)"
        count_title = "Trade Count by time of the day (15-minute)"
        x_title = "Time of Day"
    else:
        raise ValueError(f"Invalid granularity: {granularity}. Must be 'Hour', '30min', or '15min'")

    # Group by time period
    time_stats = df_copy.groupby(['TimeGroup', 'TimeRange']).agg({
        'NetPnL': ['sum', 'count'],
        'IsWinner': 'sum'
    }).round(2)

    time_stats.columns = ['Total_PnL', 'Trade_Count', 'Winning_Trades']
    time_stats['Win_Rate'] = (time_stats['Winning_Trades'] / time_stats['Trade_Count']) * 100
    time_stats = time_stats.reset_index()

    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(chart_title, count_title),
        shared_xaxes=True,
        vertical_spacing=0.25
    )

    # Create color mapping for P&L bars (green for profit, red for loss)
    bar_colors = [CHART_COLORS['positive'] if pnl >= 0 else CHART_COLORS['negative'] 
                  for pnl in time_stats['Total_PnL']]

    # P&L by time period
    fig.add_trace(
        go.Bar(
            x=time_stats['TimeRange'],
            y=time_stats['Total_PnL'],
            name='P&L',
            marker_color=bar_colors,
            customdata=time_stats['TimeRange'],
            hovertemplate='<b>Time Range:</b> %{customdata}<br>' +
                         '<b>Total P&L:</b> $%{y:,.2f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )

    # Trade count by time period
    fig.add_trace(
        go.Bar(
            x=time_stats['TimeRange'],
            y=time_stats['Trade_Count'],
            name='Trade Count',
            marker_color=CHART_COLORS['secondary'],
            customdata=time_stats['TimeRange'],
            hovertemplate='<b>Time Range:</b> %{customdata}<br>' +
                         '<b>Trade Count:</b> %{y}<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )

    # Update x-axis labels and formatting
    fig.update_xaxes(title_text=x_title, row=1, col=1)
    fig.update_xaxes(title_text=x_title, row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trade Count", row=2, col=1)
    
    # Rotate x-axis labels for better readability when showing time ranges
    if granularity in ['30min', '15min']:
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=1)

    # Update layout with increased height to accommodate spacing
    layout_config = CHART_THEME.copy()
    layout_config['height'] = 650  # Increased from default 500 to 650
    fig.update_layout(**layout_config)

    return fig


def create_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create drawdown chart showing underwater equity curve.
    
    Args:
        df: DataFrame containing trade data
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return _create_empty_chart("No data available")

    df_sorted = df.sort_values('OpenDateTime')
    cumulative_pnl = df_sorted['NetPnL'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max

    fig = go.Figure()

    # Add drawdown area
    fig.add_trace(
        go.Scatter(
            x=df_sorted['OpenDateTime'],
            y=drawdown,
            fill='tonexty',
            fillcolor=CHART_COLORS['negative_alpha'],
            line=dict(color=CHART_COLORS['negative'], width=2),
            name='Drawdown',
            mode='lines'
        )
    )

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Break-even"
    )

    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown ($)",
        **CHART_THEME
    )

    return fig


def create_trade_duration_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create chart showing trade duration analysis.
    
    Args:
        df: DataFrame containing trade data
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return _create_empty_chart("No data available")

    # Convert duration to hours for better readability
    df['DurationHours'] = df['TradeDurationMinutes'] / 60

    fig = px.scatter(
        df,
        x='DurationHours',
        y='NetPnL',
        color='IsWinner',
        color_discrete_map={True: CHART_COLORS['positive'], False: CHART_COLORS['negative']},
        title='Trade Duration vs P&L',
        labels={
            'DurationHours': 'Trade Duration (Hours)',
            'NetPnL': 'Net P&L ($)',
            'IsWinner': 'Winner'
        },
        hover_data=['TradeType', 'Strategy', 'OpenDateTime']
    )

    fig.update_layout(**CHART_THEME)

    return fig


def _create_cumulative_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """Create cumulative P&L chart."""
    cumulative_pnl = df['NetPnL'].cumsum()

    fig = go.Figure()

    # Add cumulative P&L line
    fig.add_trace(
        go.Scatter(
            x=df['OpenDateTime'],
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color=CHART_COLORS['primary'], width=3),
            fill='tonexty',
            fillcolor=CHART_COLORS['primary_alpha']
        )
    )

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Break-even"
    )

    fig.update_layout(
        title="Cumulative P&L Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        **CHART_THEME
    )

    return fig


def _create_daily_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """Create daily P&L chart."""
    daily_pnl = df.groupby(df['Date'].dt.date)['NetPnL'].sum()

    # Color bars based on positive/negative
    colors = [CHART_COLORS['positive'] if x >= 0 else CHART_COLORS['negative'] for x in daily_pnl.values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=daily_pnl.index,
            y=daily_pnl.values,
            name='Daily P&L',
            marker_color=colors,
            text=daily_pnl.values.round(2),
            textposition='auto',
        )
    )

    fig.update_layout(
        title="Daily P&L",
        xaxis_title="Date",
        yaxis_title="Daily P&L ($)",
        **CHART_THEME
    )

    return fig


def _create_monthly_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """Create monthly P&L chart."""
    monthly_pnl = df.groupby(df['Date'].dt.to_period('M'))['NetPnL'].sum()

    # Color bars based on positive/negative
    colors = [CHART_COLORS['positive'] if x >= 0 else CHART_COLORS['negative'] for x in monthly_pnl.values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[str(x) for x in monthly_pnl.index],
            y=monthly_pnl.values,
            name='Monthly P&L',
            marker_color=colors,
            text=monthly_pnl.values.round(2),
            textposition='auto',
        )
    )

    fig.update_layout(
        title="Monthly P&L",
        xaxis_title="Month",
        yaxis_title="Monthly P&L ($)",
        **CHART_THEME
    )

    return fig


def _create_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """Create histogram for specified column."""
    fig = px.histogram(
        df,
        x=column,
        nbins=50,
        title=f"Distribution of {column}",
        color_discrete_sequence=[CHART_COLORS['primary']]
    )

    fig.update_layout(**CHART_THEME)

    return fig


def _create_box_plot(df: pd.DataFrame, column: str) -> go.Figure:
    """Create box plot for specified column."""
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=df[column],
            name=column,
            marker_color=CHART_COLORS['primary'],
            boxmean='sd'
        )
    )

    fig.update_layout(
        title=f"Box Plot of {column}",
        yaxis_title=column,
        **CHART_THEME
    )

    return fig


def _create_violin_plot(df: pd.DataFrame, column: str) -> go.Figure:
    """Create violin plot for specified column."""
    fig = go.Figure()

    fig.add_trace(
        go.Violin(
            y=df[column],
            name=column,
            fillcolor=CHART_COLORS['primary_alpha'],
            line_color=CHART_COLORS['primary'],
            box_visible=True,
            meanline_visible=True
        )
    )

    fig.update_layout(
        title=f"Violin Plot of {column}",
        yaxis_title=column,
        **CHART_THEME
    )

    return fig


def _create_empty_chart(message: str) -> go.Figure:
    """Create empty chart with message."""
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )

    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        **CHART_THEME
    )

    return fig
