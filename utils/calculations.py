"""
Trade analysis calculations for TAT-Analyzer.

This module contains functions for calculating various trading metrics and performance indicators.
"""


import numpy as np
import pandas as pd
from loguru import logger


def calculate_pnl_metrics(df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate comprehensive P&L metrics for the trade data.

    Args:
        df: DataFrame containing trade data

    Returns:
        Dictionary containing P&L metrics
    """
    if df.empty:
        return {}

    # Calculate slippage statistics
    avg_slippage = df['Slippage'].mean() if 'Slippage' in df.columns else 0.0
    total_slippage = df['Slippage'].sum() if 'Slippage' in df.columns else 0.0

    # Calculate premium sold (positive premium indicates premium received)
    premium_sold = df[df['TotalPremium'] > 0]['TotalPremium'].sum() if 'TotalPremium' in df.columns else 0.0

    metrics = {
        'total_pnl': df['NetPnL'].sum(),
        'gross_pnl': df['GrossPnL'].sum() if 'GrossPnL' in df.columns else df['ProfitLoss'].sum(),
        'total_commissions': df['Commission'].sum(),
        'average_pnl': df['NetPnL'].mean(),
        'median_pnl': df['NetPnL'].median(),
        'best_trade': df['NetPnL'].max(),
        'worst_trade': df['NetPnL'].min(),
        'total_premium': df['TotalPremium'].sum() if 'TotalPremium' in df.columns else 0.0,
        'avg_premium': df['TotalPremium'].mean() if 'TotalPremium' in df.columns else 0.0,
        'roi_percent': (df['NetPnL'].sum() / df['TotalPremium'].sum()) * 100 if 'TotalPremium' in df.columns and df['TotalPremium'].sum() > 0 else 0,
        'pcr_percent': _calculate_pcr(df) if 'IsCreditTrade' in df.columns else 0.0,
        'profit_factor': _calculate_profit_factor(df),
        'sharpe_ratio': _calculate_sharpe_ratio(df),
        'avg_slippage': avg_slippage,
        'total_slippage': total_slippage,
        'premium_sold': premium_sold,
    }

    return metrics


def calculate_win_rate(df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate win rate and related statistics.

    Args:
        df: DataFrame containing trade data

    Returns:
        Dictionary containing win rate metrics
    """
    if df.empty:
        return {}

    winners = df[df['IsWinner']]
    losers = df[~df['IsWinner']]

    # Calculate trade status breakdowns
    stopped_trades = len(df[df['Status'] == 'Stopped']) if 'Status' in df.columns else 0
    expired_trades = len(df[df['Status'] == 'Expired']) if 'Status' in df.columns else 0
    manual_closed_trades = len(df[df['Status'] == 'Manual Closed']) if 'Status' in df.columns else 0
    closed_trades = stopped_trades + expired_trades + manual_closed_trades

    metrics = {
        'win_rate': (len(winners) / len(df)) * 100,
        'total_trades': len(df),
        'winning_trades': len(winners),
        'losing_trades': len(losers),
        'avg_winner': winners['NetPnL'].mean() if not winners.empty else 0,
        'avg_loser': losers['NetPnL'].mean() if not losers.empty else 0,
        'largest_winner': winners['NetPnL'].max() if not winners.empty else 0,
        'largest_loser': losers['NetPnL'].min() if not losers.empty else 0,
        'expectancy': _calculate_expectancy(df),
        'stopped_trades': stopped_trades,
        'expired_trades': expired_trades,
        'manual_closed_trades': manual_closed_trades,
        'closed_trades': closed_trades,
    }

    return metrics


def calculate_drawdown(df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate drawdown metrics for the trade data.

    Args:
        df: DataFrame containing trade data sorted by date

    Returns:
        Dictionary containing drawdown metrics
    """
    if df.empty:
        return {}

    # Calculate cumulative P&L
    df_sorted = df.sort_values('OpenDateTime')
    cumulative_pnl = df_sorted['NetPnL'].cumsum()

    # Calculate running maximum
    running_max = cumulative_pnl.expanding().max()

    # Calculate drawdown
    drawdown = cumulative_pnl - running_max

    # Calculate metrics
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / running_max.max()) * 100 if running_max.max() > 0 else 0

    # Find drawdown periods
    drawdown_periods = _find_drawdown_periods(cumulative_pnl, running_max)

    metrics = {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'current_drawdown': drawdown.iloc[-1],
        'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
        'drawdown_periods': len(drawdown_periods),
        'longest_drawdown_days': _calculate_longest_drawdown_period(df_sorted, drawdown),
    }

    return metrics


def calculate_strategy_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics grouped by strategy.

    Args:
        df: DataFrame containing trade data

    Returns:
        DataFrame with strategy performance metrics
    """
    if df.empty:
        return pd.DataFrame()

    strategy_stats = df.groupby('Strategy').agg({
        'NetPnL': ['sum', 'mean', 'count'],
        'IsWinner': 'sum',
        'TotalPremium': 'sum',
        'Commission': 'sum',
        'TradeDurationMinutes': 'mean'
    }).round(2)

    # Flatten column names
    strategy_stats.columns = [
        'Total_PnL', 'Avg_PnL', 'Trade_Count', 'Winning_Trades',
        'Total_Premium', 'Total_Commission', 'Avg_Duration_Min'
    ]

    # Calculate additional metrics with safety checks
    strategy_stats['Win_Rate'] = (strategy_stats['Winning_Trades'] / strategy_stats['Trade_Count']) * 100
    strategy_stats['ROI_Percent'] = 0.0
    strategy_stats['Profit_Per_Trade'] = strategy_stats['Total_PnL'] / strategy_stats['Trade_Count']

    # Calculate ROI with safety check for division by zero
    for idx in strategy_stats.index:
        if strategy_stats.loc[idx, 'Total_Premium'] > 0:
            strategy_stats.loc[idx, 'ROI_Percent'] = (strategy_stats.loc[idx, 'Total_PnL'] / strategy_stats.loc[idx, 'Total_Premium']) * 100

    # Calculate PCR for each strategy
    strategy_stats['PCR_Percent'] = 0.0
    for idx, _row in strategy_stats.iterrows():
        strategy_name = "Unknown"
        try:
            # The strategy name is in the index at this point
            strategy_name = idx
            strategy_data = df[df['Strategy'] == strategy_name]
            if not strategy_data.empty:
                strategy_stats.loc[idx, 'PCR_Percent'] = _calculate_pcr(strategy_data)
            else:
                strategy_stats.loc[idx, 'PCR_Percent'] = 0.0
        except Exception as e:
            logger.warning(f"Error calculating PCR for strategy {strategy_name}: {e}")
            strategy_stats.loc[idx, 'PCR_Percent'] = 0.0

    return strategy_stats.reset_index()


def calculate_time_based_performance(df: pd.DataFrame, period: str = 'daily') -> pd.DataFrame:
    """
    Calculate performance metrics over time periods.

    Args:
        df: DataFrame containing trade data
        period: Time period for grouping ('daily', 'weekly', 'monthly')

    Returns:
        DataFrame with time-based performance metrics
    """
    if df.empty:
        return pd.DataFrame()

    df_sorted = df.sort_values('OpenDateTime')

    # Define grouping based on period
    if period == 'daily':
        group_col = df_sorted['Date'].dt.date
    elif period == 'weekly':
        group_col = df_sorted['Date'].dt.to_period('W')
    elif period == 'monthly':
        group_col = df_sorted['Date'].dt.to_period('M')
    else:
        raise ValueError("Period must be 'daily', 'weekly', or 'monthly'")

    # Group and calculate metrics
    time_stats = df_sorted.groupby(group_col).agg({
        'NetPnL': ['sum', 'count'],
        'IsWinner': 'sum',
        'TotalPremium': 'sum'
    }).round(2)

    # Flatten column names
    time_stats.columns = ['PnL', 'Trades', 'Winners', 'Premium']

    # Calculate additional metrics
    time_stats['Win_Rate'] = (time_stats['Winners'] / time_stats['Trades']) * 100
    time_stats['Cumulative_PnL'] = time_stats['PnL'].cumsum()

    return time_stats.reset_index()


def calculate_risk_metrics(df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate risk-related metrics for the trade data.

    Args:
        df: DataFrame containing trade data

    Returns:
        Dictionary containing risk metrics
    """
    if df.empty:
        return {}

    pnl_series = df['NetPnL']

    metrics = {
        'volatility': pnl_series.std(),
        'downside_deviation': _calculate_downside_deviation(pnl_series),
        'var_95': np.percentile(pnl_series, 5),  # Value at Risk (95%)
        'var_99': np.percentile(pnl_series, 1),  # Value at Risk (99%)
        'skewness': pnl_series.skew(),
        'kurtosis': pnl_series.kurtosis(),
        'calmar_ratio': _calculate_calmar_ratio(df),
        'sortino_ratio': _calculate_sortino_ratio(df),
    }

    return metrics


def _calculate_pcr(df: pd.DataFrame) -> float:
    """Calculate PCR (Premium Capture Rate) for credit trades only."""
    if df.empty:
        return 0.0

    # Check if required columns exist
    required_columns = ['IsCreditTrade', 'TotalPremium', 'NetPnL']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.warning(f"Missing columns for PCR calculation: {missing_columns}")
        return 0.0

    # Filter for credit trades only (positive PriceOpen)
    try:
        credit_trades = df[df['IsCreditTrade']]

        if credit_trades.empty:
            return 0.0

        # Calculate PCR only for credit trades with positive total premium
        valid_credit_trades = credit_trades[credit_trades['TotalPremium'] > 0]

        if valid_credit_trades.empty:
            return 0.0

        total_credit_pnl = valid_credit_trades['NetPnL'].sum()
        total_credit_premium = valid_credit_trades['TotalPremium'].sum()

        if total_credit_premium == 0:
            return 0.0

        return (total_credit_pnl / total_credit_premium) * 100

    except Exception as e:
        logger.warning(f"Error in PCR calculation: {e}")
        return 0.0


def _calculate_profit_factor(df: pd.DataFrame) -> float:
    """Calculate profit factor (gross profits / gross losses)."""
    winners = df[df['NetPnL'] > 0]['NetPnL'].sum()
    losers = abs(df[df['NetPnL'] < 0]['NetPnL'].sum())
    return winners / losers if losers > 0 else float('inf')


def _calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio for the trade data."""
    if df.empty:
        return 0.0

    returns = df['NetPnL']
    excess_returns = returns.mean() - risk_free_rate
    return excess_returns / returns.std() if returns.std() > 0 else 0.0


def _calculate_expectancy(df: pd.DataFrame) -> float:
    """Calculate expectancy (average P&L per trade)."""
    if df.empty:
        return 0.0

    winners = df[df['IsWinner']]
    losers = df[~df['IsWinner']]

    if len(winners) == 0 or len(losers) == 0:
        return df['NetPnL'].mean()

    win_rate = len(winners) / len(df)
    avg_win = winners['NetPnL'].mean()
    avg_loss = losers['NetPnL'].mean()

    return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)


def _calculate_downside_deviation(returns: pd.Series, target_return: float = 0.0) -> float:
    """Calculate downside deviation."""
    downside_returns = returns[returns < target_return]
    return downside_returns.std() if not downside_returns.empty else 0.0


def _calculate_calmar_ratio(df: pd.DataFrame) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)."""
    if df.empty:
        return 0.0

    total_return = df['NetPnL'].sum()
    drawdown_metrics = calculate_drawdown(df)
    max_drawdown = abs(drawdown_metrics.get('max_drawdown', 0))

    return total_return / max_drawdown if max_drawdown > 0 else 0.0


def _calculate_sortino_ratio(df: pd.DataFrame, target_return: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    if df.empty:
        return 0.0

    returns = df['NetPnL']
    excess_returns = returns.mean() - target_return
    downside_deviation = _calculate_downside_deviation(returns, target_return)

    return excess_returns / downside_deviation if downside_deviation > 0 else 0.0


def _find_drawdown_periods(cumulative_pnl: pd.Series, running_max: pd.Series) -> list[tuple[int, int]]:
    """Find periods of drawdown in the cumulative P&L."""
    drawdown = cumulative_pnl - running_max
    in_drawdown = drawdown < 0

    periods = []
    start_idx = None

    for i, is_dd in enumerate(in_drawdown):
        if is_dd and start_idx is None:
            start_idx = i
        elif not is_dd and start_idx is not None:
            periods.append((start_idx, i - 1))
            start_idx = None

    # Handle case where drawdown continues to the end
    if start_idx is not None:
        periods.append((start_idx, len(in_drawdown) - 1))

    return periods


def _calculate_longest_drawdown_period(df: pd.DataFrame, drawdown: pd.Series) -> int:
    """Calculate the longest drawdown period in days."""
    if df.empty or drawdown.empty:
        return 0

    in_drawdown = drawdown < 0
    max_period = 0
    current_period = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_period += 1
            max_period = max(max_period, current_period)
        else:
            current_period = 0

    return max_period
