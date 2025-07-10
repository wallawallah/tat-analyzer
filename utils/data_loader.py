"""
Data loading utilities for TAT-Analyzer.

This module handles loading and preprocessing of trade data from CSV files.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from loguru import logger


@st.cache_data
def load_trades_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess trade data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing trade data
        
    Returns:
        DataFrame with cleaned and processed trade data
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file has invalid format
    """
    try:
        # Check if file exists
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} trades from {csv_path}")

        # Validate required columns
        required_columns = [
            'Account', 'Date', 'TimeOpened', 'TimeClosed', 'TradeType',
            'StopType', 'StopMultiple', 'PriceOpen', 'PriceClose',
            'ProfitLoss', 'Status', 'Strategy', 'TradeID'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Data preprocessing
        df = _preprocess_data(df)

        logger.info(f"Processed {len(df)} trades successfully")
        return df

    except Exception as e:
        logger.error(f"Error loading trade data: {str(e)}")
        raise


def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the trade data for analysis.
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Convert date columns
    df['Date'] = pd.to_datetime(df['Date'])
    df['OpenDate'] = pd.to_datetime(df['OpenDate'])
    df['CloseDate'] = pd.to_datetime(df['CloseDate'])

    # Combine date and time columns for precise timestamps
    df['OpenDateTime'] = pd.to_datetime(
        df['OpenDate'].dt.strftime('%Y-%m-%d') + ' ' + df['OpenTime']
    )
    df['CloseDateTime'] = pd.to_datetime(
        df['CloseDate'].dt.strftime('%Y-%m-%d') + ' ' + df['CloseTime']
    )

    # Calculate trade duration
    df['TradeDuration'] = df['CloseDateTime'] - df['OpenDateTime']
    df['TradeDurationMinutes'] = df['TradeDuration'].dt.total_seconds() / 60

    # Clean and convert numeric columns
    numeric_columns = [
        'StopMultiple', 'PriceOpen', 'PriceClose', 'PriceStopTarget',
        'TotalPremium', 'Qty', 'Commission', 'ProfitLoss', 'BuyingPower',
        'StopMultipleResult', 'Slippage'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create derived columns
    # Note: ProfitLoss column already contains net P&L (profit/loss after commissions)
    df['NetPnL'] = df['ProfitLoss']
    df['GrossPnL'] = df['ProfitLoss'] + df['Commission']  # Gross P&L before commissions
    df['IsWinner'] = df['NetPnL'] > 0
    df['ROI'] = (df['NetPnL'] / df['TotalPremium']) * 100

    # Credit trades are those with positive PriceOpen (premium received when opening)
    df['IsCreditTrade'] = df['PriceOpen'] > 0

    # PCR (Premium Capture Rate) for credit trades only
    df['PCR'] = 0.0  # Default to 0 for non-credit trades
    credit_mask = df['IsCreditTrade'] & (df['TotalPremium'] > 0)
    df.loc[credit_mask, 'PCR'] = (df.loc[credit_mask, 'NetPnL'] / df.loc[credit_mask, 'TotalPremium']) * 100

    # Clean strategy names
    df['Strategy'] = df['Strategy'].fillna('Unknown')

    # Create time-based features
    df['TradeWeek'] = df['Date'].dt.isocalendar().week
    df['TradeMonth'] = df['Date'].dt.month
    df['TradeYear'] = df['Date'].dt.year
    df['TradeHour'] = df['OpenDateTime'].dt.hour

    # Sort by date
    df = df.sort_values('OpenDateTime').reset_index(drop=True)

    return df


@st.cache_data
def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for the trade data.
    
    Args:
        df: Trade data DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_trades': len(df),
        'date_range': {
            'start': df['Date'].min(),
            'end': df['Date'].max()
        },
        'strategies': df['Strategy'].nunique(),
        'trade_types': df['TradeType'].nunique(),
        'accounts': df['Account'].nunique(),
        'total_pnl': df['NetPnL'].sum(),
        'win_rate': (df['IsWinner'].sum() / len(df)) * 100,
        'avg_trade_duration': df['TradeDurationMinutes'].mean()
    }

    return summary


def validate_csv_format(csv_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate if the CSV file has the expected format.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to read just the header
        df = pd.read_csv(csv_path, nrows=1)

        # Check for required columns
        required_columns = [
            'Account', 'Date', 'TradeType', 'ProfitLoss', 'Status', 'Strategy'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        return True, None

    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"
