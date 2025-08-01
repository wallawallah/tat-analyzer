"""
Data loading utilities for TAT-Analyzer.

This module handles loading and preprocessing of trade data from CSV files.
"""

import csv
import io
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from loguru import logger

from .session_manager import get_session_cache, get_session_id


def detect_csv_format(csv_path: str) -> Tuple[str, str, str]:
    """
    Detect CSV delimiter, decimal separator, and date format.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (delimiter, decimal_separator, date_format)
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Read first few lines to analyze
            sample_lines = [f.readline() for _ in range(min(5, sum(1 for _ in f) + 1))]
        
        # Reset file pointer and read again
        with open(csv_path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)  # First 2KB
        
        # Detect delimiter using CSV sniffer
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample, delimiters=',;').delimiter
        except:
            # Fallback: count occurrences
            comma_count = sample.count(',')
            semicolon_count = sample.count(';')
            delimiter = ';' if semicolon_count > comma_count else ','
        
        # Detect decimal separator by looking at numeric patterns
        decimal_separator = '.'  # Default
        
        # Look for patterns like "1,23" vs "1.23" in numeric contexts
        comma_decimal_pattern = r'\d+,\d{1,6}(?:[^\d,]|$)'
        period_decimal_pattern = r'\d+\.\d{1,6}(?:[^\d.]|$)'
        
        comma_decimals = len(re.findall(comma_decimal_pattern, sample))
        period_decimals = len(re.findall(period_decimal_pattern, sample))
        
        # If we find more comma decimals and delimiter is semicolon, likely European format
        if comma_decimals > period_decimals and delimiter == ';':
            decimal_separator = ','
        
        # Detect date format by looking for DD.MM.YYYY vs MM/DD/YYYY patterns
        date_format = '%m/%d/%Y'  # Default US format
        
        # Look for European date patterns (DD.MM.YYYY)
        european_date_pattern = r'\b\d{1,2}\.\d{1,2}\.\d{4}\b'
        us_date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'
        iso_date_pattern = r'\b\d{4}-\d{1,2}-\d{1,2}\b'
        
        european_dates = len(re.findall(european_date_pattern, sample))
        us_dates = len(re.findall(us_date_pattern, sample))
        iso_dates = len(re.findall(iso_date_pattern, sample))
        
        if european_dates > us_dates and european_dates > 0:
            date_format = '%d.%m.%Y'
        elif iso_dates > us_dates and iso_dates > european_dates:
            date_format = '%Y-%m-%d'
        
        logger.info(f"Detected CSV format - delimiter: '{delimiter}', decimal: '{decimal_separator}', date: '{date_format}'")
        return delimiter, decimal_separator, date_format
        
    except Exception as e:
        logger.warning(f"Could not detect CSV format, using defaults: {e}")
        return ',', '.', '%m/%d/%Y'


def preprocess_locale_numbers(df: pd.DataFrame, decimal_separator: str) -> pd.DataFrame:
    """
    Preprocess numeric columns to handle locale-specific decimal separators.
    
    Args:
        df: DataFrame with potentially locale-specific numbers
        decimal_separator: The decimal separator used in the data (',' or '.')
        
    Returns:
        DataFrame with standardized numeric columns
    """
    if decimal_separator == ',':
        # Convert comma decimals to period decimals for pandas
        numeric_columns = [
            'StopMultiple', 'PriceOpen', 'PriceClose', 'PriceStopTarget',
            'TotalPremium', 'Qty', 'Commission', 'ProfitLoss', 'BuyingPower',
            'StopMultipleResult', 'Slippage', 'PutDelta', 'CallDelta',
            'PriceLong', 'PriceShort'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert comma decimals to period decimals
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    
    return df


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

        # Detect CSV format (delimiter, decimal separator, and date format)
        delimiter, decimal_separator, date_format = detect_csv_format(csv_path)

        # Load CSV data with detected delimiter and proper quoting
        df = pd.read_csv(csv_path, delimiter=delimiter, quoting=csv.QUOTE_ALL, on_bad_lines='skip')
        logger.info(f"Loaded {len(df)} trades from {csv_path} using delimiter '{delimiter}'")

        # Preprocess locale-specific numbers before validation
        df = preprocess_locale_numbers(df, decimal_separator)

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
        df = _preprocess_data(df, date_format)

        logger.info(f"Processed {len(df)} trades successfully")
        return df

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {csv_path}")
        raise ValueError("The uploaded CSV file is empty. Please check your file.")
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {str(e)}")
        raise ValueError(f"Unable to parse CSV file. This might be due to formatting issues or unexpected delimiters. Error: {str(e)}")
    except ValueError as e:
        if "Missing required columns" in str(e):
            logger.error(f"Column validation failed: {str(e)}")
            raise ValueError(f"CSV file format error: {str(e)}. Please ensure you're uploading a valid TAT export file.")
        else:
            logger.error(f"Data validation error: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error loading trade data: {str(e)}")
        raise ValueError(f"An unexpected error occurred while processing your CSV file. Please check the file format and try again. Error: {str(e)}")


def _preprocess_data(df: pd.DataFrame, date_format: str = '%m/%d/%Y') -> pd.DataFrame:
    """
    Preprocess the trade data for analysis.

    Args:
        df: Raw DataFrame from CSV
        date_format: Date format string for parsing (e.g., '%d.%m.%Y' for European)

    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Convert date columns with robust parsing
    def parse_date_with_fallback(date_series, primary_format):
        """Parse dates with primary format and fallback options."""
        try:
            # Try primary format first
            if primary_format == '%d.%m.%Y':
                # European format - use dayfirst=True
                return pd.to_datetime(date_series, format=primary_format, errors='coerce').fillna(
                    pd.to_datetime(date_series, dayfirst=True, errors='coerce')
                )
            else:
                # US or ISO format
                return pd.to_datetime(date_series, format=primary_format, errors='coerce').fillna(
                    pd.to_datetime(date_series, errors='coerce')
                )
        except:
            # Ultimate fallback - let pandas infer
            return pd.to_datetime(date_series, errors='coerce', dayfirst=(primary_format == '%d.%m.%Y'))
    
    df['Date'] = parse_date_with_fallback(df['Date'], date_format)
    
    # OpenDate and CloseDate are typically in ISO format (YYYY-MM-DD) regardless of locale
    df['OpenDate'] = pd.to_datetime(df['OpenDate'], errors='coerce')
    df['CloseDate'] = pd.to_datetime(df['CloseDate'], errors='coerce')

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
    
    # Clean template names - ensure no NaN values that would cause sorting errors
    if 'Template' in df.columns:
        df['Template'] = df['Template'].fillna('Unknown')
    
    # Ensure other categorical columns don't have NaN values that could cause sorting errors
    categorical_columns = ['TradeType', 'Status', 'StopType', 'Account']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Create time-based features
    df['TradeWeek'] = df['Date'].dt.isocalendar().week
    df['TradeMonth'] = df['Date'].dt.month
    df['TradeYear'] = df['Date'].dt.year
    df['TradeHour'] = df['OpenDateTime'].dt.hour

    # Sort by date
    df = df.sort_values('OpenDateTime').reset_index(drop=True)

    return df


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


def load_trades_from_buffer(file_buffer: io.BytesIO, original_filename: str = "uploaded.csv") -> pd.DataFrame:
    """
    Load and preprocess trade data from an in-memory buffer.
    
    This function uses session-aware caching to improve performance while
    maintaining user isolation.
    
    Args:
        file_buffer: BytesIO buffer containing the CSV data
        original_filename: Original filename for logging purposes
        
    Returns:
        DataFrame with cleaned and processed trade data
        
    Raises:
        ValueError: If the CSV file has invalid format
    """
    # Get session ID and cache
    session_id = get_session_id()
    cache = get_session_cache()
    
    # Create cache key from file content
    file_content = file_buffer.getvalue()
    cache_key = cache.get_cache_key(file_content, session_id)
    
    # Check cache first
    cached_df = cache.get(cache_key)
    if cached_df is not None:
        logger.info(f"Loaded {len(cached_df)} trades from cache for {original_filename}")
        return cached_df
    
    # Reset buffer position
    file_buffer.seek(0)
    
    try:
        # Detect CSV format from buffer
        sample = file_buffer.read(2048)
        file_buffer.seek(0)  # Reset after sampling
        
        delimiter, decimal_separator, date_format = _detect_format_from_sample(sample.decode('utf-8'))
        
        # Load CSV data with detected delimiter
        df = pd.read_csv(file_buffer, delimiter=delimiter, quoting=csv.QUOTE_ALL, on_bad_lines='skip')
        logger.info(f"Loaded {len(df)} trades from {original_filename} using delimiter '{delimiter}'")
        
        # Preprocess locale-specific numbers before validation
        df = preprocess_locale_numbers(df, decimal_separator)
        
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
        df = _preprocess_data(df, date_format)
        
        # Cache the processed data
        cache.set(cache_key, df)
        
        logger.info(f"Processed and cached {len(df)} trades successfully")
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {original_filename}")
        raise ValueError("The uploaded CSV file is empty. Please check your file.")
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {str(e)}")
        raise ValueError(f"Unable to parse CSV file. This might be due to formatting issues or unexpected delimiters. Error: {str(e)}")
    except ValueError as e:
        if "Missing required columns" in str(e):
            logger.error(f"Column validation failed: {str(e)}")
            raise ValueError(f"CSV file format error: {str(e)}. Please ensure you're uploading a valid TAT export file.")
        else:
            logger.error(f"Data validation error: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error loading trade data: {str(e)}")
        raise ValueError(f"An unexpected error occurred while processing your CSV file. Please check the file format and try again. Error: {str(e)}")


def _detect_format_from_sample(sample: str) -> Tuple[str, str, str]:
    """
    Detect CSV format from a string sample.
    
    Args:
        sample: String sample of the CSV file
        
    Returns:
        Tuple of (delimiter, decimal_separator, date_format)
    """
    # Use CSV sniffer to detect delimiter
    sniffer = csv.Sniffer()
    try:
        delimiter = sniffer.sniff(sample, delimiters=',;').delimiter
    except:
        # Fallback: count occurrences
        comma_count = sample.count(',')
        semicolon_count = sample.count(';')
        delimiter = ';' if semicolon_count > comma_count else ','
    
    # Detect decimal separator
    decimal_separator = '.'
    comma_decimal_pattern = r'\d+,\d{1,6}(?:[^\d,]|$)'
    period_decimal_pattern = r'\d+\.\d{1,6}(?:[^\d.]|$)'
    
    comma_decimals = len(re.findall(comma_decimal_pattern, sample))
    period_decimals = len(re.findall(period_decimal_pattern, sample))
    
    if comma_decimals > period_decimals and delimiter == ';':
        decimal_separator = ','
    
    # Detect date format
    date_format = '%m/%d/%Y'  # Default US format
    
    european_date_pattern = r'\b\d{1,2}\.\d{1,2}\.\d{4}\b'
    us_date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'
    iso_date_pattern = r'\b\d{4}-\d{1,2}-\d{1,2}\b'
    
    european_dates = len(re.findall(european_date_pattern, sample))
    us_dates = len(re.findall(us_date_pattern, sample))
    iso_dates = len(re.findall(iso_date_pattern, sample))
    
    if european_dates > us_dates and european_dates > 0:
        date_format = '%d.%m.%Y'
    elif iso_dates > us_dates and iso_dates > european_dates:
        date_format = '%Y-%m-%d'
    
    return delimiter, decimal_separator, date_format


def validate_csv_format(csv_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate if the CSV file has the expected format.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Detect CSV format first
        delimiter, decimal_separator, date_format = detect_csv_format(csv_path)
        
        # Try to read just the header with detected delimiter and proper quoting
        df = pd.read_csv(csv_path, delimiter=delimiter, quoting=csv.QUOTE_ALL, nrows=1, on_bad_lines='skip')

        # Check for required columns
        required_columns = [
            'Account', 'Date', 'TradeType', 'ProfitLoss', 'Status', 'Strategy'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        return True, None

    except Exception as e:
        return False, f"Error reading CSV with locale detection: {str(e)}"
