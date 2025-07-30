"""
Security utilities for TAT-Analyzer.

This module provides validation and security features to ensure safe file handling
and prevent malicious inputs.
"""

from typing import BinaryIO, Optional, Tuple

import pandas as pd
import streamlit as st
from loguru import logger


# Maximum file size in bytes (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.csv'}


def validate_file_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded file for security and format requirements.
    
    Args:
        uploaded_file: The Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        error_msg = f"File size ({uploaded_file.size / (1024*1024):.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE / (1024*1024)}MB)"
        logger.warning(f"File upload rejected: {error_msg}")
        return False, error_msg
    
    # Check file extension
    file_extension = '.' + uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ''
    if file_extension not in ALLOWED_EXTENSIONS:
        error_msg = f"Invalid file type. Only CSV files are allowed."
        logger.warning(f"File upload rejected: {error_msg} (got {file_extension})")
        return False, error_msg
    
    # Check if file is empty
    if uploaded_file.size == 0:
        error_msg = "The uploaded file is empty."
        logger.warning("File upload rejected: Empty file")
        return False, error_msg
    
    # Validate filename for path traversal attempts
    if '..' in uploaded_file.name or '/' in uploaded_file.name or '\\' in uploaded_file.name:
        error_msg = "Invalid filename detected."
        logger.warning(f"File upload rejected: Suspicious filename '{uploaded_file.name}'")
        return False, error_msg
    
    # Basic content validation - check if it starts like a CSV
    try:
        # Read first few bytes to check if it looks like text
        content_sample = uploaded_file.read(1024)
        uploaded_file.seek(0)  # Reset file pointer
        
        # Check if content is binary (not text)
        try:
            content_sample.decode('utf-8')
        except UnicodeDecodeError:
            # Try with common encodings
            try:
                content_sample.decode('latin-1')
            except:
                error_msg = "File appears to be binary, not a text CSV file."
                logger.warning("File upload rejected: Binary content detected")
                return False, error_msg
        
    except Exception as e:
        error_msg = f"Error validating file content: {str(e)}"
        logger.error(f"File validation error: {str(e)}")
        return False, error_msg
    
    logger.info(f"File upload validated successfully: {uploaded_file.name} ({uploaded_file.size} bytes)")
    return True, None


def sanitize_dataframe_values(df) -> None:
    """
    Sanitize DataFrame values to prevent XSS or injection attacks when displaying data.
    
    This function modifies the DataFrame in-place.
    
    Args:
        df: The pandas DataFrame to sanitize
    """
    # String columns that might contain user input
    string_columns = df.select_dtypes(include=['object']).columns
    
    for col in string_columns:
        # Remove any HTML/script tags from string values
        df[col] = df[col].astype(str).str.replace(r'<[^>]+>', '', regex=True)
        
        # Escape special characters that could be interpreted as code
        df[col] = df[col].str.replace('&', '&amp;')\
                        .str.replace('<', '&lt;')\
                        .str.replace('>', '&gt;')\
                        .str.replace('"', '&quot;')\
                        .str.replace("'", '&#x27;')
    
    logger.debug(f"Sanitized {len(string_columns)} string columns in DataFrame")


def get_safe_session_info() -> dict:
    """
    Get session information for logging without exposing sensitive data.
    
    Returns:
        Dictionary with safe session information
    """
    from .session_manager import get_session_id
    
    session_id = get_session_id()
    # Only include first 8 characters of session ID for privacy
    safe_session_id = session_id[:8] + "..."
    
    return {
        "session_id": safe_session_id,
        "data_loaded": st.session_state.get('data_loaded', False),
        "has_trade_data": 'trade_data' in st.session_state,
        "timestamp": pd.Timestamp.now().isoformat()
    }