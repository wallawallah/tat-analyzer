"""
Session management utilities for TAT-Analyzer.

This module provides session-aware caching to prevent data leakage between users
and ensures proper isolation of user data in multi-user environments.
"""

import hashlib
import time
import uuid
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from loguru import logger


class SessionAwareCache:
    """
    Custom cache implementation that ties cached data to user sessions.
    
    This prevents data leakage between users by ensuring cache keys include
    session identifiers and implements automatic TTL-based cleanup.
    """
    
    def __init__(self, ttl_seconds: int = 1800):  # 30 minutes default
        """
        Initialize the session-aware cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttl = ttl_seconds
        logger.info(f"Initialized SessionAwareCache with TTL of {ttl_seconds} seconds")
    
    def get_cache_key(self, file_content: bytes, session_id: str) -> str:
        """
        Create a unique cache key from file content and session ID.
        
        Args:
            file_content: The raw bytes of the uploaded file
            session_id: The unique session identifier
            
        Returns:
            A unique cache key string
        """
        content_hash = hashlib.sha256(file_content).hexdigest()[:16]  # Use first 16 chars of hash
        return f"{session_id}:{content_hash}"
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve an item from cache with expiration check.
        
        Args:
            key: The cache key
            
        Returns:
            The cached DataFrame or None if not found/expired
        """
        self._cleanup_expired()
        
        if key in self.cache:
            self.timestamps[key] = time.time()  # Update access time
            logger.debug(f"Cache hit for key: {key}")
            return self.cache[key]
        
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, value: pd.DataFrame) -> None:
        """
        Store an item in the cache.
        
        Args:
            key: The cache key
            value: The DataFrame to cache
        """
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self._cleanup_expired()
        logger.debug(f"Cached data with key: {key}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
            logger.debug(f"Removed expired cache entry: {key}")
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all cache entries for a specific session.
        
        Args:
            session_id: The session ID to clear
        """
        session_keys = [
            key for key in self.cache.keys() 
            if key.startswith(f"{session_id}:")
        ]
        
        for key in session_keys:
            del self.cache[key]
            del self.timestamps[key]
        
        if session_keys:
            logger.info(f"Cleared {len(session_keys)} cache entries for session {session_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return {
            "total_entries": len(self.cache),
            "memory_usage_mb": sum(
                df.memory_usage(deep=True).sum() for df in self.cache.values()
            ) / (1024 * 1024),
            "oldest_entry_age": max(
                time.time() - ts for ts in self.timestamps.values()
            ) if self.timestamps else 0
        }


def get_session_id() -> str:
    """
    Get or create a unique session ID for the current user session.
    
    Returns:
        A unique session identifier
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"Created new session ID: {st.session_state.session_id}")
    
    return st.session_state.session_id


def get_session_cache() -> SessionAwareCache:
    """
    Get the global session-aware cache instance.
    
    Returns:
        The SessionAwareCache instance
    """
    # Store cache at the app level, not in session state
    # This allows sharing the cache object while maintaining session isolation
    cache_key = '_session_aware_cache'
    
    if not hasattr(st, cache_key):
        setattr(st, cache_key, SessionAwareCache())
        logger.info("Initialized global session-aware cache")
    
    return getattr(st, cache_key)


def cleanup_current_session() -> None:
    """Clean up all data for the current session."""
    session_id = get_session_id()
    cache = get_session_cache()
    cache.clear_session(session_id)
    
    # Clear session state data
    if 'trade_data' in st.session_state:
        del st.session_state.trade_data
    if 'data_summary' in st.session_state:
        del st.session_state.data_summary
    if 'data_loaded' in st.session_state:
        st.session_state.data_loaded = False
    
    logger.info(f"Cleaned up session: {session_id}")