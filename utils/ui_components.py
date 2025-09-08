"""
UI components and common elements for TAT-Analyzer.

This module contains reusable UI components that can be used across different pages.
"""

import streamlit as st


def display_coffee_support():
    """
    Display the 'Buy me a coffee' support section in the sidebar.
    
    This function renders a consistent support section that can be used
    across all pages of the application.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Support This Project")
    st.sidebar.markdown("If you find this tool helpful, consider buying me a coffee! ☕")
    
    st.sidebar.link_button("☕ Buy me a coffee", "https://coff.ee/wallawallah", use_container_width=True)