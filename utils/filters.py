"""
Centralized filter system for TAT-Analyzer.

This module provides reusable filter components to eliminate duplicate code
across pages and provide consistent filtering functionality.
"""

from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import streamlit as st

from .constants import DATE_RANGES


class BaseFilter:
    """Base class for all filters."""

    def __init__(self, key: str, label: str):
        self.key = key
        self.label = label
        self.value = None

    def render(self) -> Any:
        """Render the filter UI component."""
        raise NotImplementedError

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter to the dataframe."""
        raise NotImplementedError


class DateRangeFilter(BaseFilter):
    """Filter for date ranges with predefined and custom options."""

    def __init__(self, key: str = "date_range", label: str = "Date Range",
                 date_column: str = "Date", show_custom: bool = True):
        super().__init__(key, label)
        self.date_column = date_column
        self.show_custom = show_custom
        self.selected_range = None
        self.custom_start = None
        self.custom_end = None

    def render(self) -> dict[str, Any]:
        """Render date range filter UI."""
        st.sidebar.subheader(self.label)

        # Date range selection
        self.selected_range = st.sidebar.selectbox(
            "Select date range:",
            options=list(DATE_RANGES.keys()),
            index=list(DATE_RANGES.keys()).index("All time"),
            key=f"{self.key}_range"
        )

        # Custom date range inputs
        if self.selected_range == "Custom" and self.show_custom:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                self.custom_start = st.date_input(
                    "Start date",
                    key=f"{self.key}_start"
                )
            with col2:
                self.custom_end = st.date_input(
                    "End date",
                    key=f"{self.key}_end"
                )

        return {
            "selected_range": self.selected_range,
            "custom_start": self.custom_start,
            "custom_end": self.custom_end
        }

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply date range filter to dataframe."""
        if df.empty or self.date_column not in df.columns:
            return df

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        if self.selected_range == "Custom":
            if self.custom_start and self.custom_end:
                start_date = pd.to_datetime(self.custom_start)
                end_date = pd.to_datetime(self.custom_end) + timedelta(days=1)
                return df[(df[self.date_column] >= start_date) & (df[self.date_column] < end_date)]
            return df

        days_back = DATE_RANGES.get(self.selected_range)
        if days_back is None:  # "All time"
            return df

        if days_back == 0:  # "Today"
            today = datetime.now().date()
            return df[df[self.date_column].dt.date == today]

        if days_back == 1:  # "Yesterday"
            yesterday = datetime.now().date() - timedelta(days=1)
            return df[df[self.date_column].dt.date == yesterday]

        # Handle week-based ranges
        if days_back == 'this_week':
            today = datetime.now().date()
            # Get start of this week (Monday)
            start_of_week = today - timedelta(days=today.weekday())
            return df[df[self.date_column].dt.date >= start_of_week]

        if days_back == 'last_week':
            today = datetime.now().date()
            # Get start of last week (Monday)
            start_of_last_week = today - timedelta(days=today.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=6)
            return df[(df[self.date_column].dt.date >= start_of_last_week) &
                     (df[self.date_column].dt.date <= end_of_last_week)]

        # Handle month-based ranges
        if days_back == 'this_month':
            today = datetime.now().date()
            # Get first day of current month
            start_of_month = today.replace(day=1)
            return df[df[self.date_column].dt.date >= start_of_month]

        if days_back == 'last_month':
            today = datetime.now().date()
            # Get first day of last month
            if today.month == 1:
                start_of_last_month = today.replace(year=today.year - 1, month=12, day=1)
            else:
                start_of_last_month = today.replace(month=today.month - 1, day=1)
            # Get last day of last month
            end_of_last_month = today.replace(day=1) - timedelta(days=1)
            return df[(df[self.date_column].dt.date >= start_of_last_month) &
                     (df[self.date_column].dt.date <= end_of_last_month)]

        # For other ranges (numeric days back)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        return df[df[self.date_column] >= cutoff_date]


class CategoryFilter(BaseFilter):
    """Filter for categorical data with single or multiple selection."""

    def __init__(self, key: str, label: str, column: str,
                 options: list[str] = None, multi_select: bool = False,
                 default_all: bool = True):
        super().__init__(key, label)
        self.column = column
        self.options = options or []
        self.multi_select = multi_select
        self.default_all = default_all
        self.selected_values = None

    def render(self) -> Any:
        """Render category filter UI."""
        if not self.options:
            return None

        # Add "All" option for single select
        if not self.multi_select and self.default_all:
            display_options = ["All"] + self.options
        else:
            display_options = self.options

        if self.multi_select:
            self.selected_values = st.sidebar.multiselect(
                self.label,
                options=self.options,
                default=self.options if self.default_all else [],
                key=f"{self.key}_multi"
            )
        else:
            selected = st.sidebar.selectbox(
                self.label,
                options=display_options,
                key=f"{self.key}_single"
            )
            self.selected_values = selected if selected != "All" else None

        return self.selected_values

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply category filter to dataframe."""
        if df.empty or self.column not in df.columns:
            return df

        if self.multi_select:
            if self.selected_values:
                return df[df[self.column].isin(self.selected_values)]
        else:
            if self.selected_values:
                return df[df[self.column] == self.selected_values]

        return df


class NumericRangeFilter(BaseFilter):
    """Filter for numeric ranges using sliders."""

    def __init__(self, key: str, label: str, column: str,
                 min_val: float = None, max_val: float = None,
                 step: float = 1.0, format_str: str = "%.2f"):
        super().__init__(key, label)
        self.column = column
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.format_str = format_str
        self.selected_range = None

    def render(self) -> tuple:
        """Render numeric range filter UI."""
        if self.min_val is None or self.max_val is None:
            return None

        self.selected_range = st.sidebar.slider(
            self.label,
            min_value=self.min_val,
            max_value=self.max_val,
            value=(self.min_val, self.max_val),
            step=self.step,
            format=self.format_str,
            key=f"{self.key}_range"
        )

        return self.selected_range

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply numeric range filter to dataframe."""
        if df.empty or self.column not in df.columns or not self.selected_range:
            return df

        min_val, max_val = self.selected_range
        return df[(df[self.column] >= min_val) & (df[self.column] <= max_val)]


class TextSearchFilter(BaseFilter):
    """Filter for text search across multiple columns."""

    def __init__(self, key: str, label: str, search_columns: list[str],
                 placeholder: str = "Search..."):
        super().__init__(key, label)
        self.search_columns = search_columns
        self.placeholder = placeholder
        self.search_term = None

    def render(self) -> str:
        """Render text search filter UI."""
        self.search_term = st.sidebar.text_input(
            self.label,
            placeholder=self.placeholder,
            key=f"{self.key}_search"
        )

        return self.search_term

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text search filter to dataframe."""
        if df.empty or not self.search_term:
            return df

        # Create a mask for search across multiple columns
        search_mask = pd.Series([False] * len(df))

        for column in self.search_columns:
            if column in df.columns:
                # Convert to string and search (case-insensitive)
                column_mask = df[column].astype(str).str.contains(
                    self.search_term, case=False, na=False
                )
                search_mask = search_mask | column_mask

        return df[search_mask]


class FilterManager:
    """Manages multiple filters and applies them to dataframes."""

    def __init__(self):
        self.filters: list[BaseFilter] = []
        self.sidebar_title = None

    def set_sidebar_title(self, title: str):
        """Set the sidebar title."""
        self.sidebar_title = title

    def add_filter(self, filter_obj: BaseFilter):
        """Add a filter to the manager."""
        self.filters.append(filter_obj)

    def add_date_range(self, key: str = "date_range", label: str = "Date Range",
                      date_column: str = "Date", show_custom: bool = True):
        """Add a date range filter."""
        filter_obj = DateRangeFilter(key, label, date_column, show_custom)
        self.add_filter(filter_obj)
        return filter_obj

    def add_category(self, key: str, label: str, column: str,
                    options: list[str] = None, multi_select: bool = False,
                    default_all: bool = True):
        """Add a category filter."""
        filter_obj = CategoryFilter(key, label, column, options, multi_select, default_all)
        self.add_filter(filter_obj)
        return filter_obj

    def add_numeric_range(self, key: str, label: str, column: str,
                         min_val: float = None, max_val: float = None,
                         step: float = 1.0, format_str: str = "%.2f"):
        """Add a numeric range filter."""
        filter_obj = NumericRangeFilter(key, label, column, min_val, max_val, step, format_str)
        self.add_filter(filter_obj)
        return filter_obj

    def add_text_search(self, key: str, label: str, search_columns: list[str],
                       placeholder: str = "Search..."):
        """Add a text search filter."""
        filter_obj = TextSearchFilter(key, label, search_columns, placeholder)
        self.add_filter(filter_obj)
        return filter_obj

    def render_sidebar(self):
        """Render all filters in the sidebar."""
        if self.sidebar_title:
            st.sidebar.title(self.sidebar_title)

        filter_values = {}
        for filter_obj in self.filters:
            filter_values[filter_obj.key] = filter_obj.render()

        return filter_values

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all filters to the dataframe."""
        filtered_df = df.copy()

        for filter_obj in self.filters:
            filtered_df = filter_obj.apply(filtered_df)

        return filtered_df

    def get_filter_by_key(self, key: str) -> Optional[BaseFilter]:
        """Get a filter by its key."""
        for filter_obj in self.filters:
            if filter_obj.key == key:
                return filter_obj
        return None

    def clear_filters(self):
        """Clear all filters."""
        self.filters.clear()
