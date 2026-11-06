"""
Centralized Malaysian Time (UTC+8) and Date Formatting Utilities for SAFELABS.
Standard timezone: Asia/Kuala_Lumpur (UTC+8)
Standard date format: dd/mm/yy (e.g. 05/11/26)
Standard datetime format: dd/mm/yy HH:MM:SS or dd/mm/yy HH:MM
"""
from datetime import datetime, timezone, timedelta

# Malaysian Standard Time is strictly UTC+8 with no daylight saving time.
MYT_TZ = timezone(timedelta(hours=8))


def get_myt_now() -> datetime:
    """
    Returns current naive datetime in Malaysian Time (UTC+8).
    Naive datetime is returned for seamless compatibility with SQLite and existing SQLAlchemy DateTime columns.
    """
    return datetime.now(timezone.utc).astimezone(MYT_TZ).replace(tzinfo=None)


def to_myt(dt: datetime = None) -> datetime:
    """
    Converts a datetime object to naive Malaysian Time (UTC+8).
    If dt is None, returns current Malaysian time.
    """
    if dt is None:
        return get_myt_now()
    if dt.tzinfo is not None:
        return dt.astimezone(MYT_TZ).replace(tzinfo=None)
    return dt


def format_date(dt, default='-') -> str:
    """Format date strictly as dd/mm/yy (e.g., 05/11/26)."""
    if not dt:
        return default
    return dt.strftime('%d/%m/%y')


def format_time(dt, include_seconds=True, default='-') -> str:
    """Format time as HH:MM:SS or HH:MM."""
    if not dt:
        return default
    fmt = '%H:%M:%S' if include_seconds else '%H:%M'
    return dt.strftime(fmt)


def format_datetime(dt, include_seconds=True, default='-') -> str:
    """Format datetime as dd/mm/yy HH:MM:SS or dd/mm/yy HH:MM."""
    if not dt:
        return default
    fmt = '%d/%m/%y %H:%M:%S' if include_seconds else '%d/%m/%y %H:%M'
    return dt.strftime(fmt)

