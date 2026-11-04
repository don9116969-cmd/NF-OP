"""
NSE Market Calendar & Holiday Utility.
Tracks official National Stock Exchange of India (NSE) trading holidays
and provides automated market-state verification.
"""
import datetime
from typing import Optional, Dict

NSE_HOLIDAYS: Dict[str, str] = {
    # 2025 Trading Holidays
    "2025-01-26": "Republic Day",
    "2025-02-26": "Mahashivratri",
    "2025-03-14": "Holi",
    "2025-03-31": "Id-Ul-Fitr (Ramzan Id)",
    "2025-04-10": "Mahavir Jayanti",
    "2025-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2025-04-18": "Good Friday",
    "2025-05-01": "Maharashtra Day",
    "2025-06-07": "Bakri Id",
    "2025-07-06": "Muharram",
    "2025-08-15": "Independence Day",
    "2025-08-27": "Ganesh Chaturthi",
    "2025-10-02": "Mahatma Gandhi Jayanti",
    "2025-10-21": "Dussehra",
    "2025-10-22": "Diwali Balipratipada",
    "2025-11-05": "Gurunanak Jayanti",
    "2025-12-25": "Christmas",

    # 2026 Trading Holidays
    "2026-01-26": "Republic Day",
    "2026-02-17": "Mahashivratri",
    "2026-03-03": "Holi",
    "2026-03-20": "Id-Ul-Fitr (Ramzan Id)",
    "2026-03-26": "Ram Navami",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-28": "Bakri Id",
    "2026-06-26": "Muharram",
    "2026-08-15": "Independence Day",
    "2026-08-26": "Milad-un-Nabi",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-09": "Diwali Laxmi Pujan / Govardhan Puja",
    "2026-11-10": "Diwali-Balipratipada",
    "2026-11-24": "Gurunanak Jayanti",
    "2026-12-25": "Christmas",
}

def get_nse_holiday_name(d: Optional[datetime.date] = None) -> Optional[str]:
    """Returns holiday name if the date is an official NSE trading holiday, else None."""
    if d is None:
        d = datetime.datetime.now().date()
    d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
    return NSE_HOLIDAYS.get(d_str)

def is_nse_holiday(d: Optional[datetime.date] = None) -> bool:
    """Returns True if the date is an official NSE trading holiday."""
    return get_nse_holiday_name(d) is not None

def is_trading_day(d: Optional[datetime.date] = None) -> bool:
    """Returns True if the date is a regular weekday trading session (Mon-Fri and not a holiday)."""
    if d is None:
        d = datetime.datetime.now().date()
    if hasattr(d, "weekday"):
        if d.weekday() in [5, 6]:  # Saturday or Sunday
            return False
    return not is_nse_holiday(d)

def get_next_trading_day(d: Optional[datetime.date] = None) -> datetime.date:
    """Computes the next valid NSE market session date."""
    if d is None:
        d = datetime.datetime.now().date()
    curr = d + datetime.timedelta(days=1)
    while not is_trading_day(curr):
        curr += datetime.timedelta(days=1)
    return curr
