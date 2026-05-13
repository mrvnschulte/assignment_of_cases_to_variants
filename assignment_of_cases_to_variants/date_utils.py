"""
Defines utility functions for using dates
"""

import datetime as dt

import numpy as np
from numpy.typing import ArrayLike


def days_since_reference(
    dates: list[dt.date], reference_date: dt.date
) -> np.ndarray:
    """
    Returns the days since reference date.

    Parameters
    ----------
    dates : list of datetime dates
        Dates for which to compute difference to reference date.
    reference_date : dt.date
        Reference date.

    Returns
    -------
    days_since_reference_date : np.ndarray
        For each date in dates, the number of days since the reference date.
    """
    days_since_reference_date = np.array(
        [(date - reference_date).days for date in dates]
    )

    return days_since_reference_date


def weeks_since_reference(
    dates: list[dt.date], reference_date: dt.date
) -> np.ndarray:
    """
    Returns the distance in ISO calendar weeks between each date and the
    reference date.

    All days within the same calendar week get the same value.

    Parameters
    ----------
    dates : list of datetime dates
        Dates for which to compute the calendar week distance.
    reference_date : dt.date
        Reference date.

    Returns
    -------
    weeks : np.ndarray
        For each date in dates, the number of calendar weeks since the
        calendar week of reference_date.
    """
    ref_monday = reference_date - dt.timedelta(days=reference_date.weekday())

    weeks = np.array(
        [
            (date - dt.timedelta(days=date.weekday()) - ref_monday).days // 7
            for date in dates
        ]
    )

    return weeks


def days_of_weeks(
    weeks: ArrayLike, reference_date: dt.date, day: str
) -> list[dt.date]:
    """
    Returns all dates that are the specified day in the weeks given.

    Parameters
    ----------
    weeks : array-like
        Array of integers, representing the weeks since reference date.
    reference_date : dt.date
        Reference date (can be any weekday).
    day : str
        Day of the week.

    Returns
    -------
    dates : list of datetime dates
        The dates of each specified day of the week for the given weeks.
    """
    day_codes = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    offset = (day_codes[day] - reference_date.weekday()) % 7

    non_duplicate_weeks = non_duplicate_list(weeks)
    dates = [
        reference_date + dt.timedelta(days=offset, weeks=int(i))
        for i in non_duplicate_weeks
    ]

    return dates


def non_duplicate_list(duplicate_list: list) -> list:
    """
    Returns the list without duplicates, preserving order.

    Parameters
    ----------
    duplicate_list : list
        List with potential duplicates.

    Returns
    -------
    result : list
        List with elements of duplicate_list, but without duplicates.
    """
    return list(dict.fromkeys(duplicate_list))


def date_to_calendar_week(date: dt.date) -> str:
    """
    Returns the string of the calendar week corresponding to date.

    Parameters
    ----------
    date : dt.date
        Input date.

    Returns
    -------
    string : str
        String of the form '{year}-KW{calendar_week}'.
    """
    year, week, _ = date.isocalendar()
    return f"{year}-KW{week:02d}"
