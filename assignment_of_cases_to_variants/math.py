"""
Defines the mathematical functions needed to compute variant fractions, split
of overall cases according to a logistic function and moving average.
"""

import datetime as dt

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd


def logistic_function(
    t: float | ArrayLike,
    alpha: float | ArrayLike = 1,
    t_0: float | ArrayLike = 0,
    c: float | ArrayLike = 1,
) -> float | np.ndarray:
    """
    Computes the logistic function with parameters alpha, t_0 and c for all
    elements in t.

    Parameters
    ----------
    t : float or array-like
        Input data for which to evaluate the logistic function.
    alpha : float or array-like, optional
        Logistic growth rate. If array-like, has to be the same size as t. The
        default is 1.
    t_0 : float or array-like, optional
        The function's midpoint. If array-like, has to be the same size as t.
        The default is 0.
    c : float or array-like, optional
        Carrying capacity. If array-like, has to be the same size as t. The
        default is 1.

    Returns
    -------
    y : float or np.ndarray
        The logistic function at t with parameters c, alpha and t_0.
    """
    t = np.asarray(t)
    c = np.asarray(c)
    alpha = np.asarray(alpha)
    t_0 = np.asarray(t_0)

    y = c / (1 + np.exp(-alpha * (t - t_0)))

    return y


def daily_cases_emerging_variant(
    t: ArrayLike,
    case_data: ArrayLike,
    alpha: float | ArrayLike,
    t_0: float | ArrayLike,
    c: float | ArrayLike = 1,
    integer: bool = False,
) -> np.ndarray:
    """
    Returns the daily number of cases according to a logistic splitting with
    parameters alpha, t_0 and c.

    Parameters
    ----------
    t : array-like
        Days since reference date.
    case_data : array-like
        The overall cases per day. Has to be the same size as t.
    alpha : float or array-like
        Logistic growth rate. If array-like, has to be the same size as t.
    t_0 : float or array-like
        The function's midpoint. If array-like, has to be the same size as t.
    c : float or array-like, optional
        Carrying capacity. If array-like, has to be the same size as t. The
        default is 1.
    integer : bool, optional
        If True, daily cases are rounded to the next integer. The default is
        False.

    Returns
    -------
    daily_variant_cases : np.ndarray
        For each entry in t, the number of cases for the emerging variant
        according to a logistic growth function with parameters alpha and t_0.
    """
    case_data = np.asarray(case_data)

    daily_fractions = logistic_function(t, alpha=alpha, t_0=t_0, c=c)

    if integer:
        daily_variant_cases = np.round(daily_fractions * case_data).astype(
            "int"
        )
    else:
        daily_variant_cases = daily_fractions * case_data

    return daily_variant_cases


def weekly_sum_of_cases(
    dates: list[dt.date], daily_cases: ArrayLike
) -> np.ndarray:
    """
    Sums up the daily cases by ISO calendar week.

    Parameters
    ----------
    dates : list of dt.date
        All dates, for which daily cases are given. Has to be ordered
        chronologically.
    daily_cases : array-like
        Cases for each date in dates. Has to be the same size as dates.

    Returns
    -------
    weekly_cases : np.ndarray
        For each calendar week with at least one date in dates, the weekly
        sum of cases.
    """
    daily_cases = np.asarray(daily_cases, dtype=float)

    week_keys = [
        (date.isocalendar()[0], date.isocalendar()[1]) for date in dates
    ]
    df = pd.DataFrame({"week": week_keys, "cases": daily_cases})
    weekly_cases = df.groupby("week", sort=True)["cases"].sum().values

    return weekly_cases


def weekly_variant_fraction(
    dates: list[dt.date], variant_cases: ArrayLike, total_cases: ArrayLike
) -> np.ndarray:
    """
    Returns the weekly variant fraction for all weeks with dates in dates.

    Parameters
    ----------
    dates : list of dt.date
        All dates, for which case data is given.
    variant_cases : array-like
        Daily cases of the respective variant.
    total_cases : array-like
        Overall daily cases.

    Returns
    -------
    weekly_fraction : np.ndarray
        Weekly fraction of variant cases among all cases.
    """
    weekly_variant_cases = weekly_sum_of_cases(dates, variant_cases)
    weekly_total_cases = weekly_sum_of_cases(dates, total_cases)
    weekly_fraction = weekly_variant_cases / weekly_total_cases

    return weekly_fraction


def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Computes the moving average of series over given window.

    Parameters
    ----------
    series : pd.Series
        Series to compute moving average of.
    window : int, optional
        Window of moving average. The default is 7.

    Returns
    -------
    average_series : pd.Series
        Moving average of series over given window.
    """
    return series.rolling(window=window, min_periods=1).mean()


def moving_average_per_variant(
    variants: list[str], cases_per_variant: pd.DataFrame, window: int = 7
) -> pd.DataFrame:
    """
    Computes and returns the moving average for cases of all variants over
    given window.

    Parameters
    ----------
    variants : list[str]
        List of variants.
    cases_per_variant : pd.DataFrame
        Data frame with daily cases split by variant.
    window : int
        Window of moving average. The default is 7.

    Returns
    -------
    moving_average_df : pd.DataFrame
        Data frame with moving average of cases over given window split by
        variant.
    """
    moving_average_df = pd.DataFrame()
    moving_average_df["Date"] = cases_per_variant["Date"].iloc[window - 1 :]
    moving_average_df["Other variants"] = moving_average(
        cases_per_variant["Other variants"], window=window
    )
    for variant in reversed(variants):
        moving_average_df[variant] = moving_average(
            cases_per_variant[variant], window=window
        )

    return moving_average_df
