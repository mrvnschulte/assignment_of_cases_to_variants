"""
Functions to load case and variant fraction data from files
"""

import datetime as dt
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .date_utils import date_to_calendar_week


@lru_cache(maxsize=4)
def _load_case_csv(filename: str) -> pd.DataFrame:
    """
    Loads and caches the case data CSV.

    Parameters
    ----------
    filename : str or Path
        Filename, where the case data is stored.

    Returns
    -------
    data : pd.DataFrame
        Case data csv as pandas data frame.
    """
    data = pd.read_csv(filename)
    data = data[data["Altersgruppe"] == "00+"]
    return data


def load_case_data(filename: str | Path, dates: list[dt.date]) -> np.ndarray:
    """
    Loads the case data specified in filename at the given dates.

    Parameters
    ----------
    filename : str or Path
        Filename, where the case data is stored.
    dates : list of datetime dates
        List of dates which are extracted.

    Returns
    -------
    case_data : np.ndarray
        Number of cases at dates according to specified file.
    """
    dates_as_strings = [f"{date:%Y-%m-%d}" for date in dates]
    data = _load_case_csv(str(filename))
    case_data = data.loc[
        data["Meldedatum"].isin(dates_as_strings), "Faelle_neu"
    ].values

    return case_data


@lru_cache(maxsize=4)
def _load_fraction_xlsx(filename: str) -> pd.DataFrame:
    """
    Loads and caches the variant fraction data XLSX.

    Parameters
    ----------
    filename : str or Path
        Filename, where the variant fraction data is stored.

    Returns
    -------
    frac_data : pd.DataFrame
        Variant fraction data xlsx as pandas data frame.
    """
    frac_data = pd.read_excel(filename)
    frac_data = frac_data.iloc[:-1]
    return frac_data


def load_weekly_variant_fraction(
    filename: str | Path,
    variant: str,
    start_date: dt.date,
    end_date: dt.date,
) -> np.ndarray:
    """
    Loads the weekly variant fraction data specified in filename for weeks
    between, and including, start_date and end_date.

    Parameters
    ----------
    filename : str or Path
        Filename, where the variant fraction data is stored.
    variant : str
        Name of the variant.
    start_date : dt.date
        Start date to load data for.
    end_date : dt.date
        End date to load data for.

    Returns
    -------
    fraction_data : np.ndarray
        Weekly variant fractions among all cases for weeks between start_date
        and end_date.
    """
    frac_data = _load_fraction_xlsx(filename)
    variant_table = variant.replace(" ", "_").replace("c", "k")
    matching_cols = [
        col
        for col in frac_data.columns
        if variant_table in col and "Anteil" in col
    ]
    start_week = date_to_calendar_week(start_date)
    end_week = date_to_calendar_week(end_date)
    mask = (frac_data["KW"] >= start_week) & (frac_data["KW"] <= end_week)
    fraction_data = frac_data.loc[mask, matching_cols[0]].tolist()
    fraction_data = np.array([frac / 100 for frac in fraction_data])

    return fraction_data
