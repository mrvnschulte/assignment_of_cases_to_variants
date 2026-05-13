import datetime as dt
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.optimize import minimize

from .math import daily_cases_emerging_variant, weekly_variant_fraction
from .date_utils import days_since_reference
from .load_data import load_case_data


def fit_logistic_variant(
    dates: list[dt.date],
    case_data: ArrayLike,
    fraction_data: ArrayLike,
    reference_date: dt.date,
    fit_carrying_capacity: bool = False,
) -> tuple[float, float, float]:
    """
    Fits a logistic function to the weekly variant fraction data.

    Parameters
    ----------
    dates : list of dt.date
        The dates for which case data is given. Has to be ordered
        chronologically.
    case_data : array-like
        The overall daily cases. Has to be the same size as dates.
    fraction_data : array-like
        The measured weekly variant fraction for each calendar week covered
        by dates.
    reference_date : dt.date
        Reference date for computing days (typically start_date).
    fit_carrying_capacity : bool, optional
        If True, carrying capacity of logistic function is fitted. The default
        is False.

    Returns
    -------
    alpha : float
        Fitted logistic growth rate.
    t_0 : float
        Fitted midpoint (days since reference_date).
    c : float
        Fitted carrying capacity. If not fitted, 1 is returned as the
        corresponding default value.
    residual : float
        Residual norm of the fit.
    """
    case_data = np.asarray(case_data)
    fraction_data = np.asarray(fraction_data)
    t = days_since_reference(dates, reference_date)

    def objective(param):
        if fit_carrying_capacity:
            daily_variant_cases = daily_cases_emerging_variant(
                t, case_data, param[0], param[1], param[2]
            )
        else:
            daily_variant_cases = daily_cases_emerging_variant(
                t, case_data, param[0], param[1]
            )
        model_fraction = weekly_variant_fraction(
            dates, daily_variant_cases, case_data
        )
        return np.linalg.norm(fraction_data - model_fraction)

    initial_midpoint = t[-1] / 2
    if fit_carrying_capacity:
        x0 = [0.1, initial_midpoint, 1]
        bounds = [(0.001, 1.0), (t[0], t[-1]), (0.0, 1.0)]
    else:
        x0 = [0.1, initial_midpoint]
        bounds = [(0.001, 1.0), (t[0], t[-1])]
    sol = minimize(
        objective,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
    )

    if fit_carrying_capacity:
        alpha, t_0, c = sol.x
    else:
        alpha, t_0 = sol.x
        c = 1
    return alpha, t_0, c, sol.fun


def split_cases_by_variant(
    variant: str,
    alpha: float,
    t_0: float,
    c: float,
    reference_date: dt.date,
    data_frame: pd.DataFrame,
    case_file: str | Path,
):
    """
    Splits daily case data between the emerging variant and others using
    fitted logistic parameters. Modifies data_frame in-place.

    Parameters
    ----------
    variant : str
        Name of the emerging variant (used as column name).
    alpha : float
        Logistic growth rate.
    t_0 : float
        Logistic midpoint (days since reference_date).
    reference_date : dt.date
        Reference date for computing days.
    data_frame : pd.DataFrame
        Data frame with a 'Date' column. Modified in-place.
    case_file : str or Path
        Path to the case data CSV file.

    Returns
    -------
    None.

    Side Effects
    ------------
    - Inserts column `variant` into `data_frame`.
    - Inserts or updates column 'Other variants' in `data_frame`.
    """
    t_all = days_since_reference(data_frame["Date"], reference_date)

    if "Other variants" in data_frame:
        all_emerging_variant_cases = daily_cases_emerging_variant(
            t_all,
            data_frame["Other variants"],
            alpha,
            t_0,
            c=c,
            integer=True,
        )
        all_other_variant_cases = (
            np.asarray(data_frame["Other variants"])
            - all_emerging_variant_cases
        )
        data_frame["Other variants"] = all_other_variant_cases
    else:
        all_case_data = load_case_data(case_file, data_frame["Date"])
        all_emerging_variant_cases = daily_cases_emerging_variant(
            t_all,
            all_case_data,
            alpha,
            t_0,
            c=c,
            integer=True,
        )
        all_other_variant_cases = (
            np.asarray(all_case_data) - all_emerging_variant_cases
        )
        data_frame.insert(1, "Other variants", all_other_variant_cases)

    data_frame.insert(2, variant, all_emerging_variant_cases)
