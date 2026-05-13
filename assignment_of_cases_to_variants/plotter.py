import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from .date_utils import (
    days_of_weeks,
    days_since_reference,
    weeks_since_reference,
)
from .math import (
    daily_cases_emerging_variant,
    weekly_variant_fraction,
)


def plot_variant_transition(
    variant: str,
    dates: list[dt.date],
    case_data: ArrayLike,
    fraction_data: ArrayLike,
    alpha: float,
    t_0: float,
    c: float,
    reference_date: dt.date,
    folder: str | Path = ".",
    fileformat: str = "jpg",
    title: bool = False,
    dpi: int = 500,
):
    """
    Plots the daily case split and weekly fraction fit for a variant
    transition.

    Parameters
    ----------
    variant : str
        Name of the emerging variant.
    dates : list of datetime dates
        The dates of the transition period.
    case_data : array-like
        Daily overall cases for dates.
    fraction_data : array-like
        Measured weekly variant fractions.
    alpha : float
        Fitted logistic growth rate.
    t_0 : float
        Fitted logistic midpoint (days since reference_date).
    c : float
        Fitted carrying capacity.
    reference_date : dt.date
        Reference date for computing days.
    folder : str or Path, optional
        Folder to save plots. The default is '.'.
    fileformat : str, optional
        File format for plots. The default is 'jpg'.
    title : bool, optional
        If True, plots include a title. The default is False.
    dpi : int, optional
        Resolution for plots. The default is 500.

    Returns
    -------
    None
    """
    folder = Path(folder)
    t = days_since_reference(dates, reference_date)
    weeks = weeks_since_reference(dates, reference_date)
    mondays = days_of_weeks(weeks, reference_date, "Monday")
    thursdays = days_of_weeks(weeks, reference_date, "Thursday")
    calendar_weeks = [
        f"{day.isocalendar()[0]:d}-{day.isocalendar()[1]:02d}"
        for day in thursdays
    ]
    variant_filename = variant.replace(" ", "_").replace(".", "_")

    emerging_variant_cases = daily_cases_emerging_variant(
        t, case_data, alpha, t_0, c=c, integer=True
    )
    other_variant_cases = np.asarray(case_data) - emerging_variant_cases

    weekly_fraction = weekly_variant_fraction(
        dates, emerging_variant_cases, case_data
    )

    # Plot 1: Daily cases
    fig, ax = plt.subplots()
    ax.plot(dates, emerging_variant_cases, "-o", label=variant)
    ax.plot(dates, other_variant_cases, "-o", label="Other variants")
    ax.set_xlabel("Date")
    ax.set_ylabel("New infections")
    ax.legend()
    ax.grid()
    if title:
        ax.set_title("Reported new infections per variant and day")
    ax.set_xticks(mondays)
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(
        folder
        / f"Daily_new_infections_{variant_filename}_variant.{fileformat}",
        dpi=dpi,
    )
    plt.show()

    # Plot 2: Weekly fraction
    fig, ax = plt.subplots()
    ax.plot(thursdays, weekly_fraction, "o", label="Computed data")
    ax.plot(thursdays, fraction_data, "x", label="Measured data")
    ax.set_xlabel("Week")
    ax.set_ylabel("Fraction of cases")
    ax.legend()
    ax.grid()
    if title:
        ax.set_title(f"Weekly case fraction of {variant} variant")
    ax.set_xticks(thursdays)
    ax.set_xticklabels(calendar_weeks, rotation=90)
    fig.tight_layout()
    fig.savefig(
        folder
        / f"Weekly_case_fraction_{variant_filename}_variant.{fileformat}",
        dpi=dpi,
    )
    plt.show()


def plot_cases_per_variant(
    variants: list[str],
    cases_per_variant: pd.DataFrame,
    folder: str | Path = ".",
    fileformat: str = "jpg",
    title: bool = False,
    dpi: int = 500,
):
    """
    Plot daily cases per variant as given by cases_per_variant.

    Parameters
    ----------
    variants : list[str]
        List of variants.
    cases_per_variant : pd.DataFrame
        Data frame with daily cases per variant.
    folder : str or Path, optional
        Folder to save the plot. The default is '.'.
    fileformat : str, optional
        File format for plot. The default is 'jpg'.
    title : bool, optional
        If True, plot includes a title. The default is False.
    dpi : int, optional
        Resolution for plot. The default is 500.

    Returns
    -------
    None.
    """
    plt.plot(
        cases_per_variant["Date"],
        cases_per_variant["Other variants"],
        "-o",
        label="Wild Type",
    )
    for variant in reversed(variants):
        plt.plot(
            cases_per_variant["Date"],
            cases_per_variant[variant],
            "-o",
            label=variant,
        )
    plt.grid()
    plt.legend()
    if title:
        plt.title("Reported new infections per variant and day")
    plt.xlabel("Date")
    plt.ylabel("New infections")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{folder}/Daily_new_infections.{fileformat}", dpi=dpi)
    plt.show()


def plot_fractions_all_variants(
    variants: list[str],
    cases_per_variant: pd.DataFrame,
    folder: str | Path = ".",
    fileformat: str = "jpg",
    title: bool = False,
    dpi: int = 500,
):
    """
    Plot weekly fractions per variant given by cases_per_variant as stackplot.

    Parameters
    ----------
    variants : list[str]
        List of variants.
    cases_per_variant : pd.DataFrame
        Data frame with daily cases per variant.
    folder : str or Path, optional
        Folder to save the plot. The default is '.'.
    fileformat : str, optional
        File format for plot. The default is 'jpg'.
    title : bool, optional
        If True, plot includes a title. The default is False.
    dpi : int, optional
        Resolution for plot. The default is 500.

    Returns
    -------
    None.
    """
    fractions = []
    labels = []
    weeks = weeks_since_reference(
        cases_per_variant["Date"], cases_per_variant["Date"][0]
    )
    thursdays = days_of_weeks(weeks, cases_per_variant["Date"][0], "Thursday")

    total_cases = cases_per_variant["Other variants"] + sum(
        [cases_per_variant[variant] for variant in variants]
    )
    variant_cases = cases_per_variant["Other variants"]
    variant_fraction = weekly_variant_fraction(
        cases_per_variant["Date"], variant_cases, total_cases
    )
    fractions.append(variant_fraction)
    labels.append("Wild Type")

    for variant in reversed(variants):
        variant_cases = cases_per_variant[variant]
        variant_fraction = weekly_variant_fraction(
            cases_per_variant["Date"], variant_cases, total_cases
        )
        fractions.append(variant_fraction)
        labels.append(variant)

    plt.stackplot(thursdays, fractions, labels=labels)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if title:
        plt.title("Weekly variant fraction")
    plt.xlabel("Week")
    plt.ylabel("Variant fraction")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{folder}/Weekly_variant_fractions.{fileformat}", dpi=dpi)
    plt.show()


def plot_moving_average_per_variant(
    variants: list[str],
    moving_average_per_variant: pd.DataFrame,
    folder: str | Path = ".",
    fileformat: str = "jpg",
    title: bool = False,
    dpi: int = 500,
):
    """
    Plot weekly fractions per variant given by cases_per_variant as stackplot.

    Parameters
    ----------
    variants : list[str]
        List of variants.
    moving_average_per_variant : pd.DataFrame
        Data frame with moving average of cases per variant.
    folder : str or Path, optional
        Folder to save the plot. The default is '.'.
    fileformat : str, optional
        File format for plot. The default is 'jpg'.
    title : bool, optional
        If True, plot includes a title. The default is False.
    dpi : int, optional
        Resolution for plot. The default is 500.

    Returns
    -------
    None.
    """
    plt.plot(
        moving_average_per_variant["Date"],
        moving_average_per_variant["Other variants"],
        "o",
        label="Wild Type",
    )
    for variant in reversed(variants):
        plt.plot(
            moving_average_per_variant["Date"],
            moving_average_per_variant[variant],
            "o",
            label=variant,
        )
    plt.grid()
    plt.legend()
    if title:
        plt.title(
            "Reported new infections (7-day moving average) per variant and day"
        )
    plt.xlabel("Date")
    plt.ylabel("New infections (7-day moving average)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        f"{folder}/Daily_new_infections_moving_average.{fileformat}", dpi=dpi
    )
    plt.show()
