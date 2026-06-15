"""
Executes the case assignment to variants for COVID-19 cases in 2021 and 2022
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from assignment_of_cases_to_variants import (
    fit_logistic_variant,
    load_case_data,
    load_weekly_variant_fraction,
    moving_average_per_variant,
    plot_cases_per_variant,
    plot_fractions_all_variants,
    plot_moving_average_per_variant,
    plot_variant_transition,
    split_cases_by_variant,
)


def compute_emerging_variant(
    variant: str,
    start_date: dt.date,
    end_date: dt.date,
    data_frame: pd.DataFrame,
    case_file: (
        str | Path
    ) = "../data/Infection Data/COVID-19-Cases_7-Day_Incidence_Germany.csv",
    fraction_file: (
        str | Path
    ) = "../data/Variant Fraction Data/VOC_VOI_Table.xlsx",
    fit_carrying_capacity: bool = False,
    verbose: bool = False,
    plot: bool = False,
    folder: str | Path = ".",
    fileformat: str = "jpg",
    title: bool = False,
    dpi: int = 500,
):
    """
    Executes fitting, splitting, and plotting for one emerging variant.

    Fits a logistic function to weekly variant fraction data, splits daily
    cases accordingly, and optionally plots the results.

    Parameters
    ----------
    variant : str
        Name of the emerging variant.
    start_date : dt.date
        First date for interpolation.
    end_date : dt.date
        Last date for interpolation.
    data_frame : pd.DataFrame
        Data frame with a 'Date' column. Modified in-place.
    case_file : str or Path, optional
        Path to case data CSV.
    fraction_file : str or Path, optional
        Path to variant fraction Excel file.
    fit_carrying_capacity : bool, optional
        If True, carrying capacity of logistic function is fitted. The default
        is False.
    verbose : bool, optional
        If True, prints fit results. The default is False.
    plot : bool, optional
        If True, plots the transition. The default is False.
    folder : str or Path, optional
        Folder to save plots. The default is '.'.
    fileformat : str, optional
        File format for plots. The default is 'jpg'.
    title : bool, optional
        If True, plots include a title. The default is False.
    dpi : int, optional
        Resolution for plots. The default is 500.
    """
    dates = pd.date_range(start_date, end_date).to_pydatetime()
    dates = [date.date() for date in dates]

    case_data = load_case_data(case_file, dates)
    fraction_data = load_weekly_variant_fraction(
        fraction_file, variant, start_date, end_date
    )

    # 1. Fit
    alpha, t_0, c, residual = fit_logistic_variant(
        dates,
        case_data,
        fraction_data,
        reference_date=start_date,
        fit_carrying_capacity=False,
    )
    if verbose:
        print(
            f"[{variant}] alpha={alpha:.4f}, t_0={t_0:.1f}, c={c:.3f} "
            f"residual={residual:.6f}"
        )

    # 2. Split
    split_cases_by_variant(
        variant,
        alpha,
        t_0,
        c,
        reference_date=start_date,
        data_frame=data_frame,
        case_file=case_file,
    )

    # 3. Plot
    if plot:
        plot_variant_transition(
            variant,
            dates,
            case_data,
            fraction_data,
            alpha,
            t_0,
            reference_date=start_date,
            c=c,
            folder=folder,
            fileformat=fileformat,
            title=title,
            dpi=dpi,
        )


if __name__ == "__main__":
    variants = [
        "Omicron BA.5",
        "Omicron BA.2",
        "Omicron BA.1",
        "Delta",
        "Alpha",
    ]
    start_dates = [
        dt.date(2022, 4, 25),
        dt.date(2022, 1, 17),
        dt.date(2021, 11, 15),
        dt.date(2021, 4, 19),
        dt.date(2021, 1, 4),
    ]
    end_dates = [
        dt.date(2022, 7, 24),
        dt.date(2022, 4, 17),
        dt.date(2022, 1, 16),
        dt.date(2021, 8, 8),
        dt.date(2021, 4, 4),
    ]
    start_date_overall = dt.date(2021, 1, 1)
    end_date_overall = dt.date(2022, 9, 1)

    fit_carrying_capacity = False
    plot_folder = "../plots"
    plot_title = False
    plot_single_transition = True
    fileformat = "jpg"
    dpi = 500
    verbose = True
    store_as = "csv"  # Can be 'csv' or 'xlsx'

    case_file = "../data/infection_data/covid-19_cases_germany.csv"
    fraction_file = "../data/variant_fraction_data/voc_voi_table.xlsx"

    # Create data frame
    all_dates = pd.date_range(
        start_date_overall, end_date_overall
    ).to_pydatetime()
    all_dates = [date.date() for date in all_dates]
    cases_per_variant = pd.DataFrame(all_dates, columns=["Date"])

    # Compute all variants
    for i in range(len(variants)):
        compute_emerging_variant(
            variants[i],
            start_dates[i],
            end_dates[i],
            cases_per_variant,
            case_file=case_file,
            fraction_file=fraction_file,
            fit_carrying_capacity=fit_carrying_capacity,
            verbose=verbose,
            plot=plot_single_transition,
            folder=plot_folder,
            fileformat=fileformat,
            title=plot_title,
            dpi=dpi,
        )
    moving_average_per_variant = moving_average_per_variant(
        variants, cases_per_variant
    )

    # Summary plots
    plot_cases_per_variant(
        variants,
        cases_per_variant,
        folder=plot_folder,
        fileformat=fileformat,
        title=plot_title,
        dpi=dpi,
    )
    plot_fractions_all_variants(
        variants,
        cases_per_variant,
        folder=plot_folder,
        fileformat=fileformat,
        title=plot_title,
        dpi=dpi,
    )
    plot_moving_average_per_variant(
        variants,
        moving_average_per_variant,
        folder=plot_folder,
        fileformat=fileformat,
        title=plot_title,
        dpi=dpi,
    )

    # Export
    if store_as == "csv":
        cases_per_variant.to_csv(
            "../data/daily_cases_per_variant.csv", index=False
        )
        moving_average_per_variant.to_csv(
            "../data/moving_average_per_variant.csv", index=False
        )
    elif store_as == "xlsx":
        cases_per_variant.to_excel(
            "../data/daily_cases_per_variant.xlsx", index=False
        )
        moving_average_per_variant.to_excel(
            "../data/moving_average_per_variant.xlsx", index=False
        )
