import datetime as dt
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from assignment_of_cases_to_variants.load_data import (
    load_case_data,
    load_weekly_variant_fraction,
)


def make_case_csv_df():
    return pd.DataFrame(
        {
            "Altersgruppe": ["00+", "00+"],
            "Meldedatum": ["2021-01-04", "2021-01-05"],
            "Faelle_neu": [100, 200],
        }
    )


def make_fraction_xlsx_df():
    return pd.DataFrame(
        {
            "KW": ["2021-KW01", "2021-KW02", "2021-KW03"],
            "Alpha_Anteil": [10.0, 20.0, 30.0],
        }
    )


class TestLoadCaseData:
    @patch(
        "assignment_of_cases_to_variants.load_data._load_case_csv",
        return_value=make_case_csv_df(),
    )
    def test_returns_correct_values(self, mock_csv):
        dates = [dt.date(2021, 1, 4), dt.date(2021, 1, 5)]
        result = load_case_data("dummy.csv", dates)
        np.testing.assert_array_equal(result, [100, 200])

    @patch(
        "assignment_of_cases_to_variants.load_data._load_case_csv",
        return_value=make_case_csv_df(),
    )
    def test_returns_ndarray(self, mock_csv):
        dates = [dt.date(2021, 1, 4)]
        result = load_case_data("dummy.csv", dates)
        assert isinstance(result, np.ndarray)

    @patch(
        "assignment_of_cases_to_variants.load_data._load_case_csv",
        return_value=make_case_csv_df(),
    )
    def test_missing_date_returns_empty(self, mock_csv):
        dates = [dt.date(2020, 1, 1)]
        result = load_case_data("dummy.csv", dates)
        assert len(result) == 0


class TestLoadWeeklyVariantFraction:
    @patch(
        "assignment_of_cases_to_variants.load_data._load_fraction_xlsx",
        return_value=make_fraction_xlsx_df(),
    )
    def test_returns_fractions_as_decimal(self, mock_xlsx):
        result = load_weekly_variant_fraction(
            "dummy.xlsx",
            "Alpha",
            dt.date(2021, 1, 4),
            dt.date(2021, 1, 17),
        )
        np.testing.assert_allclose(result, [0.1, 0.2])

    @patch(
        "assignment_of_cases_to_variants.load_data._load_fraction_xlsx",
        return_value=make_fraction_xlsx_df(),
    )
    def test_returns_ndarray(self, mock_xlsx):
        result = load_weekly_variant_fraction(
            "dummy.xlsx",
            "Alpha",
            dt.date(2021, 1, 4),
            dt.date(2021, 1, 10),
        )
        assert isinstance(result, np.ndarray)

    @patch(
        "assignment_of_cases_to_variants.load_data._load_fraction_xlsx",
        return_value=make_fraction_xlsx_df(),
    )
    def test_fractions_between_0_and_1(self, mock_xlsx):
        result = load_weekly_variant_fraction(
            "dummy.xlsx",
            "Alpha",
            dt.date(2021, 1, 4),
            dt.date(2021, 1, 21),
        )
        assert np.all(result >= 0) and np.all(result <= 1)
