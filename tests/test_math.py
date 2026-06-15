import datetime as dt

import numpy as np
import pandas as pd
import pytest

from assignment_of_cases_to_variants.math import (
    logistic_function,
    daily_cases_emerging_variant,
    weekly_sum_of_cases,
    weekly_variant_fraction,
    moving_average,
    moving_average_per_variant,
)


class TestLogisticFunction:
    def test_midpoint_value(self):
        # At t=t_0, logistic should equal c/2
        result = logistic_function(0, alpha=1, t_0=0, c=1)
        assert pytest.approx(result, abs=1e-9) == 0.5

    def test_carrying_capacity(self):
        # Large t → approaches c
        result = logistic_function(1000, alpha=1, t_0=0, c=2)
        assert pytest.approx(result, abs=1e-3) == 2.0

    def test_approaches_zero(self):
        result = logistic_function(-1000, alpha=1, t_0=0, c=1)
        assert pytest.approx(result, abs=1e-3) == 0.0

    def test_array_input(self):
        t = np.array([-1, 0, 1])
        result = logistic_function(t, alpha=1, t_0=0, c=1)
        assert result.shape == (3,)
        assert result[1] == pytest.approx(0.5)

    def test_default_parameters(self):
        result = logistic_function(0)
        assert result == pytest.approx(0.5)


class TestDailyCasesEmergingVariant:
    def test_output_shape(self):
        t = np.arange(10)
        cases = np.ones(10) * 100
        result = daily_cases_emerging_variant(t, cases, alpha=0.1, t_0=5)
        assert result.shape == (10,)

    def test_integer_rounding(self):
        t = np.array([0, 1, 2])
        cases = np.array([100, 200, 300])
        result = daily_cases_emerging_variant(
            t, cases, alpha=0.5, t_0=1, integer=True
        )
        assert result.dtype in [np.int32, np.int64, int]

    def test_fraction_between_zero_and_cases(self):
        t = np.arange(10)
        cases = np.full(10, 100.0)
        result = daily_cases_emerging_variant(t, cases, alpha=0.2, t_0=5)
        assert np.all(result >= 0)
        assert np.all(result <= 100)

    def test_custom_carrying_capacity(self):
        t = np.array([1000.0])
        cases = np.array([100.0])
        result = daily_cases_emerging_variant(t, cases, alpha=1, t_0=0, c=0.5)
        assert result[0] == pytest.approx(50.0, abs=0.1)


class TestWeeklySumOfCases:
    def test_single_week(self):
        # 7 days in same ISO week
        monday = dt.date(2021, 1, 4)
        dates = [monday + dt.timedelta(days=i) for i in range(7)]
        cases = np.ones(7)
        result = weekly_sum_of_cases(dates, cases)
        assert result[0] == pytest.approx(7.0)

    def test_two_weeks(self):
        monday = dt.date(2021, 1, 4)
        dates = [monday + dt.timedelta(days=i) for i in range(14)]
        cases = np.ones(14) * 2
        result = weekly_sum_of_cases(dates, cases)
        assert len(result) == 2
        np.testing.assert_allclose(result, [14.0, 14.0])

    def test_returns_ndarray(self):
        dates = [dt.date(2021, 1, 4)]
        result = weekly_sum_of_cases(dates, [5])
        assert isinstance(result, np.ndarray)


class TestWeeklyVariantFraction:
    def test_fraction_is_one_when_all_cases_are_variant(self):
        monday = dt.date(2021, 1, 4)
        dates = [monday + dt.timedelta(days=i) for i in range(7)]
        cases = np.ones(7) * 100
        result = weekly_variant_fraction(dates, cases, cases)
        assert result[0] == pytest.approx(1.0)

    def test_fraction_is_half(self):
        monday = dt.date(2021, 1, 4)
        dates = [monday + dt.timedelta(days=i) for i in range(7)]
        total = np.ones(7) * 100
        variant = np.ones(7) * 50
        result = weekly_variant_fraction(dates, variant, total)
        assert result[0] == pytest.approx(0.5)

    def test_output_length_matches_weeks(self):
        monday = dt.date(2021, 1, 4)
        dates = [monday + dt.timedelta(days=i) for i in range(14)]
        cases = np.ones(14) * 100
        result = weekly_variant_fraction(dates, cases, cases)
        assert len(result) == 2


class TestMovingAverage:
    def test_constant_series(self):
        s = pd.Series([5.0] * 10)
        result = moving_average(s, window=3)
        assert result.iloc[-1] == pytest.approx(5.0)

    def test_default_window_is_7(self):
        s = pd.Series(range(10), dtype=float)
        result = moving_average(s)
        assert len(result) == 10

    def test_window_1_returns_same(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = moving_average(s, window=1)
        pd.testing.assert_series_equal(result, s)


class TestMovingAveragePerVariant:
    def _make_df(self):
        monday = dt.date(2021, 1, 4)
        dates = [monday + dt.timedelta(days=i) for i in range(14)]
        return pd.DataFrame(
            {
                "Date": dates,
                "Other variants": np.ones(14) * 50,
                "Alpha": np.ones(14) * 30,
                "Delta": np.ones(14) * 20,
            }
        )

    def test_output_contains_date_column(self):
        df = self._make_df()
        result = moving_average_per_variant(["Alpha", "Delta"], df, window=7)
        assert "Date" in result.columns

    def test_output_contains_all_variant_columns(self):
        df = self._make_df()
        result = moving_average_per_variant(["Alpha", "Delta"], df, window=7)
        assert "Alpha" in result.columns
        assert "Delta" in result.columns
        assert "Other variants" in result.columns

    def test_constant_data_unchanged_by_average(self):
        df = self._make_df()
        result = moving_average_per_variant(["Alpha"], df, window=3)
        assert result["Alpha"].iloc[-1] == pytest.approx(30.0)
