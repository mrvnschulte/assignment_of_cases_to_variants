import datetime as dt
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from assignment_of_cases_to_variants.core import (
    fit_logistic_variant,
    split_cases_by_variant,
)


REF = dt.date(2021, 1, 4)


def make_synthetic_data(alpha=0.2, t_0=30, n=60):
    """Generates synthetic case + fraction data from known parameters."""
    dates = [REF + dt.timedelta(days=i) for i in range(n)]
    case_data = np.full(n, 1000.0)
    t = np.arange(n, dtype=float)
    fractions_daily = 1 / (1 + np.exp(-alpha * (t - t_0)))

    # Aggregate to weekly fractions
    from assignment_of_cases_to_variants.math import weekly_variant_fraction

    variant_cases = fractions_daily * case_data
    fraction_data = weekly_variant_fraction(dates, variant_cases, case_data)
    return dates, case_data, fraction_data


class TestFitLogisticVariant:
    def test_returns_four_values(self):
        dates, case_data, fraction_data = make_synthetic_data()
        result = fit_logistic_variant(dates, case_data, fraction_data, REF)
        assert len(result) == 4

    def test_alpha_is_positive(self):
        dates, case_data, fraction_data = make_synthetic_data()
        alpha, t_0, c, residual = fit_logistic_variant(
            dates, case_data, fraction_data, REF
        )
        assert alpha > 0

    def test_c_is_one_without_capacity_fit(self):
        dates, case_data, fraction_data = make_synthetic_data()
        alpha, t_0, c, residual = fit_logistic_variant(
            dates, case_data, fraction_data, REF
        )
        assert c == 1.0

    def test_c_is_fitted_with_capacity_fit(self):
        dates, case_data, fraction_data = make_synthetic_data()
        alpha, t_0, c, residual = fit_logistic_variant(
            dates, case_data, fraction_data, REF, fit_carrying_capacity=True
        )
        assert 0.0 <= c <= 1.0

    def test_residual_is_non_negative(self):
        dates, case_data, fraction_data = make_synthetic_data()
        *_, residual = fit_logistic_variant(
            dates, case_data, fraction_data, REF
        )
        assert residual >= 0

    def test_fit_recovers_approximate_parameters(self):
        true_alpha, true_t0 = 0.2, 30.0
        dates, case_data, fraction_data = make_synthetic_data(
            alpha=true_alpha, t_0=true_t0
        )
        alpha, t_0, c, residual = fit_logistic_variant(
            dates, case_data, fraction_data, REF
        )
        assert alpha == pytest.approx(true_alpha, rel=0.1)
        assert t_0 == pytest.approx(true_t0, rel=0.1)


class TestSplitCasesByVariant:
    def _make_df(self, n=30):
        dates = [REF + dt.timedelta(days=i) for i in range(n)]
        return pd.DataFrame({"Date": dates})

    @patch(
        "assignment_of_cases_to_variants.core.load_case_data",
        return_value=np.full(30, 1000),
    )
    def test_variant_column_added(self, mock_load):
        df = self._make_df()
        split_cases_by_variant("Alpha", 0.2, 15.0, 1.0, REF, df, "dummy.csv")
        assert "Alpha" in df.columns

    @patch(
        "assignment_of_cases_to_variants.core.load_case_data",
        return_value=np.full(30, 1000),
    )
    def test_other_variants_column_added(self, mock_load):
        df = self._make_df()
        split_cases_by_variant("Alpha", 0.2, 15.0, 1.0, REF, df, "dummy.csv")
        assert "Other variants" in df.columns

    @patch(
        "assignment_of_cases_to_variants.core.load_case_data",
        return_value=np.full(30, 1000),
    )
    def test_cases_sum_to_total(self, mock_load):
        df = self._make_df()
        split_cases_by_variant("Alpha", 0.2, 15.0, 1.0, REF, df, "dummy.csv")
        total = df["Alpha"] + df["Other variants"]
        np.testing.assert_array_equal(total, np.full(30, 1000))

    def test_splits_from_existing_other_variants(self):
        n = 30
        df = self._make_df(n)
        df["Other variants"] = np.full(n, 500)
        split_cases_by_variant("Delta", 0.2, 15.0, 1.0, REF, df, "dummy.csv")
        assert "Delta" in df.columns
        total = df["Delta"] + df["Other variants"]
        np.testing.assert_array_equal(total, np.full(n, 500))
