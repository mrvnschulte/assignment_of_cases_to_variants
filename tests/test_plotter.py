import datetime as dt
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from assignment_of_cases_to_variants.plotter import (
    plot_variant_transition,
    plot_cases_per_variant,
    plot_fractions_all_variants,
    plot_moving_average_per_variant,
)

REF = dt.date(2021, 1, 4)


def make_dates(n=28):
    return [REF + dt.timedelta(days=i) for i in range(n)]


def make_cases_df(variants=("Alpha",), n=28):
    dates = make_dates(n)
    data = {"Date": dates, "Other variants": np.full(n, 600.0)}
    for v in variants:
        data[v] = np.full(n, 200.0)
    return pd.DataFrame(data)


@patch("matplotlib.pyplot.show")
@patch("matplotlib.figure.Figure.savefig")
class TestPlotVariantTransition:
    def test_runs_without_error(self, mock_save, mock_show, tmp_path):
        dates = make_dates(28)
        case_data = np.full(28, 1000.0)
        fraction_data = np.full(4, 0.25)
        plot_variant_transition(
            "Alpha",
            dates,
            case_data,
            fraction_data,
            alpha=0.2,
            t_0=14.0,
            c=1.0,
            reference_date=REF,
            folder=tmp_path,
        )
        assert mock_save.call_count == 2

    def test_with_title(self, mock_save, mock_show, tmp_path):
        dates = make_dates(28)
        case_data = np.full(28, 1000.0)
        fraction_data = np.full(4, 0.25)
        plot_variant_transition(
            "Alpha",
            dates,
            case_data,
            fraction_data,
            alpha=0.2,
            t_0=14.0,
            c=1.0,
            reference_date=REF,
            folder=tmp_path,
            title=True,
        )
        assert mock_show.call_count == 2


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
class TestPlotCasesPerVariant:
    def test_runs_without_error(self, mock_save, mock_show, tmp_path):
        df = make_cases_df(["Alpha"])
        plot_cases_per_variant(["Alpha"], df, folder=tmp_path)
        mock_save.assert_called_once()

    def test_multiple_variants(self, mock_save, mock_show, tmp_path):
        df = make_cases_df(["Alpha", "Delta"])
        plot_cases_per_variant(["Alpha", "Delta"], df, folder=tmp_path)
        mock_save.assert_called_once()

    def test_with_title(self, mock_save, mock_show, tmp_path):
        df = make_cases_df(["Alpha"])
        plot_cases_per_variant(["Alpha"], df, folder=tmp_path, title=True)
        mock_save.assert_called_once()


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
class TestPlotFractionsAllVariants:
    def test_runs_without_error(self, mock_save, mock_show, tmp_path):
        df = make_cases_df(["Alpha"])
        plot_fractions_all_variants(["Alpha"], df, folder=tmp_path)
        mock_save.assert_called_once()

    def test_with_title(self, mock_save, mock_show, tmp_path):
        df = make_cases_df(["Alpha"])
        plot_fractions_all_variants(["Alpha"], df, folder=tmp_path, title=True)
        mock_save.assert_called_once()


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
class TestPlotMovingAveragePerVariant:
    def test_runs_without_error(self, mock_save, mock_show, tmp_path):
        df = make_cases_df(["Alpha"])
        plot_moving_average_per_variant(["Alpha"], df, folder=tmp_path)
        mock_save.assert_called_once()

    def test_with_title(self, mock_save, mock_show, tmp_path):
        df = make_cases_df(["Alpha"])
        plot_moving_average_per_variant(
            ["Alpha"], df, folder=tmp_path, title=True
        )
        mock_save.assert_called_once()
