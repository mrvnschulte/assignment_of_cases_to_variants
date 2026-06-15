import datetime as dt

import numpy as np
import pytest

from assignment_of_cases_to_variants.date_utils import (
    days_since_reference,
    weeks_since_reference,
    days_of_weeks,
    non_duplicate_list,
    date_to_calendar_week,
)

REF = dt.date(2021, 1, 4)  # Monday


class TestDaysSinceReference:
    def test_same_day_returns_zero(self):
        result = days_since_reference([REF], REF)
        np.testing.assert_array_equal(result, [0])

    def test_one_day_later(self):
        dates = [REF + dt.timedelta(days=1)]
        result = days_since_reference(dates, REF)
        np.testing.assert_array_equal(result, [1])

    def test_multiple_dates(self):
        dates = [REF + dt.timedelta(days=i) for i in range(5)]
        result = days_since_reference(dates, REF)
        np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])

    def test_returns_ndarray(self):
        result = days_since_reference([REF], REF)
        assert isinstance(result, np.ndarray)

    def test_date_before_reference_is_negative(self):
        dates = [REF - dt.timedelta(days=3)]
        result = days_since_reference(dates, REF)
        np.testing.assert_array_equal(result, [-3])


class TestWeeksSinceReference:
    def test_same_week_returns_zero(self):
        dates = [REF, REF + dt.timedelta(days=1)]  # Mon + Tue same week
        result = weeks_since_reference(dates, REF)
        np.testing.assert_array_equal(result, [0, 0])

    def test_next_week(self):
        dates = [REF + dt.timedelta(weeks=1)]
        result = weeks_since_reference(dates, REF)
        np.testing.assert_array_equal(result, [1])

    def test_multiple_weeks(self):
        dates = [REF + dt.timedelta(weeks=i) for i in range(4)]
        result = weeks_since_reference(dates, REF)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_returns_ndarray(self):
        result = weeks_since_reference([REF], REF)
        assert isinstance(result, np.ndarray)


class TestDaysOfWeeks:
    def test_returns_correct_weekday_monday(self):
        weeks = [0, 1, 2]
        result = days_of_weeks(weeks, REF, "Monday")
        assert all(d.weekday() == 0 for d in result)

    def test_returns_correct_weekday_friday(self):
        weeks = [0, 1]
        result = days_of_weeks(weeks, REF, "Friday")
        assert all(d.weekday() == 4 for d in result)

    def test_no_duplicates_in_result(self):
        weeks = [0, 0, 1, 1]
        result = days_of_weeks(weeks, REF, "Monday")
        assert len(result) == len(set(result))

    def test_correct_number_of_dates(self):
        weeks = [0, 1, 2]
        result = days_of_weeks(weeks, REF, "Wednesday")
        assert len(result) == 3


class TestNonDuplicateList:
    def test_removes_duplicates(self):
        assert non_duplicate_list([1, 2, 2, 3]) == [1, 2, 3]

    def test_preserves_order(self):
        assert non_duplicate_list([3, 1, 2, 1, 3]) == [3, 1, 2]

    def test_empty_list(self):
        assert non_duplicate_list([]) == []

    def test_no_duplicates_unchanged(self):
        assert non_duplicate_list([1, 2, 3]) == [1, 2, 3]

    def test_all_duplicates(self):
        assert non_duplicate_list([5, 5, 5]) == [5]


class TestDateToCalendarWeek:
    def test_format(self):
        date = dt.date(2021, 1, 4)
        result = date_to_calendar_week(date)
        assert result == "2021-KW01"

    def test_week_10(self):
        date = dt.date(2021, 3, 8)  # KW10
        result = date_to_calendar_week(date)
        assert "KW10" in result

    def test_returns_string(self):
        result = date_to_calendar_week(dt.date(2021, 6, 1))
        assert isinstance(result, str)
