import assignment_of_cases_to_variants as pkg


class TestPublicAPI:
    def test_all_symbols_importable(self):
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"{name} missing in module"

    def test_fit_logistic_variant_callable(self):
        assert callable(pkg.fit_logistic_variant)

    def test_split_cases_by_variant_callable(self):
        assert callable(pkg.split_cases_by_variant)

    def test_load_case_data_callable(self):
        assert callable(pkg.load_case_data)

    def test_load_weekly_variant_fraction_callable(self):
        assert callable(pkg.load_weekly_variant_fraction)

    def test_moving_average_per_variant_callable(self):
        assert callable(pkg.moving_average_per_variant)

    def test_plot_functions_callable(self):
        assert callable(pkg.plot_cases_per_variant)
        assert callable(pkg.plot_fractions_all_variants)
        assert callable(pkg.plot_moving_average_per_variant)
        assert callable(pkg.plot_variant_transition)
