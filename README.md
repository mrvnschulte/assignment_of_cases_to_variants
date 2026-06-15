[![Tests](https://gitlab.itwm.fraunhofer.de/schulte/assignment-of-cases-to-variants/badges/master/pipeline.svg?key_text=Tests&key_width=40)](https://gitlab.itwm.fraunhofer.de/schulte/assignment-of-cases-to-variants/-/pipelines)
[![Coverage](https://gitlab.itwm.fraunhofer.de/schulte/assignment-of-cases-to-variants/badges/master/coverage.svg)](https://gitlab.itwm.fraunhofer.de/schulte/assignment-of-cases-to-variants/-/commits/master)


# Assignment of Cases to Variants

A tool for assigning daily COVID-19 case counts to individual variants using logistic growth fitting.

## Overview

This package fits logistic growth curves to measured weekly variant fraction data and uses
the fitted curves to split daily case counts between an emerging variant and all other variants.
It provides:

- **Logistic growth fitting** of weekly variant fraction data
- **Daily case splitting** between an emerging variant and all other variants
- **Sequential multi-variant assignment**, applying the splitting iteratively for each variant
- Tools for computing **weekly variant fractions** and **moving averages** per variant
- **Visualisation** of variant transitions, daily cases per variant, and weekly variant fractions

## Installation

For development:
```bash
git clone git@gitlab.itwm.fraunhofer.de:schulte/assignment-of-cases-to-variants.git
cd assignment-of-cases-to-variants
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.10
- matplotlib >= 3.5.0
- numpy >= 1.20.0
- openpyxl >= 3.0.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## Method Description

For each variant, a logistic function is fitted to the measured weekly variant fractions:

$$f(t) = \frac{c}{1 + e^{-\alpha (t - t_0)}}$$

where:

- $\alpha$ is the logistic growth rate
- $t_0$ is the midpoint (days since the reference date)
- $c$ is the carrying capacity (fixed at 1 by default, or fitted optionally)

The fitted function is then used to split the daily case count $N(t)$ into:

- **Emerging variant:** $N_\text{variant}(t) = f(t) \cdot N(t)$
- **Other variants:** $N_\text{other}(t) = N(t) - N_\text{variant}(t)$

Variants are processed **sequentially** from latest to earliest. Each variant's cases are
split from the remaining "Other variants" pool of the previous step.

## Project Structure

```
cross_immunity_model/
├── __init__.py                               # Public API
├── core.py                                   # Logistic fitting and case splitting
├── date_utils.py                             # Date utility functions
├── load_data.py                              # Loading case and variant fraction data
├── math.py                                   # Mathematical functions (logistic, moving average, ...)
└── plotter.py                                # Visualisation functions
```

## Running the Script

To run the full case assignment pipeline for all COVID-19 variants in 2021–2022,
navigate to `scripts/` and run the script:

```bash
cd scripts
python case_assignment.py
```

## API Reference

### Core (`core.py`)

| Function | Description |
|---|---|
| `fit_logistic_variant` | Fits a logistic function to weekly variant fraction data |
| `split_cases_by_variant` | Splits daily cases between an emerging variant and others |

### Data Loading (`load_data.py`)

| Function | Description |
|---|---|
| `load_case_data` | Loads daily case counts from CSV for given dates |
| `load_weekly_variant_fraction` | Loads weekly variant fractions from Excel for a date range |
| `_load_case_csv` | Loads and caches the case data CSV (internal) |
| `_load_fraction_xlsx` | Loads and caches the variant fraction Excel file (internal) |

### Mathematics (`math.py`)

| Function | Description |
|---|---|
| `logistic_function` | Evaluates the logistic function with parameters $\alpha$, $t_0$, $c$ |
| `daily_cases_emerging_variant` | Computes daily variant cases via logistic splitting |
| `weekly_sum_of_cases` | Sums daily cases by ISO calendar week |
| `weekly_variant_fraction` | Computes weekly fraction of variant cases among all cases |
| `moving_average` | Computes moving average of a pandas Series |
| `moving_average_per_variant` | Computes moving average of cases per variant |

### Date Utilities (`date_utils.py`)

| Function | Description |
|---|---|
| `days_since_reference` | Returns days elapsed since a reference date |
| `weeks_since_reference` | Returns ISO calendar weeks elapsed since a reference date |
| `days_of_weeks` | Returns the date of a specific weekday for each given week |
| `non_duplicate_list` | Returns a list with duplicates removed, preserving order |
| `date_to_calendar_week` | Converts a date to a `'{year}-KW{week}'` string |

### Visualisation (`plotter.py`)

| Function | Description |
|---|---|
| `plot_variant_transition` | Plots daily case split and weekly fraction fit for one variant |
| `plot_cases_per_variant` | Plots daily cases per variant |
| `plot_fractions_all_variants` | Plots weekly variant fractions as a stacked area chart |
| `plot_moving_average_per_variant` | Plots 7-day moving average of cases per variant |


## Citation

If you use this software in your research, please cite:
[Citation will be added upon publication]


## License

tba

## Author

Marvin Schulte - [marvin.schulte@itwm.fraunhofer.de](mailto:marvin.schulte@itwm.fraunhofer.de)

Fraunhofer Institute for Industrial Mathematics ITWM
