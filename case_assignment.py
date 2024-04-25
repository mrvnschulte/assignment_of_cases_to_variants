# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:39:51 2024

@author: schulte
"""

import numpy as np
import datetime as dt
from numpy.linalg import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd


# As the ansatz function, we use a logistic function

def ansatz_function(t, alpha=1, t_0=0):
    '''
    Computes the logistic function with parameters alpha and t_0 for all elements in t.

    Parameters
    ----------
    t : array_like
        Input data.
    alpha : scalar or array_like, optional
        Logistic growth rate. If array_like, has to be the same size as t.
    t_0 : scalar or array_like, optional
        The function's midpoint. If array_like, has to be the same size as t.

    Returns
    -------
    y : array_like
        The logistic function with parameters alpha and t_0 of each element in t

    '''
    
    y = 1/(1+np.exp(-alpha*(t-t_0)))
    
    return y



def minimizer_function(param, dates, fraction_data, case_data):
    '''
    Given the parameters alpha and t_0 of the logistic function, computes the mean-squared error between the modeled variant fraction per week and the given data.

    Parameters
    ----------
    param : array_like
        The parameters alpha and t_0 of the logistic function.
    dates : array_like of datetime dates
        The dates for which case and variant fraction data is given. Has to be ordered chronologically.
    fraction_data : array_like
        The variant fraction of the emerging variant for each week with at least one date in dates.
    case_data : array_like
        The overall cases for each date in dates. Has to be the same size as dates.

    Returns
    -------
    residual_norm : float
        The norm of the residual vector between the modeled variant fraction per week and the given data.

    '''
    
    # Compute daily cases of emerging variant and add cases weekly
    daily_variant_cases = daily_cases_emerging_variant(param[0], param[1], dates, case_data)
    ansatz_variant_fraction = weekly_variant_fraction(dates, daily_variant_cases, case_data)
    
    
    # Compute mean-squared error between model and data for variant fractions
    # Residual norm gives more accurate optimization results than mean-squared error (by direct computing implementation)
    residual_norm = norm(fraction_data - ansatz_variant_fraction)
    
    return residual_norm



def daily_cases_emerging_variant(alpha, t_0, dates, case_data, integer=False):
    '''
    Returns the daily number of cases according to a logistic splitting with parameters alpha and t_0

    Parameters
    ----------
    alpha : scalar or array_like
        Logistic growth rate. If array_like, has to be the same size as dates
    t_0 : scalar or array_like
        The function's midpoint. If array_like, has to be the same size as t.
    dates : array_like of datetime dates
        The dates for which case data is given. Has to be ordered chronologically.
    case_data : array_like
        The overall cases for each date in dates. Has to be the same size as dates.
    integer : Boolean, optional
        If True, daily cases are rounded to the next integer

    Returns
    -------
    daily_variant_cases : array_like
        For each date in dates, the number of cases for the emerging variant according to a logistic growth ansatz with parameters alpha and t_0.

    '''
    
    days_since_reference_monday = days_since_reference(dates)
    
    # Compute variant fraction for each relevant day according to ansatz function and then daily cases
    daily_fractions = ansatz_function(days_since_reference_monday, alpha, t_0)
    
    if integer:
        daily_variant_cases = np.round(daily_fractions * case_data).astype('int')
    else:
        daily_variant_cases = daily_fractions * case_data
    
    return daily_variant_cases



def days_since_reference(dates):
    '''
    Returns the days since reference monday

    Parameters
    ----------
    dates : array_like of datetime dates
        Input data.

    Returns
    -------
    days_since_reference_monday : array_like
        For each date in dates, the number of days since reference monday.

    '''
    
    days_since_reference_monday = np.array([(date - reference_monday).days for date in dates])
    
    return days_since_reference_monday

def weeks_since_reference(dates):
    '''
    Returns the weeks since reference monday

    Parameters
    ----------
    dates : array_like of datetime dates
        Input data.

    Returns
    -------
    weeks_since_reference_week : array_like
        For each date in dates, the number of complete weeks since reference monday.

    '''
    
    weeks_since_reference_week = (days_since_reference(dates)/7).astype(int)
    
    return weeks_since_reference_week

def weekly_sum_of_cases(dates, daily_cases):
    '''
    Sums up the daily cases to weekly ones

    Parameters
    ----------
    dates : array_like of datetime dates
        All dates, for which daily cases are given.
    daily_cases : array_like
        Cases for each date in dates. Has to be the same size as dates.

    Returns
    -------
    weekly_cases : array_like
        For each week with at least one date in dates, returns the weekly sum of cases.

    '''
    
    weeks_since_reference_week = weeks_since_reference(dates)
    
    weekly_cases = np.array([ np.sum(daily_cases, where = (weeks_since_reference_week == week))
                                     for week in range(weeks_since_reference_week[0], weeks_since_reference_week[-1] + 1) ])
    
    return weekly_cases

def days_of_weeks(weeks, day):
    '''
    Returns all date that are the specified day in the weeks given.

    Parameters
    ----------
    weeks : array_like
        List of integers, representing the weeks since reference monday.
    day : String
        Day of the week.

    Returns
    -------
    dates : array_like of datetime dates
        The dates of each specified day of the week for the given weeks.

    '''
    
    day_codes = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6}
    
    non_duplicate_weeks = non_duplicate_list(weeks)
    dates = [reference_monday + dt.timedelta(days=day_codes[day], weeks=int(i)) for i in non_duplicate_weeks]
    
    return dates

def weekly_variant_fraction(dates, variant_cases, total_cases):
    '''
    Returns the weekly variant fraction for all weeks with dates in dates

    Parameters
    ----------
    dates : array_like of datetime dates
        All dates, for which case data is given.
    variant_cases : array_like
        Daily cases of the respective variant.
    total_cases : array_like
        Overall daily cases.

    Returns
    -------
    weekly_fraction : array_like
        Weekly fraction of variant cases among all cases.

    '''
    
    weekly_variant_cases = weekly_sum_of_cases(dates, variant_cases)
    weekly_total_cases = weekly_sum_of_cases(dates, total_cases)
    weekly_fraction = weekly_variant_cases/weekly_total_cases
    
    return weekly_fraction

def non_duplicate_list(duplicate_list):
    '''
    Returns the list without duplicates

    Parameters
    ----------
    duplicate_list : list
        Input data.

    Returns
    -------
    non_duplicate_list : list
        List with elements of duplicate_list, but without duplicates.

    '''
    
    non_duplicate_list = list(dict.fromkeys(duplicate_list))
    
    return non_duplicate_list

def get_previous_monday(date):
    '''
    Returns the last Monday before date

    Parameters
    ----------
    date : datetime date
        Input data.

    Returns
    -------
    prev_monday : datetime date
        The Monday previous to date.

    '''
    
    prev_monday = date - dt.timedelta(days=date.weekday())
    
    return prev_monday

def german_percent_strings_to_float(strings):
    '''
    Converts an array of strings which represent percentages with decimal commas into floats

    Parameters
    ----------
    strings : array_like of strings
        Array of strings which represent percentages with decimal commas.

    Returns
    -------
    float_list : array_like
        Array of float, given by the strings in strings.

    '''
    
    float_list = [float(string.rstrip('%').replace(',', '.'))/100 for string in strings]
    
    return float_list



def compute_emerging_variant(variant, start_date, end_date, data_frame):
    '''
    Splits data between the respective variant and others according to an interpolation respecting data between start and end date

    Parameters
    ----------
    variant : string
        Name of the emerging variant.
    start_date : datetime date
        First date, for which data should be used for interpolation.
    end_date : datetime date
        Last date, for which data should be used for interpolation.
    data_frame : Pandas dataframe
        Data Frame, in which to store computed values.

    Returns
    -------
    None.

    '''
    
    # Initialize
    dates = pd.date_range(start_date, end_date).to_pydatetime()
    dates = [date.date() for date in dates]
    dates_as_strings = [f'{date:%Y-%m-%d}' for date in dates]
    
    weeks = weeks_since_reference(dates)
    
    mondays = days_of_weeks(weeks, 'Monday')
    month_names = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']
    mondays_as_german_strings = [f'{date.day}. {month_names[date.month-1]} {date.year}' for date in mondays]
    
    
    # Load data
    
    data = pd.read_csv('./Data/Infection Data/COVID-19-Cases 7-Day Incidence Germany.csv')
    data = data[data['Altersgruppe'] == '00+']
    case_data = data.loc[data['Meldedatum'].isin(dates_as_strings)]
    case_data = case_data['Faelle_neu'].tolist()
    
    frac_data = pd.read_csv('./Data/Variant Fraction Data/VOC Data.csv', sep=';')
    frac_data = frac_data[frac_data['WHO_LABEL_RENAMED'] == variant]
    frac_data = frac_data.loc[frac_data['ISO-Woche von Sequence.Date Of Sampling (filtered)'].isin(mondays_as_german_strings)]
    fraction_data = frac_data['% von insgesamt Anzahl von SEQUENCE.ID'].tolist()
    fraction_data = german_percent_strings_to_float(fraction_data)
    
    
    # Future feature: Compute uncertainties
    
    
    
    # Minimize function to obtain optimal parameters and compute daily variant cases according to them
    start_end_relative = days_since_reference([start_date, end_date])
    initial_start = np.average(start_end_relative)
    
    sol = minimize(minimizer_function, [0.1, initial_start], args=(dates, fraction_data, case_data))
    print(sol.x)
    print(sol.fun)
    alpha, t_0 = sol.x
    emerging_variant_cases = daily_cases_emerging_variant(alpha, t_0, dates, case_data, integer=True)
    other_variant_cases = case_data - emerging_variant_cases
    
    
    weekly_fraction = weekly_variant_fraction(dates, emerging_variant_cases, case_data)
    thursdays = days_of_weeks(weeks, 'Thursday')
    calendar_weeks = ['{0:d}-{1:02d}'.format(day.isocalendar()[0], day.isocalendar()[1]) for day in thursdays]
    
    
    # Plot optimal solution
    var = variant
    if var=='Omikron':
        var = var.replace('k', 'c')
        
    plt.plot(dates, emerging_variant_cases, '-o', label=f'{var} variant')
    plt.plot(dates, other_variant_cases, '-o', label='Other variants')
    plt.grid()
    plt.legend()
    plt.title('Reported new infections per variant and day')
    plt.xlabel('Date')
    plt.ylabel('New infections')
    plt.xticks(mondays, rotation=60)
    plt.tight_layout()
    plt.savefig(f'./Plots/Daily_new_infections_{var}_variant.{fileformat}')
    plt.show()
    
    
    plt.plot(thursdays, weekly_fraction, 'o', label='Computed data')
    plt.plot(thursdays, fraction_data, 'x', label='Measured data')
    plt.grid()
    plt.legend()
    
    plt.title(f'Weekly case fraction of {var} variant')
    plt.xlabel('Week')
    plt.ylabel('Fraction of cases')
    plt.xticks(thursdays, calendar_weeks, rotation=60)
    plt.tight_layout()
    plt.savefig(f'./Plots/Weekly_case_fraction_{var}_variant.{fileformat}')
    plt.show()
    
    all_dates_as_strings = [f'{date:%Y-%m-%d}' for date in data_frame['Date']]
    all_case_data = data.loc[data['Meldedatum'].isin(all_dates_as_strings)]
    all_case_data = all_case_data['Faelle_neu'].tolist()
    all_emerging_variant_cases = daily_cases_emerging_variant(alpha, t_0, data_frame['Date'].tolist(), all_case_data, integer=True)
    
    
    if 'Other variants' in data_frame:
        all_emerging_variant_cases = daily_cases_emerging_variant(alpha, t_0, data_frame['Date'].tolist(), data_frame['Other variants'], integer=True)
        all_other_variant_cases = data_frame['Other variants'].tolist() - all_emerging_variant_cases
        data_frame['Other variants'] = all_other_variant_cases
    else:
        all_case_data = data.loc[data['Meldedatum'].isin(all_dates_as_strings)]
        all_case_data = all_case_data['Faelle_neu'].tolist()
        all_emerging_variant_cases = daily_cases_emerging_variant(alpha, t_0, data_frame['Date'].tolist(), all_case_data, integer=True)
        all_other_variant_cases = all_case_data - all_emerging_variant_cases
        data_frame.insert(1, 'Other variants', all_other_variant_cases)

    data_frame.insert(2, variant, all_emerging_variant_cases)
    

# Reference monday is the first monday of 2021, the week where variant fraction data in Germany became available
reference_monday = dt.date(2021, 1, 4)

# Specify start and end dates of interpolation for respective variants
start_dates = [dt.date(2021, 11, 15), dt.date(2021, 4, 19), dt.date(2021, 1, 4)]
end_dates = [dt.date(2022, 1, 2), dt.date(2021, 8, 8), dt.date(2021, 4, 4)]
variants = ['Omikron', 'Delta', 'Alpha']

# Create data frame, in which to store computed values.
all_dates = pd.date_range(dt.date(2021, 1, 1), dt.date(2021, 12, 31)).to_pydatetime()
all_dates = [date.date() for date in all_dates]
cases_per_variant = pd.DataFrame(all_dates, columns=['Date'])

fileformat = 'svg'

# Compute optimal solution
for i in range(len(variants)):
    compute_emerging_variant(variants[i], start_dates[i], end_dates[i], cases_per_variant)

# Plot optimal solution    
plt.plot(cases_per_variant['Date'], cases_per_variant['Other variants'], '-o', label='Wild Type')
plt.plot(cases_per_variant['Date'], cases_per_variant['Alpha'], '-o', label='Alpha')
plt.plot(cases_per_variant['Date'], cases_per_variant['Delta'], '-o', label='Delta')
plt.plot(cases_per_variant['Date'], cases_per_variant['Omikron'], '-o', label='Omicron')
plt.grid()
plt.legend()
plt.title('Reported new infections per variant and day')
plt.xlabel('Date')
plt.ylabel('New infections')
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f'./Plots/Daily_new_infections_2021.{fileformat}')
plt.show()

# Store optimal solution as csv and excel
cases_per_variant.to_csv('./Data/daily_cases_per_variant.csv', index=False)
cases_per_variant.to_excel('./Data/daily_cases_per_variant.xlsx', index=False)