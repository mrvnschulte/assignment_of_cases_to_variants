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



def daily_cases_emerging_variant(alpha, t_0, dates, case_data):
    '''
    

    Parameters
    ----------
    alpha : scalar or array_like, optional
        Logistic growth rate. If array_like, has to be the same size as dates
    t_0 : scalar or array_like, optional
        The function's midpoint. If array_like, has to be the same size as t.
    dates : array_like of datetime dates
        The dates for which case data is given. Has to be ordered chronologically.
    case_data : array_like
        The overall cases for each date in dates. Has to be the same size as dates.

    Returns
    -------
    daily_variant_cases : array_like
        For each date in dates, the number of cases for the emerging variant according to a logistic growth ansatz with parameters alpha and t_0.

    '''
    
    days_since_reference_monday = days_since_reference(dates)
    
    # Compute variant fraction for each relevant day according to ansatz function and then daily cases
    daily_fractions = ansatz_function(days_since_reference_monday, alpha, t_0)
    daily_variant_cases = daily_fractions * case_data
    
    return daily_variant_cases



def days_since_reference(dates):
    '''
    

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
    

    Parameters
    ----------
    dates : TYPE
        DESCRIPTION.

    Returns
    -------
    weeks_since_reference_week : TYPE
        DESCRIPTION.

    '''
    
    weeks_since_reference_week = (days_since_reference(dates)/7).astype(int)
    
    return weeks_since_reference_week

def weekly_sum_of_cases(dates, daily_cases):
    '''
    

    Parameters
    ----------
    dates : TYPE
        DESCRIPTION.
    daily_cases : TYPE
        DESCRIPTION.

    Returns
    -------
    weekly_cases : TYPE
        DESCRIPTION.

    '''
    
    weeks_since_reference_week = weeks_since_reference(dates)
    
    weekly_cases = np.array([ np.sum(daily_cases, where = (weeks_since_reference_week == week))
                                     for week in range(weeks_since_reference_week[0], weeks_since_reference_week[-1] + 1) ])
    
    return weekly_cases

def days_of_weeks(weeks, day):
    '''
    

    Parameters
    ----------
    weeks : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.

    Returns
    -------
    thursdays : TYPE
        DESCRIPTION.

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
    thursdays = [reference_monday + dt.timedelta(days=day_codes[day], weeks=int(i)) for i in non_duplicate_weeks]
    
    return thursdays

def weekly_variant_fraction(dates, variant_cases, total_cases):
    '''
    

    Parameters
    ----------
    dates : TYPE
        DESCRIPTION.
    variant_cases : TYPE
        DESCRIPTION.
    total_cases : TYPE
        DESCRIPTION.

    Returns
    -------
    weekly_fraction : TYPE
        DESCRIPTION.

    '''
    
    weekly_variant_cases = weekly_sum_of_cases(dates, variant_cases)
    weekly_total_cases = weekly_sum_of_cases(dates, case_data)
    weekly_fraction = weekly_variant_cases/weekly_total_cases
    
    return weekly_fraction

def non_duplicate_list(duplicate_list):
    
    non_duplicate_list = list(dict.fromkeys(duplicate_list))
    
    return non_duplicate_list


# Reference monday is the first monday of 2021, the week where variant fraction data in Germany became available
reference_monday = dt.date(2021, 1, 4)

start_date = dt.date(2021, 1, 4)
end_date = dt.date(2021, 2, 22)

dates = pd.date_range(start_date, end_date).to_pydatetime()
dates = [date.date() for date in dates]
dates_as_strings = [f'{date:%Y-%m-%d}' for date in dates]
#dates = [dt.date(2021, 1, 4) + dt.timedelta(i) for i in range(0,50)]

test_data = [1.681628290371829995e-03,
1.316999199067206416e-02,
9.812610056247740575e-02,
4.515924237042467371e-01,
8.578712838094859627e-01,
9.796706985589836991e-01,
9.974643890465627605e-01,
9.992539711661633239e-01
]




# Load data (done: infection data, to-do: variant fraction data)

data = pd.read_csv('./Data/Infection Data/COVID-19-Cases 7-Day Incidence Germany.csv')
data = data[data['Altersgruppe'] == '00+']
data = data.loc[data['Meldedatum'].isin(dates_as_strings)]
case_data = data['Faelle_neu'].tolist()


fraction_data = np.array([0.5] * 8)
test_mse = minimizer_function([0.3, 25], dates, test_data, case_data)
print(test_mse)

# Future feature: Compute uncertainties

# Minimize function to obtain optimal parameters and compute daily variant cases according to them
sol = minimize(minimizer_function, [0, 0], args=(dates, test_data, case_data))
print(sol.x)
alpha, t_0 = sol.x
emerging_variant_cases = daily_cases_emerging_variant(alpha, t_0, dates, case_data)
other_variant_cases = case_data - emerging_variant_cases

weeks = weeks_since_reference(dates)
weekly_fraction = weekly_variant_fraction(dates, emerging_variant_cases, case_data)
thursdays = days_of_weeks(weeks, 'Thursday')
mondays = days_of_weeks(weeks, 'Monday')
calendar_weeks = ['{0:d}-{1:02d}'.format(day.isocalendar()[0], day.isocalendar()[1]) for day in thursdays]


# Plot optimal solution
# TODO: Write to csv_file
plt.plot(dates, emerging_variant_cases, '-o', label='Emerging variant')
plt.plot(dates, other_variant_cases, '-o', label='Other variant')
plt.grid()
plt.legend()
plt.title('Daily new infections per variant')
plt.xlabel('Date')
plt.ylabel('New infections')
plt.xticks(mondays, rotation=60)
plt.show()

plt.plot(thursdays, weekly_fraction, 'o', label='Computed data')
plt.plot(thursdays, test_data, 'x', label='Measured data')
plt.grid()
plt.legend()
plt.title('Weekly case fraction of emerging variant')
plt.xlabel('Week')
plt.ylabel('Fraction of cases')
plt.xticks(thursdays, calendar_weeks, rotation=60)
plt.show()