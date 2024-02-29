# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:39:51 2024

@author: schulte
"""

import numpy as np
import datetime as dt
from numpy.linalg import norm
from scipy.optimize import minimize


# As the ansatz function, we use a logistic function

def ansatz_function(t, alpha=1, t_0=0):
    '''
    Computes the logistic function with parameters alpha and t_0 for all elements in t.

    Parameters
    ----------
    t : array_like
        Input data.
    alpha : float, optional
        Logistic growth rate.
    t_0 : array_like, optional
        The function's midpoint, has to be the same size as t.

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
        The variant fraction of the emerging variant for each week with at least one date in dates. Has to be the same size as dates.
    case_data : array_like
        The overall cases for each date in dates. Has to be the same size as dates.

    Returns
    -------
    mean_squared_error : float
        The mean-squared error between the modeled variant fraction per week and the given data.

    '''
    
    
    # Return days since reference monday and compute weeks since week beginning with reference monday
    days_since_reference_monday = np.array([(date - reference_monday).days for date in dates])
    week_since_reference_week = (days_since_reference_monday/7).astype(int)
    
    # Compute variant fraction for each relevant day according to ansatz function and then daily cases
    daily_fractions = ansatz_function(days_since_reference_monday, param[0], param[1])
    daily_variant_cases = daily_fractions * case_data
    
    # Sum over all daily cases within a week
    weekly_variant_cases = np.array([ np.sum(daily_variant_cases, where = (week_since_reference_week == week))
                                     for week in range(week_since_reference_week[0], week_since_reference_week[-1] + 1) ])
    
    # Compute total weekly cases
    total_weekly_cases = np.array([ np.sum(case_data, where = (week_since_reference_week == week))
                                     for week in range(week_since_reference_week[0], week_since_reference_week[-1] + 1) ])
    
    # Compute weekly variant fraction according to ansatz function
    ansatz_variant_fraction = weekly_variant_cases/total_weekly_cases
    
    # Compute mean-squared error between model and data for variant fractions
    mean_squared_error = norm(fraction_data - ansatz_variant_fraction)
    
    return mean_squared_error

reference_monday = dt.date(2021, 1, 4)

dates = [dt.date(2021, 1, 4) + dt.timedelta(i) for i in range(0,50)]

test_data = [1.123326273421474010e-09,
1.528596494836050375e-06,
1.345074042812720088e-03,
3.712180266456135969e-01,
9.903258463629376340e-01,
9.999887054671490194e-01,
9.999999916996794580e-01,
9.999999999622486424e-01]


case_data = np.array([1000, 2000] * 25)
fraction_data = np.array([0.5] * 8)
test_mse = minimizer_function([1, 25], dates, test_data, case_data)
print(test_mse)
# Load data

# Future feature: Compute uncertainties

# Minimize function to obtain optimal parameter
sol = minimize(minimizer_function, [0, 0], args=(dates, test_data, case_data))
print(sol.x)

# Plot optimal solution