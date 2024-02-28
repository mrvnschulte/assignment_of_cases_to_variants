# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:39:51 2024

@author: schulte
"""

import numpy as np
import datetime as dt
from numpy.linalg import norm


# As the ansatz function, we use a logistic function

def ansatz_function(t, alpha, t_0):
    '''
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    t_0 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    return 1/(1+np.exp(-alpha*(t-t_0)))



def minimizer_function(param, dates, fraction_data, case_data):
    '''
    

    Parameters
    ----------
    param : TYPE
        DESCRIPTION.
    dates : TYPE
        DESCRIPTION.
    fraction_data : TYPE
        DESCRIPTION.
    case_data : TYPE
        DESCRIPTION.

    Returns
    -------
    week_since_reference_week : TYPE
        DESCRIPTION.

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
    
    return ansatz_variant_fraction

reference_monday = dt.date(2021, 1, 4)

dates = [dt.date(2021, 1, 4) + dt.timedelta(i) for i in range(0,50)]


case_data = np.array([1000, 2000] * 25)
fraction_data = np.array([0.5] * 8)
test2 = minimizer_function([1, 25], dates, fraction_data, case_data)
print(test2)
# Load data

# Future feature: Compute uncertainties

# Minimize function to obtain optimal parameter

# Plot optimal solution