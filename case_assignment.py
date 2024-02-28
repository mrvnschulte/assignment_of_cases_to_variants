# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:39:51 2024

@author: schulte
"""

import numpy as np
import datetime as dt


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
    days_since_reference_monday = [(date - reference_monday).days for date in dates]
    week_since_reference_week = [int(i/7) for i in days_since_reference_monday]
    
    # Compute variant fraction for each relevant day according to ansatz function
    daily_fractions = [ansatz_function(day, param[0], param[1]) for day in days_since_reference_monday]
    
    return daily_fractions

reference_monday = dt.date(2021, 1, 4)

dates = [dt.date(2021, 1, 4) + dt.timedelta(i) for i in range(0,50)]

test = minimizer_function([1, 25], dates, 0, 0)
print(test)
# Load data

# Future feature: Compute uncertainties

# Minimize function to obtain optimal parameter

# Plot optimal solution