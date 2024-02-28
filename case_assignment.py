# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:39:51 2024

@author: schulte
"""

import numpy as np
import datetime as dt


# As the ansatz function, we use a logistic function

def ansatz_function(t, alpha, t_0):
    
    return 1/(1+np.exp(-alpha*(t-t_0)))



def minimizer_function(param, dates, fraction_data, case_data):
    
    # Return days since reference monday and compute weeks since week beginning with reference monday
    days_since_reference_monday = [(date - reference_monday).days for date in dates]
    week_since_reference_week = [int(i/7) for i in days_since_reference_monday]
    
    return week_since_reference_week

reference_monday = dt.date(2021, 1, 4)

dates = [dt.date(2021, 1, 4) + dt.timedelta(i) for i in range(0,50)]

test = minimizer_function(0, dates, 0, 0)
print(test)
# Load data

# Future feature: Compute uncertainties

# Minimize function to obtain optimal parameter

# Plot optimal solution