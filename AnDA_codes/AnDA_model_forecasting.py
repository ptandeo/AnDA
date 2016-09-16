#!/usr/bin/env python

""" AnDA_model_forecasting.py: Apply the dynamical models to generate numerical forecasts. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np
from scipy.integrate import odeint
from AnDA_codes.AnDA_dynamical_models import AnDA_Lorenz_63, AnDA_Lorenz_96

def AnDA_model_forecasting(x,GD):
    """ Apply the dynamical models to generate numerical forecasts. """

    # initializations
    N, n = x.shape;
    xf = np.zeros([N,n]);
    xf_mean = np.zeros([N,n]);

    if (GD.model == 'Lorenz_63'):
        for i_N in range(0,N):
            S = odeint(AnDA_Lorenz_63,x[i_N,:],np.arange(0,GD.dt_integration+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
            xf[i_N,:] = S[-1,:];

    elif (GD.model == 'Lorenz_96'):
        for i_N in range(0,N):
            S = odeint(AnDA_Lorenz_96,x[i_N,:],np.arange(0,GD.dt_integration+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
            xf[i_N,:] = S[-1,:];

    xf_mean = xf;
    return xf, xf_mean
            
            
