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

    fmodel = {}

    fmodel['Lorenz_63'] = lambda x0: odeint(AnDA_Lorenz_63,x0,(0,GD.dt_integration),
                                       args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta))[-1]

    fmodel['Lorenz_96'] = lambda x0: odeint(AnDA_Lorenz_96,x0,(0,GD.dt_integration),
                                       args=(GD.parameters.F,GD.parameters.J))[-1]

    xf = np.apply_along_axis(fmodel[GD.model], 1, x)
    xf_mean = np.copy(xf)

    return xf, xf_mean
            
            
