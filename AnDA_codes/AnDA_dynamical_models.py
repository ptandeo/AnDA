#!/usr/bin/env python

""" AnDA_dynamical_models.py: Define the dynamical models used in the analog and classical data assimilation (Lorenz-63, Lorenz-96 and possibly others). """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np

def AnDA_Lorenz_63(S,t,sigma,rho,beta):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(S[1]-S[0]);
    x_2 = S[0]*(rho-S[2])-S[1];
    x_3 = S[0]*S[1] - beta*S[2];
    dS  = np.array([x_1,x_2,x_3]);
    return dS

def AnDA_Lorenz_96(S,t,F,J):
    """ Lorenz-96 dynamical model. """
    x = np.zeros(J);
    x[0] = (S[1]-S[J-2])*S[J-1]-S[0];
    x[1] = (S[2]-S[J-1])*S[0]-S[1];
    x[J-1] = (S[0]-S[J-3])*S[J-2]-S[J-1];
    for j in range(2,J-1):
        x[j] = (S[j+1]-S[j-2])*S[j-1]-S[j];
    dS = x.T + F;
    return dS

# define here your own dynamical model
