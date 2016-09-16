#!/usr/bin/env python

""" AnDA_generate_data.py: Use the dynamical models to generate true state, noisy observations and catalog of numerical simulations. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np
from scipy.integrate import odeint
from AnDA_codes.AnDA_dynamical_models import AnDA_Lorenz_63, AnDA_Lorenz_96

def AnDA_generate_data(GD):
    """ Generate the true state, noisy observations and catalog of numerical simulations. """

    # initialization
    class xt:
        values = [];
        time = [];
    class yo:
        values = [];
        time = [];
    class catalog:
        analogs = [];
        successors = [];
        source = [];
    
    # test on parameters
    if GD.dt_states>GD.dt_obs:
        print('Error: GD.dt_obs must be bigger than GD.dt_states');
    if (np.mod(GD.dt_obs,GD.dt_states)!=0):
        print('Error: GD.dt_obs must be a multiple of GD.dt_states');

    # use this to generate the same data for different simulations
    np.random.seed(1);
    
    if (GD.model == 'Lorenz_63'):
    
        # 5 time steps (to be in the attractor space)       
        x0 = np.array([8.0,0.0,30.0]);
        S = odeint(AnDA_Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        x0 = S[S.shape[0]-1,:];

        # generate true state (xt)
        S = odeint(AnDA_Lorenz_63,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        T_test = S.shape[0];      
        t_xt = np.arange(0,T_test,GD.dt_states);       
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];
        
        # generate  partial/noisy observations (yo)
        eps = np.random.multivariate_normal(np.zeros(3),GD.sigma2_obs*np.eye(3,3),T_test);
        yo_tmp = S[t_xt,:]+eps[t_xt,:];
        t_yo = np.arange(0,T_test,GD.dt_obs);
        i_t_obs = np.where((np.in1d(t_xt,t_yo))==True)[0];
        yo.values = xt.values*np.nan;
        yo.values[np.ix_(i_t_obs,GD.var_obs)] = yo_tmp[np.ix_(i_t_obs,GD.var_obs)];
        yo.time = xt.time;
       

        #generate catalog
        S =  odeint(AnDA_Lorenz_63,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(3),GD.sigma2_catalog*np.eye(3,3),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-GD.dt_states,:];
        catalog.successors = catalog_tmp[GD.dt_states:,:]
        catalog.source = GD.parameters;
    
    elif (GD.model == 'Lorenz_96'):
        
        # 5 time steps (to be in the attractor space)
        x0 = GD.parameters.F*np.ones(GD.parameters.J);
        x0[np.int(np.around(GD.parameters.J/2))] = x0[np.int(np.around(GD.parameters.J/2))] + 0.01;
        S = odeint(AnDA_Lorenz_96,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
        x0 = S[S.shape[0]-1,:];
       

        # generate true state (xt)
        S = odeint(AnDA_Lorenz_96,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));       
        T_test = S.shape[0];     
        t_xt = np.arange(0,T_test,GD.dt_states);
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];

        
        # generate partial/noisy observations (yo)
        eps = np.random.multivariate_normal(np.zeros(GD.parameters.J),GD.sigma2_obs*np.eye(GD.parameters.J),T_test);
        yo_tmp = S[t_xt,:]+eps[t_xt,:];
        t_yo = np.arange(0,T_test,GD.dt_obs);
        i_t_obs = np.where((np.in1d(t_xt,t_yo))==True)[0];
        yo.values = xt.values*np.nan;
        yo.values[np.ix_(i_t_obs,GD.var_obs)] = yo_tmp[np.ix_(i_t_obs,GD.var_obs)];
        yo.time = xt.time;
        
        
        # generate catalog
        S =  odeint(AnDA_Lorenz_96,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));        
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(GD.parameters.J),GD.sigma2_catalog*np.eye(GD.parameters.J,GD.parameters.J),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-GD.dt_states,:];
        catalog.successors = catalog_tmp[GD.dt_states:,:]
        catalog.source = GD.parameters;
    
    # reinitialize random generator number
    np.random.seed()

    return catalog, xt, yo;


    
    

                                      
    
    
