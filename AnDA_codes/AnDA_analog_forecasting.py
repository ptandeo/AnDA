#!/usr/bin/env python

""" AnDA_analog_forecasting.py: Apply the analog method on catalog of historical data to generate forecasts. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from AnDA_codes.AnDA_stat_functions import normalise, mk_stochastic, sample_discrete

def AnDA_analog_forecasting(x, AF):
    """ Apply the analog method on catalog of historical data to generate forecasts. """

    # initializations
    N, n = x.shape;
    xf = np.zeros([N,n]);
    xf_mean = np.zeros([N,n]);

    # local or global analog forecasting
    stop_condition = 0;
    i_var = np.array([0]);
    while (stop_condition !=1):

        # in case of global approach
        if np.array_equal(AF.neighborhood, np.ones([n,n])):
            i_var_neighboor = np.arange(0,n);
            i_var = np.arange(0,n);
            stop_condition = 1;
        # in case of local approach
        else:
            i_var_neighboor = np.where(AF.neighborhood[int(i_var),:]==1)[0];

        # find the indices and distances of the k-nearest neighbors (knn)
        kdt = KDTree(AF.catalog.analogs[:,i_var_neighboor], leaf_size=50, metric='euclidean')
        dist_knn, index_knn = kdt.query(x[:,i_var_neighboor], AF.k)  
        
        # normalisation parameter for the kernels
        lambdaa = np.median(dist_knn);

        # compute weights
        if AF.k == 1:
            weights = np.ones([N,1]);
        else:
            weights = mk_stochastic(np.exp(-np.power(dist_knn,2)/lambdaa));
        
        # for each member/particle
        for i_N in range(0,N):

            xf_tmp = np.zeros([AF.k,np.max(i_var)+1]);
            
            # select the regression method
            if (AF.regression == 'locally_constant'):
                xf_tmp[:,i_var] = AF.catalog.successors[np.ix_(index_knn[i_N,:],i_var)];
                # weighted mean and covariance
                xf_mean[i_N,i_var] = np.sum(xf_tmp[:,i_var]*np.repeat(weights[i_N,:][np.newaxis].T,len(i_var),1),0)
                E_xf = (xf_tmp[:,i_var]-np.repeat(xf_mean[i_N,i_var][np.newaxis],AF.k,0)).T;
                cov_xf = 1.0/(1.0-np.sum(np.power(weights[i_N,:],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T);

            elif (AF.regression == 'increment'):
                xf_tmp[:,i_var] = np.repeat(x[i_N,i_var][np.newaxis],AF.k,0) + AF.catalog.successors[np.ix_(index_knn[i_N,:],i_var)]-AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var)];
                # weighted mean and covariance
                xf_mean[i_N,i_var] = np.sum(xf_tmp[:,i_var]*np.repeat(weights[i_N,:][np.newaxis].T,len(i_var),1),0);
                E_xf = (xf_tmp[:,i_var]-np.repeat(xf_mean[i_N,i_var][np.newaxis],AF.k,0)).T;               
                cov_xf = 1.0/(1-np.sum(np.power(weights[i_N,:],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T);

            elif (AF.regression == 'local_linear'):                
                # NEW VERSION (USING PCA)
                # pca with weighted observations
                mean_x = np.sum(AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var_neighboor)]*np.repeat(weights[i_N,:][np.newaxis].T,len(i_var_neighboor),1),0)
                analog_centered = AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var_neighboor)] - np.repeat(mean_x[np.newaxis],AF.k,0)
                analog_centered = analog_centered*np.repeat(np.sqrt(weights[i_N,:])[np.newaxis].T,len(i_var_neighboor),1)
                U, S, V = np.linalg.svd(analog_centered,full_matrices=False)
                coeff = V.T[:,0:5];
                
                W = np.sqrt(np.diag(weights[i_N,:]));
                A = np.insert(np.dot(AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var_neighboor)],coeff),0,1,1);
                Aw = np.dot(W,A);
                B = AF.catalog.successors[np.ix_(index_knn[i_N,:],i_var)];			
                Bw = np.dot(W,B);		
                mu = np.dot( np.insert(np.dot(x[i_N,i_var_neighboor],coeff),0,1),np.linalg.lstsq(Aw,Bw)[0]);               
                pred = np.dot( A ,np.linalg.lstsq(A,B)[0]);
                res = B-pred;
                xf_tmp[:,i_var] = np.tile(mu,(AF.k,1))+res;
                # weighted mean and covariance
                xf_mean[i_N,i_var] = mu;
                if len(i_var)>1:
                    cov_xf = np.cov(res.T);
                else:
                    cov_xf = np.cov(res.T)[np.newaxis][np.newaxis];		
                # constant weights for local linear
                weights[i_N,:] = 1.0/len(weights[i_N,:]);
            else:
                print("Error: choose AF.regression between 'locally_constant', 'increment', 'local_linear' ")
                quit() 
            # select the sampling method
            if (AF.sampling =='gaussian'):
                # random sampling from the multivariate Gaussian distribution
                xf[i_N,i_var] = np.random.multivariate_normal(xf_mean[i_N,i_var],cov_xf);
            elif (AF.sampling =='multinomial'):
                # random sampling from the multinomial distribution of the weights
                i_good = sample_discrete(weights[i_N,:],1,1);
                xf[i_N,i_var] = xf_tmp[i_good,i_var];
            else:
                print("Error: choose AF.sampling between 'gaussian', 'multinomial' ")
                quit()

        # stop condition
        if (np.array_equal(i_var,np.array([n-1])) or (len(i_var) == n)):
            stop_condition = 1;
            
        else:
            i_var = i_var + 1;
    
    return xf, xf_mean; # end
        
            
            
        
        

            
        
    
    
