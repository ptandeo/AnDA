#!/usr/bin/env python

""" AnDA_data_assimilation.py: Apply stochastic and sequential data assimilation technics using model forecasting or analog forecasting. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.1"
__date__ = "2019-02-06"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@imt-atlantique.fr"

import numpy as np
from scipy.stats import multivariate_normal
from AnDA_codes.AnDA_stat_functions import resampleMultinomial, inv_using_SVD
from tqdm import tqdm

def AnDA_data_assimilation(yo, DA):
    """ 
    Apply stochastic and sequential data assimilation technics using 
    model forecasting or analog forecasting. 
    """

    # dimensions
    n = len(DA.xb)
    T = yo.values.shape[0]
    p = yo.values.shape[1] 

    # check dimensions
    if p!=DA.R.shape[0]:
        print("Error: bad R matrix dimensions")
        quit()

    # initialization
    class x_hat:
        part = np.zeros([T,DA.N,n])
        weights = np.zeros([T,DA.N])
        values = np.zeros([T,n])
        loglik = np.zeros([T])
        time = yo.time

    # EnKF and EnKS methods
    if (DA.method =='AnEnKF' or DA.method =='AnEnKS'):
        m_xa_part = np.zeros([T,DA.N,n])
        xf_part = np.zeros([T,DA.N,n])
        Pf = np.zeros([T,n,n])
        for k in tqdm(range(0,T)):
            # update step (compute forecasts)            
            if k==0:
                xf = np.random.multivariate_normal(DA.xb, DA.B, DA.N)
            else:
                xf, m_xa_part_tmp = DA.m(x_hat.part[k-1,:,:])
                m_xa_part[k,:,:] = m_xa_part_tmp         
            xf_part[k,:,:] = xf
            Ef = np.dot(xf.T,np.eye(DA.N)-np.ones([DA.N,DA.N])/DA.N)
            Pf[k,:,:] = np.dot(Ef,Ef.T)/(DA.N-1)
            # analysis step (correct forecasts with observations)          
            i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]            
            if (len(i_var_obs)>0):                
                eps = np.random.multivariate_normal(np.zeros(len(i_var_obs)),DA.R[np.ix_(i_var_obs,i_var_obs)],DA.N)
                yf = np.dot(DA.H[i_var_obs,:],xf.T).T
                SIGMA = np.dot(np.dot(DA.H[i_var_obs,:],Pf[k,:,:]),DA.H[i_var_obs,:].T)+DA.R[np.ix_(i_var_obs,i_var_obs)]
                SIGMA_INV = np.linalg.inv(SIGMA)
                K = np.dot(np.dot(Pf[k,:,:],DA.H[i_var_obs,:].T),SIGMA_INV)             
                d = yo.values[k,i_var_obs][np.newaxis]+eps-yf
                x_hat.part[k,:,:] = xf + np.dot(d,K.T)           
                # compute likelihood
                innov_ll = np.mean(yo.values[k,i_var_obs][np.newaxis]-yf,0)
                loglik = -0.5*(np.dot(np.dot(innov_ll.T,SIGMA_INV),innov_ll))-0.5*(n*np.log(2*np.pi)+np.log(np.linalg.det(SIGMA)))
            else:
                x_hat.part[k,:,:] = xf          
            x_hat.weights[k,:] = 1.0/DA.N
            x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0)
            x_hat.loglik[k] = loglik
        # end AnEnKF
        
        # EnKS method
        if (DA.method == 'AnEnKS'):
            for k in tqdm(range(T-1,-1,-1)):           
                if k==T-1:
                    x_hat.part[k,:,:] = x_hat.part[T-1,:,:]
                else:
                    m_xa_part_tmp = m_xa_part[k+1,:,:]
                    tej, m_xa_tmp = DA.m(np.mean(x_hat.part[k,:,:],0)[np.newaxis])
                    tmp_1 =(x_hat.part[k,:,:]-np.repeat(np.mean(x_hat.part[k,:,:],0)[np.newaxis],DA.N,0)).T
                    tmp_2 = m_xa_part_tmp - m_xa_tmp
                    Ks = 1.0/(DA.N-1)*np.dot(np.dot(tmp_1,tmp_2),inv_using_SVD(Pf[k+1,:,:],0.9999))                    
                    x_hat.part[k,:,:] = x_hat.part[k,:,:]+np.dot(x_hat.part[k+1,:,:]-xf_part[k+1,:,:],Ks.T)
                x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0)
        # end AnEnKS  
    
    # particle filter method
    elif (DA.method =='AnPF'):
        # special case for k=1
        k=0
        k_count = 0
        m_xa_traj = []
        weights_tmp = np.zeros(DA.N)
        xf = np.random.multivariate_normal(DA.xb, DA.B, DA.N)
        i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]
        if (len(i_var_obs)>0):
            # weights
            for i_N in range(0,DA.N):
                weights_tmp[i_N] = multivariate_normal.pdf(yo.values[k,i_var_obs].T,np.dot(DA.H[i_var_obs,:],xf[i_N,:].T),DA.R[np.ix_(i_var_obs,i_var_obs)])
            # normalization
            weights_tmp = weights_tmp/np.sum(weights_tmp)
            # resampling
            indic = resampleMultinomial(weights_tmp)
            x_hat.part[k,:,:] = xf[indic,:]         
            weights_tmp_indic = weights_tmp[indic]/sum(weights_tmp[indic])
            x_hat.values[k,:] = sum(xf[indic,:]*weights_tmp_indic[np.newaxis].T,0)
            # find number of iterations before new observation
            k_count_end = np.min(np.where(np.sum(1*~np.isnan(yo.values[k+1:,:]),1)>=1)[0])
        else:
            # weights
            weights_tmp = 1.0/N
            # resampling
            indic = resampleMultinomial(weights_tmp)
        x_hat.weights[k,:] = weights_tmp_indic
        
        for k in tqdm(range(1,T)):
            # update step (compute forecasts) and add small Gaussian noise
            xf, tej = DA.m(x_hat.part[k-1,:,:]) +np.random.multivariate_normal(np.zeros(xf.shape[1]),DA.B/100.0,xf.shape[0])        
            if (k_count<len(m_xa_traj)):
                m_xa_traj[k_count] = xf
            else:
                m_xa_traj.append(xf)
            k_count = k_count+1
            # analysis step (correct forecasts with observations)
            i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]
            if len(i_var_obs)>0:
                # weights
                for i_N in range(0,DA.N):
                    weights_tmp[i_N] = multivariate_normal.pdf(yo.values[k,i_var_obs].T,np.dot(DA.H[i_var_obs,:],xf[i_N,:].T),DA.R[np.ix_(i_var_obs,i_var_obs)])
                # normalization
                weights_tmp = weights_tmp/np.sum(weights_tmp)
                # resampling
                indic = resampleMultinomial(weights_tmp)            
                # stock results
                x_hat.part[k-k_count_end:k+1,:,:] = np.asarray(m_xa_traj)[:,indic,:]
                weights_tmp_indic = weights_tmp[indic]/np.sum(weights_tmp[indic])            
                x_hat.values[k-k_count_end:k+1,:] = np.sum(np.asarray(m_xa_traj)[:,indic,:]*np.tile(weights_tmp_indic[np.newaxis].T,(k_count_end+1,1,n)),1)
                k_count = 0
                # find number of iterations  before new observation
                try:
                    k_count_end = np.min(np.where(np.sum(1*~np.isnan(yo.values[k+1:,:]),1)>=1)[0])
                except ValueError:
                    pass
            else:
                # stock results
                x_hat.part[k,:,:] = xf
                x_hat.values[k,:] = np.sum(xf*weights_tmp_indic[np.newaxis].T,0)
            # stock weights
            x_hat.weights[k,:] = weights_tmp_indic   
        # end AnPF

    # error
    else :
        print("Error: choose DA.method between 'AnEnKF', 'AnEnKS', 'AnPF' ")
        quit()
    return x_hat
