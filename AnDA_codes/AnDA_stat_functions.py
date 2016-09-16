#!/usr/bin/env python

""" AnDA_stat_functions.py: Collection of statistical functions used in AnDA. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np

def AnDA_RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
 
    return np.sqrt(np.mean((a-b)**2))

def normalise(M):
    """ Normalize the entries of a multidimensional array sum to 1. """

    c = np.sum(M);
    # Set any zeros to one before dividing
    d = c + 1*(c==0);
    M = M/d;
    return M;

def mk_stochastic(T):
    """ Ensure the matrix is stochastic, i.e., the sum over the last dimension is 1. """

    if len(T.shape) == 1:
        T = normalise(T);
    else:
        n = len(T.shape);
        # Copy the normaliser plane for each i.
        normaliser = np.sum(T,n-1);
        normaliser = np.dstack([normaliser]*T.shape[n-1])[0];
        # Set zeros to 1 before dividing
        # This is valid since normaliser(i) = 0 iff T(i) = 0

        normaliser = normaliser + 1*(normaliser==0);
        T = T/normaliser.astype(float);
    return T;

def sample_discrete(prob, r, c):
    """ Sampling from a non-uniform distribution. """

    # this speedup is due to Peter Acklam
    cumprob = np.cumsum(prob);
    n = len(cumprob);
    R = np.random.rand(r,c);
    M = np.zeros([r,c]);
    for i in range(0,n-1):
        M = M+1*(R>cumprob[i]);    
    return int(M)

def resampleMultinomial(w):
    """ Multinomial resampler. """

    M = np.max(w.shape);
    Q = np.cumsum(w,0);
    Q[M-1] = 1; # Just in case...
    i = 0;
    indx = [];
    while (i<=(M-1)):
        sampl = np.random.rand(1,1);
        j = 0;
        while (Q[j]<sampl):
            j = j+1;
        indx.append(j);
        i = i+1
    return indx

def inv_using_SVD(Mat, eigvalMax):
    """ SVD decomposition of Matrix. """
    
    U,S,V = np.linalg.svd(Mat, full_matrices=True);
    eigval = np.cumsum(S)/np.sum(S);
    # search the optimal number of eigen values
    i_cut_tmp = np.where(eigval>=eigvalMax)[0];
    S = np.diag(S);
    V = V.T;
    i_cut = np.min(i_cut_tmp)+1;
    U_1 = U[0:i_cut,0:i_cut];
    U_2 = U[0:i_cut,i_cut:];
    U_3 = U[i_cut:,0:i_cut];
    U_4 = U[i_cut:,i_cut:];
    S_1 = S[0:i_cut,0:i_cut];
    S_2 = S[0:i_cut,i_cut:];
    S_3 = S[i_cut:,0:i_cut];
    S_4 = S[i_cut:,i_cut:];
    V_1 = V[0:i_cut,0:i_cut];
    V_2 = V[0:i_cut,i_cut:];
    V_3 = V[i_cut:,0:i_cut];
    V_4 = V[i_cut:,i_cut:];
    tmp1 = np.dot(np.dot(V_1,np.linalg.inv(S_1)),U_1.T);
    tmp2 = np.dot(np.dot(V_1,np.linalg.inv(S_1)),U_3.T);
    tmp3 = np.dot(np.dot(V_3,np.linalg.inv(S_1)),U_1.T);
    tmp4 = np.dot(np.dot(V_3,np.linalg.inv(S_1)),U_3.T);
    inv_Mat = np.concatenate((np.concatenate((tmp1,tmp2),axis=1),np.concatenate((tmp3,tmp4),axis=1)),axis=0);
    tmp1 = np.dot(np.dot(U_1,S_1),V_1.T);
    tmp2 = np.dot(np.dot(U_1,S_1),V_3.T);
    tmp3 = np.dot(np.dot(U_3,S_1),V_1.T);
    tmp4 = np.dot(np.dot(U_3,S_1),V_3.T);
    hat_Mat = np.concatenate((np.concatenate((tmp1,tmp2),axis=1),np.concatenate((tmp3,tmp4),axis=1)),axis=0);
    det_inv_Mat = np.prod(np.diag(S[0:i_cut,0:i_cut]));   
    return inv_Mat;
    
