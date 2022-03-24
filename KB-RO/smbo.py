#!/usr/bin/env python
# coding: utf-8

# In[9]:


import time
from mpi4py import MPI
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
import lib
import doe
import new_sample


def mb_optimisation (optimal_solution, n_iters, x0, test_problem, dim = 5, 
                    robustness = "mini-max", 
                    nl = 0.05, epsilon = 1e-7):
    
    """ 
    Kriging-based Robust Optimization (Implementation of Algorithm 1)
    
    Arguments:
    -------------
        optimal_solution: The optimal solution/ ground truth for the case being considered
    
        n_iters: integer.
            Number of iterations to run the Algorithm 1.
            
        x0: Array-like, shape = [n_pre_samples, n_params].
            Locations of initial data points. 
            
        test_problem: string
            Indicates the test problem, e.g., F2.
            
        dim: integer
            Determines the dimensionality of the problem.
            
        nl: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
            
        robustness: string
            Gives the robustness formulation to be employed.
        
    """
    res = []
    cpu_time = []
    
    start_time = time.time()
    
    
    optimal_function_value, optimal_parameters = optimal_solution #ground truth
    
    
    n_params = dim
    x_list = []
    y_list = []
    
    for params in x0:
        x_list.append (params)
        y_list.append (getattr(doe, test_problem)(params))
        
    xp = np.array (x_list)
    yp = np.array (y_list)
    
    
    for n in range(n_iters):
        
        init_time = time.time()
        
        
        kernel = Matern (nu = 3/2) 
        model = GaussianProcessRegressor (kernel = kernel, 
                                         n_restarts_optimizer = 5, normalize_y = True, random_state = 0).fit(xp, yp)
        
        
        
        
        opt = lib.__find__robust__optimum__ (Model = model,  Dimensionality = dim , 
                                             Robustness = robustness, Noise_Level = nl) 
        
        res.append (opt)
        
       
        new_location = new_sample.__sample__next__point__ (acquisition_func = new_sample.__EIC__, Model = model,
                                           robust_optimum = opt [0], Robustness = robustness,
                                           Dimensionality = dim, Noise_Level = nl)
        
        
        
        new_response = getattr (doe, test_problem)(new_location)
        
        
        
        x_list.append(new_location)
        y_list.append(new_response)
        
        
        xp = np.array(x_list)
        yp = np.array(y_list)
    
          
        cpu_time.append (time.time() - init_time)
        
        
    return res , cpu_time


# ## Run Kriging-based Optimization

# In[3]:


def run (Combination, Occassion = 1):
    
    """
    Run Kriging-based robust optimization for a certain case (determined by Combination), and 
    independent run (determined by Occassion)
    
    ------------------
    Combination: integer
            determines the test case 
    
    Occassion: integer
            determines the independent run
    """
    
    # load the settings from Settings.csv (for 2 and 5D) or Settings__10.csv (for 10D problems) 
    rf, dimensionality, noise_level, test_problem = pd.read_csv ('Settings.csv', 
                                                               index_col = 0).loc ["Comb" + str (Combination)]
    
    # set the Computational budget and initial sample size
    N , N_Iters = lib.__set__budget__ (dimensionality)
    
    # set the bounds for the search space
    xlimits =  np.array([-5,5] * dimensionality).reshape(dimensionality, -1)

    print ('Global Parameters Set..............')
    
    print ('Run the experiment')
    
    # Find initial design locations using doe
    X = doe.__LHS__(n_obs = N , xlimits = xlimits, 
                        random_state = Occassion + 1 , criterion = 'm' )
    
    # load ground truth data, for a particular robustness formulation and dimensionality
    # essentially a numpy array with 20 elements whose each element contains case id and the ground truth data for that id  
    
    GT_Data = np.load ('GT__5D__CR.npy', allow_pickle = True)
    
        
    optimal_solution = [case for case in GT_Data if case[0] == Combination][0][1]
    
    # run the Kriging-based robust optimization    
    Temp = mb_optimisation ( optimal_solution = optimal_solution, n_iters = N_Iters , x0 = X,
                           test_problem = test_problem, dim = dimensionality, robustness = rf, nl = noise_level)
    
    
    
    with open (str (Combination)+ "__" + str(Occassion)+ '.pickle', 'wb') as handle:
        pickle.dump (Temp, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    print('Test Case {} is completed'.format (Combination))   


# ## Parallelize the Operation

# In[ ]:


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
Occassion = int (sys.argv[1])

if rank == 0:
    
    # load ground truth data, for a particular robustness formulation and dimensionality
    # essentially a numpy array with 20 elements whose each element contains case id and the ground truth data for that id  
    
    
    GT_Data = np.load ('GT__5D__CR.npy', allow_pickle = True)
    
    data = [case[0] for case in GT_Data]
    
else:
    
    data = None

    
data = comm.scatter (data, root = 0)

res = run (data, Occassion)    

