#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mpi4py import MPI
from itertools import product
import sys
import pickle
import time
import numpy as np
import pandas as pd
import ground_truth

def run (Combination):
    
    """
    Run the entire code to compute ground truth
    """
    start_time = time.time()
    
    # specify the file which contains the settings 
    rf, dimensionality, noise_level, test_problem = pd.read_csv ('Settings__10D.csv', 
                                                               index_col = 0).loc["Comb" + str(Combination)]
    
    
    print('Test Case {} has started'.format(Combination))
    print ('Compute Ground Truth')
    
    optimal_solution = ground_truth.__compute__ground__truth__ (test_problem, dimensionality, noise_level, rf)
    
    res = Combination, optimal_solution , time.time() - start_time
    
    with open (str(Combination) + '.pickle', 'wb') as handle:
        pickle.dump (res, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    print('Test Case {} is completed'.format(Combination))


# # Initialize the Communicator

# In[ ]:


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# In[ ]:


if rank == 0:
    
    # specify the .npy files which contains the ids of the cases to be run
    # e.g., [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    data = np.load ('10D_Cases.npy', allow_pickle = True)
    
else:
    
    data = None

    
data = comm.scatter (data, root = 0)

res = run (data)

