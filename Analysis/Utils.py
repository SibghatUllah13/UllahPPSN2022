import numpy as np
import pandas as pd
import os
import sys
import pickle5 as pickle

def __load__results__ (path):
    
    """
    Load results for a specific robustness formulation, e.g., mini-max, and dimensionality, e.g., 10.
    
    Arguments:
    ------------
    path: string
        indicates the path of the folder where the results of a specific robustness formulation and dimensionality are saved.
        
    Note:
       The folder in question (for which the path is to be specified) has 15 sub-folders inside it corresponding
       to 15 independent runs of KB-RO. Furthermore, inside each sub-folder, we have 20 '*__#.pickle' files, where
       * refers to the test case number (retrieved from settings.csv or settings__10.csv depending upon dimensionality),
       and # refers to the independent run.
    
    """
    
    paths = [path + str (occassion) for occassion in np.arange (1,16)] # for independent runs
    extract_files = lambda p : os.listdir (p)
    files = [extract_files (case) for case in paths]
    return [read_results (path, file) for path, file in zip (paths, files)]


def read_results (path, folder):
    
    """
    Read results from a specific folder -- for a specific independent run
    
    Arguments:
    ----------
    path: string
        indicates the path of the sub-folder (corresponding to a specific run)
    folder: array like
        indicates the name of all '*__#.pickle' files inside the folder of a specific run
    
    Note:
        This function is called by the function '__load__results__'. 
    """
    res = []
    for file in folder:
        res.append ((file, pd.read_pickle (path + "\\" + file)))
    return res

def __retrieve__case__data__ (case, results):
    
    """
    Retrieve optimization data (optimal functional values per iteration) as well as cpu time for a particular case of interest
    
    Arguments:
    ----------
    
    case: integer
        Identifies the case number for which the results are to be retrieved
        
    results: nested list (the output of the function __load__results__)
    
    Note: The data to bre trieved includes optimization data (robust optimal function values) and cpu time per iteration.
    for the particular case number for 15 independent runs. 
    Therefore, the output of this function is a nested list whose each element contains two dataframes for optimization data and
    time.
    """
    occ_opt_data = []
    occ_time = []
    for occ in results:
        for test in occ:
            if test [0].split ('__')[0] == case:
                f_values = [float(itera[0]) for itera in test[1][0]]
                time = [itera for itera in test[1][1]]
                occ_opt_data.append (f_values)
                occ_time.append(np.array(time).cumsum())
    occ_opt_data = pd.DataFrame(occ_opt_data).T
    occ_opt_data.columns = ['Run'+ str(occ) for occ in np.arange(1,16)]
    occ_time = pd.DataFrame(occ_time).T
    occ_time.columns = ['Run'+ str(occ) for occ in np.arange(1,16)]
    
    return [occ_opt_data, occ_time]

def __load__gt__ (path, settings, rf = 'mini-max', dim = 2):
    
    """
    Load Ground Truth Data for a specific robustness formulation, e.g., mini-max, and dimensionality, e.g., 10.
    
    Arguments:
    ------------
    path: string
            indicates the path of the folder where the ground truth data of a specific robustness formulation and dimensionality are saved.
    
    settings: dataframe (csv file)
            indicates the dataframe of all the unique test case settings. For the 10D case, settings__10D dataframe is used.
    
    dim: integer
            Determines the dimensionality of the problem.
        
    rf: string
            Gives the robustness formulation 
   
    """
    index = settings.loc [(settings['rf'] == rf) & (settings['dimensionality'] == dim)].index
    index = [ case[4:] for case in index]
    index = [case + ".pickle" for case in index]
    temp = []
    for case in index:
        with open (path + case, "rb") as input_file:
            temp.append (pickle.load (input_file))
    
    return temp


def __compute__quality__ (case, res, opt_data):
    
    """
    Compute the quality of the optimal solution based on NMAE for a particular test case
    
    Arguments:
    ------------
    case: integer
        Identifies the case number for which the results are to be retrieved
        
    res: array like (the result of the function __load__gt__)
        ground truth data/res
    
    opt_data: array like (the result of the function __retrieve__case__data__)
        optimization data for the particular case for which the quality of the optimal solution will be determined
    
    """
    for i in range (len(res)):
        if res[i][0] == case:
            gt = res [i][1][0]
            quality = np.minimum.accumulate (__rmae__ (gt, opt_data [i][0]))
        
    return quality

def __extract__information__ (iter_data, opt_data):
    
    """
    Extract function information
    
    Arguments:
    -----------
    
    iter_data: array like
        indicates the checkpoints (the iterations) at which the quality of the optimal solutions is to be determined
        
    opt_data: optimization data (the result of the function __compute__quality__)
    """
    temp = []
    for i in range (len(iter_data)):
        temp_1 = []
        for j in range (len(opt_data)):
            temp_1.append (opt_data [j].iloc[iter_data[i],:]) 
        temp.append (temp_1)
        del temp_1
    return np.mean(np.array(temp), axis = 0).reshape(300,)

def __data__indices (time, mmr, mmrr, ebr, dbr, cr):
    
    """
    For each setting of the cpu time, determine the corresponding index in the optimization data where the data is to be retrieved
    
    Arguments:
    ---------
    
    time: array like
        contains the cpu time settings for which the quality of the solution is to be reported
        
    mmr: array like
        contains the optimization data for the mmr for all three settings of dimensionality
        
    mmrr: array like
        contains the optimization data for the mmrr for all three settings of dimensionality
        
    ebr: array like
        contains the optimization data for the ebr for all three settings of dimensionality
        
    dbr: array like
        contains the optimization data for the dbr for all three settings of dimensionality
        
    cr: array like
        contains the optimization data for the cr for all three settings of dimensionality
    """
    
    time_2D, time_5D, time_10D = time 
    mmr_2D, mmr_5D, mmr_10D = mmr
    mmrr_2D, mmrr_5D, mmrr_10D = mmrr
    ebr_2D, ebr_5D, ebr_10D = ebr
    dbr_2D, dbr_5D, dbr_10D = dbr
    cr_2D, cr_5D, cr_10D = cr
    
    time_ind_mmr_2D = __find__indices__ (time_2D, mmr_2D, 1, 21)
    time_ind_mmr_5D = __find__indices__ (time_5D, mmr_5D, 21, 41)
    time_ind_mmr_10D = __find__indices__ (time_10D, mmr_10D, 201, 221)

    time_ind_mmrr_2D = __find__indices__ (time_2D, mmrr_2D, 41, 61)
    time_ind_mmrr_5D = __find__indices__ (time_5D, mmrr_5D, 61, 81)
    time_ind_mmrr_10D = __find__indices__ (time_10D, mmrr_10D, 221, 241)

    time_ind_ebr_2D = __find__indices__ (time_2D, ebr_2D, 81, 101)
    time_ind_ebr_5D = __find__indices__ (time_5D, ebr_5D, 101, 121)
    time_ind_ebr_10D = __find__indices__ (time_10D, ebr_10D, 241, 261)
    
    time_ind_dbr_2D = __find__indices__ (time_2D, dbr_2D, 121, 141)
    time_ind_dbr_5D = __find__indices__ (time_5D, dbr_5D, 141, 161)
    time_ind_dbr_10D = __find__indices__ (time_10D, dbr_10D, 261, 281)
    
    time_ind_cr_2D = __find__indices__ (time_2D, cr_2D, 161, 181)
    time_ind_cr_5D = __find__indices__ (time_5D, cr_5D, 181, 201)
    time_ind_cr_10D = __find__indices__ (time_10D, cr_10D, 281, 301)
    
    return [time_ind_mmr_2D, time_ind_mmr_5D, time_ind_mmr_10D,
           time_ind_mmrr_2D, time_ind_mmrr_5D, time_ind_mmrr_10D,
           time_ind_ebr_2D, time_ind_ebr_5D, time_ind_ebr_10D,
           time_ind_dbr_2D, time_ind_dbr_5D, time_ind_dbr_10D,
           time_ind_cr_2D, time_ind_cr_5D, time_ind_cr_10D]


def __find__indices__ (time, res, range_l, range_u):
    
    """
    Find indices for each case where the time threshold is satisfied
    
    Works in combination with the above function
    """
    indices = []
    for i in range (len(res)):
        temp = []
        for j in range(len(res[i][1].columns)):
            if (len([i for i, x in enumerate((res[i][1].iloc[:,j] > time)) if x])) > 0:
                temp.append ([i for i, x in enumerate((res[i][1].iloc[:,j] > time)) if x][0])
            else:
                temp.append (res[i][1].shape[0]-1)
        indices.append (temp)
    indices = pd.DataFrame(indices)
    indices.index = ['Case'+str(case) for case in range (range_l, range_u)]
    indices.columns = ['Run' + str(occ) for occ in range(1,16)]
    
    return indices

def __extract__function__information (time_index, opt_data):
    
    """
    Extract the function information/value based on the time index --- as soon as the time threshold is passed, stop and retrieve 
    the best quality function value until that time
    
    This function is to be run for a particular choice of RF and dimensionality 
    
    Arguments:
    ---------
    
    time_index: dataframe
            contains information about the time index for each test case and each run 
    
    opt_data: Optimization data retrieved from the function __compute__quality__ 
    
    """
    case = []
    for i in range(len(time_index)): #for each case
        run = []
        for j in range(time_index.shape[1]): # for each run
            run.append (opt_data[i].iloc[time_index.iloc[i,j],j])
        case.append (run)
        del run
    return pd.DataFrame (case)

def __extract__time__information (target, res, data, dim):
    
    """
    Extract the c.p.u time for each test case (aggregated over multiple multiple fixed values)
    
    Arguments:
    ---------
    
    target: array like
        target values/thresholds for assessing the quality of the solutions
        
    res: Optimization data retrieved from the function __compute__quality__
    
    data: data received from the function __load__results__
    
    dim: integer
        indicates the dimensionality of the problem
    """
    case = []
    for i in range (len(res)): #for each test case
        runs = []
        for j in range (len(res[i].columns)): #for each independent run
            time = []
            for k in range (len(target)): #for each fixed target
                if (len (np.where(res[i].iloc[:,j] < target[k])[0])) > 0: #if such target has been achieved by BO
                    temp = data[i][1].iloc[np.where(res[i].iloc[:,j] < target[k])[0][0],j] #find cpu time for such target
                else:
                    temp = dim * data[i][1].iloc[-1,j] #Otherwise, penalize the cpu time with the maximum cpu time for that runs
                time.append (temp) 
            runs.append (np.mean (time)) #averaged across ten target values
        case.append (runs) 
    return np.array (case)



settings = pd.read_csv ('C:\\Users\\sefi\\Cost of Robustness\\Results\\Graphs\\Graphs - Latest\\Settings.csv', index_col = 0)
settings_10 = pd.read_csv ('C:\\Users\\sefi\\Cost of Robustness\\Results\\Graphs\\Graphs - Latest\\Settings.csv', index_col = 0)