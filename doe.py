import math
import numpy as np
import pandas as pd
import bbobbenchmarks as bn
from smt.sampling_methods import LHS

""" Category 1: F1-F5 (Separable Functions) """

"""F1"""
def F1(X):
    f = bn.F1()
    X = np.array(X)
    return f(X)

"""F2"""
def F2(X):
    f = bn.F2()
    X = np.array(X)
    return f(X)

"""F3"""
def F3(X):
    f = bn.F3()
    X = np.array(X)
    return f(X)

"""F4"""
def F4(X):
    f = bn.F4()
    X = np.array(X)
    return f(X)

"""F5"""
def F5(X):
    f = bn.F5()
    X = np.array(X)
    return f(X)

""" Category 2: F6-F9 (Functions with low or moderate conditioning) """

"""F6"""
def F6(X):
    f = bn.F6()
    X = np.array(X)
    return f(X)

"""F7"""
def F7(X):
    f = bn.F7()
    X = np.array(X)
    return f(X)

"""F8"""
def F8(X):
    f = bn.F8()
    X = np.array(X)
    return f(X)

"""F9"""
def F9(X):
    f = bn.F9()
    X = np.array(X)
    return f(X)

""" Category 3: F10-F14 (Functions with high conditioning and unimodal) """


"""F10"""
def F10(X):
    f = bn.F10()
    X = np.array(X)
    return f(X)

"""F11"""
def F11(X):
    f = bn.F11()
    X = np.array(X)
    return f(X)

"""F12"""
def F12(X):
    f = bn.F12()
    X = np.array(X)
    return f(X)

"""F13"""
def F13(X):
    f = bn.F13()
    X = np.array(X)
    return f(X)

"""F14"""
def F14(X):
    f = bn.F14()
    X = np.array(X)
    return f(X)

""" Category 4: F15-F19 (Multi-modal functions with adequate global structure) """

"""F15"""
def F15(X):
    f = bn.F15()
    X = np.array(X)
    return f(X)

"""F16"""
def F16(X):
    f = bn.F16()
    X = np.array(X)
    return f(X)

"""F17"""
def F17(X):
    f = bn.F17()
    X = np.array(X)
    return f(X)

"""F18"""
def F18(X):
    f = bn.F18()
    X = np.array(X)
    return f(X)

"""F19"""
def F19(X):
    f = bn.F19()
    X = np.array(X)
    return f(X)

""" Category 5: F20-F24 (Multi-modal functions with weak global structure) """

"""F20"""
def F20(X):
    f = bn.F20()
    X = np.array(X)
    return f(X)

"""F21"""
def F21(X):
    f = bn.F21()
    X = np.array(X)
    return f(X)

"""F22"""
def F22(X):
    f = bn.F22()
    X = np.array(X)
    return f(X)


def F23(X):
    """
    F23
    """
    f = bn.F23()
    X = np.array(X)
    return f(X)


def F24(X):
    """
    F24
    """
    f = bn.F24()
    X = np.array(X)
    return f(X)



def __LHS__ (n_obs, xlimits, random_state = 0, criterion = 'm'):
    
    """
    Latin HyperCube Sampling Design of Experiment
    Arguments:
    ----------
    n_obs: integer ------------ To specify the number of samples -------------
    xlimits: A numpy array with shape: [n_params, 2] to specify the box constraints
    random_state : integer ---------------- To specify the seed of the random generator ----------------
    criterion: ----- A string character to specify the criterion for the LHS ----------------------
    """ 
    sampling = LHS (xlimits = xlimits, random_state = random_state, criterion = criterion)
    return sampling(n_obs)

def __DoE__ (fun, n_obs, xlimits, random_state = 0, criterion = 'm'):
    
    """
    Generate Initial design data with LHS scheme
    Arguments:
    ----------
    fun: The function to be evaluated 
    n_obs: integer  ------------ To specify the number of samples -------------
    xlimits: A numpy array with shape: [n_params, 2] to specify the box constraints
    random_state : integer ---------------- To specify the seed of the random generator ----------------
    criterion: ----- A string character to specify the criterion for the LHS ----------------------
    """ 
    responses = []
    samples = __LHS__ (n_obs, xlimits, random_state = 0, criterion = 'm')
    for location in samples:
        responses.append(fun(location))
    return samples, responses



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

