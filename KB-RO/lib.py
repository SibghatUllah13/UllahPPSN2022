import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import robust_lib

def __find__robust__optimum__ (Model, Dimensionality, Robustness, Noise_Level):
    
    """ 
    Find the robust optimum in step 3 of the Algorithm 1.
    
    Arguments:
    ----------
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated function responses.
            
    Dimensionality: integer
            Determines the dimensionality of the problem.
            
    Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
            
    Robustness: string
            Gives the robustness formulation to be employed.
    """
    
    seed = 0
    Range = 10
    best_x = None
    best_function_value = np.inf
    n_params = Dimensionality
    n_restarts = int (Dimensionality / 2) # fix the number of restarts
    Bou = np.array([-5 + (Noise_Level * Range), 5 - (Noise_Level * Range)] * Dimensionality).reshape(Dimensionality, -1)
    # Bounds to find the robust optimum, the robust optimum is found in a restricted search space of the original domain
    
    np.random.seed (seed)
    for starting_point in np.random.uniform (np.array(Bou)[:,0], 
                                            np.array(Bou)[:,1], size = (n_restarts, n_params)):
        
        res = minimize (fun = __effective__Kriging__value__,
                       x0 = np.atleast_2d (starting_point),
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 10 * n_params},
                       args = (Model, Robustness, Dimensionality, Noise_Level))
        
        if res.fun < best_function_value:
            best_function_value = res.fun
            best_x = res.x
    
    
    return best_function_value, best_x


def __effective__Kriging__value__ (Point, Model, Robustness, Dimensionality, Noise_Level):
    
    """
    Compute Effective/Robust Kriging Value based on the Chosen Robustness Criterion.
    
    For a Given point X in the search space, Robustness Criterion, and a Kriging model of the objective function, 
    Compute the Effective Kriging Value for X in the face of uncertainty.
    
    ----------
    Point: array-like, 
            The point in the search space for which the effective Kriging prediction needs to be computed.
            
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated function responses.
            
    Dimensionality: integer
            Determines the dimensionality of the problem.
            
    Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
            
    Robustness: string
            Gives the robustness formulation to be employed. 
    """
    
    
    if Robustness == "mini-max":
        K__eff = robust_lib.mini__max__robustness (Point, Model, Dimensionality, Noise_Level)
        
    elif Robustness == "mini-max-regret":
        K__eff = robust_lib.mini__max__regret__robustness (Point, Model, Dimensionality, Noise_Level)
        
    elif Robustness == "expectation":
        K__eff = robust_lib.expectation__based__robustness (Point, Model, Dimensionality, Noise_Level)
        
    elif Robustness == "dispersion":
        K__eff = robust_lib.dispersion__based__robustness (Point, Model, Dimensionality, Noise_Level)
        
    elif Robustness == "composite":
        K__eff = robust_lib.composite__robustness (Point, Model, Dimensionality, Noise_Level)
        
    return K__eff 


def __set__budget__ (dim):
    
    """
    Set the computational budget based on dimensionality 
    
    The computational budget includes the number of samples for the initial design data
    and the total number of iterations for Algorithm 1.
    
    --------------
    dim: integer, from the set {2,5,10}
    Describes the dimensionality of the problem
    """
    return (2 * dim, 50 * dim)



        
      
 
        
        

