import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def mini__max__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute Robust Optimum based on Mini-Max Robustness
    
    ----------
        
    Point: array-like, 
            The point in the search domain for which the effective Kriging prediction needs to be computed.
            
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated samples.
            
    Dimensionality: integer
            Determines the dimensionality of the problem.
            
    Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    seed = 0
    Range = 10
    worst_noise = None
    worst_function_value = np.inf
    Bou = np.array([- (Noise_Level * Range), Noise_Level * Range] * Dimensionality).reshape(Dimensionality, -1)
    n_restarts = int (Dimensionality / 2) # fix the number of restarts
    n_params = Dimensionality
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(Bou)[:,0], np.array(Bou)[:,1], size = (n_restarts, n_params)):
        
        res = minimize (fun =  noisy_prediction,
                       x0 = np.atleast_2d (starting_point),
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 10 * n_params},
                       args = (Point , Model))
        
        if res.fun < worst_function_value:
            worst_function_value = res.fun
            worst_noise = res.x
    
    return Model.predict (np.atleast_2d (worst_noise + Point))

def mini__max__regret__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute Robust Optimum based on Mini-Max Regret Robustness
    
    ----------
        
    Point: array-like, 
            The point in the search domain for which the effective Kriging prediction needs to be computed.
            
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated samples.
            
    Dimensionality: integer
            Determines the dimensionality of the problem.
            
    Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    seed = 0
    Range = 10
    worst_noise = None
    worst_function_value = np.inf
    Bou = np.array([- (Noise_Level * Range), Noise_Level * Range] * Dimensionality).reshape(Dimensionality, -1)
    n_restarts = int (Dimensionality / 2) # fix the number of restarts
    n_params = Dimensionality
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(Bou)[:,0], np.array(Bou)[:,1], size = (n_restarts, n_params)):
        
        res = minimize (fun =  __regret__,
                       x0 = np.atleast_2d (starting_point),
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 10 * n_params},
                       args = (Point , Model, Dimensionality, Noise_Level))
        
        if res.fun < worst_function_value:
            worst_function_value = res.fun
            worst_noise = res.x
    
    
    
    return Model.predict (np.atleast_2d (worst_noise + Point))   
        
    
def __regret__ (Noise, Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute the regret that may result from making non-optimal decision
    """
    
    A = Model.predict (np.atleast_2d (Point + Noise))
    
    
    seed = 0
    Range = 10
    best_x = None
    best_function_value = np.inf
    n_params = Dimensionality
    n_restarts = 1 #int (Dimensionality/2) # fix the number of restarts
    Bou = np.array([-5 + (Noise_Level * Range), 5 - (Noise_Level * Range)] * Dimensionality).reshape(Dimensionality, -1)
    # Bounds to find the robust optimum, the robust optimum is found in a restricted search space of the original domain
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(Bou)[:,0], 
                                            np.array(Bou)[:,1], size = (n_restarts, n_params)):
        res = minimize(fun = prediction,
                       x0 = np.atleast_2d (starting_point),
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 5 * n_params},
                       args = (Noise, Model))
        
        if res.fun < best_function_value:
            best_function_value = res.fun
            best_x = res.x
     
    B = best_function_value
    
    return -1 * (A-B)

def noisy_prediction (Noise, Point, Model):
    
    """
    For a Given point in the design space --- X --- and a Kriging model of the objective function,
    Returns negative Kriging Prediction under additive Noise (negative since the function is to be minimized).
    
    Noise: array-like, 
            The noise: Delta_x 
            
    Point: array-like, 
            The point in the search space for which the effective Kriging prediction needs to be computed.  
            
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated samples.
    """
    
    prediction = Model.predict (np.atleast_2d (Point + Noise))
    return -1 * prediction

def prediction (Point, Noise, Model):
    
    """
    For a Given point in the design space --- X --- and a Kriging model of the objective function,
    Returns Kriging Prediction under additive Noise.
    
    Noise: array-like, 
            The noise: Delta_x 
            
    Point: array-like, 
            The point in the search space for which the effective Kriging prediction needs to be computed.  
            
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated samples.
    """
    return Model.predict (np.atleast_2d (Point + Noise))
    

def expectation__based__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute Robust Kriging response of the function based on Expectation-based Robustness
    
    ----------
        
    Point: array-like, 
            The point in the search domain for which the effective Kriging prediction needs to be computed.
            
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated samples.
            
    Dimensionality: integer
            Determines the dimensionality of the problem.
            
    Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    constant = 100
    Range = 10
    Bou = np.array([- (Noise_Level * Range), Noise_Level * Range] * Dimensionality).reshape(Dimensionality, -1)
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = (constant * Dimensionality, Dimensionality))
    summation = np.array ([Model.predict (np.atleast_2d (Point + sample)) for sample in samples])
    
    return np.mean(summation)


def dispersion__based__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute Robust Kriging response of the function based on Dispersion-based Robustness
    
    ----------
        
    Point: array-like, 
            The point in the search domain for which the effective Kriging prediction needs to be computed.
            
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated samples.
            
    Dimensionality: integer
            Determines the dimensionality of the problem.
            
    Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    
    constant = 100
    Range = 10
    Bou = np.array([- (Noise_Level * Range), Noise_Level * Range] * Dimensionality).reshape(Dimensionality, -1)
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = (constant * Dimensionality, Dimensionality))
    summation = np.array ([Model.predict (np.atleast_2d (Point + sample)) for sample in samples])
    
    mean = np.mean (summation)
    
    dispersion = np.sqrt ( np.sum(np.square(mean - summation)) / ((constant * Dimensionality) - 1))
    
    return dispersion


def composite__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute Robust Kriging response of the function based on Composite Robustness
    
    ----------
        
    Point: array-like, 
            The point in the search domain for which the effective Kriging prediction needs to be computed.
            
    Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated samples.
            
    Dimensionality: integer
            Determines the dimensionality of the problem.
            
    Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    
    constant = 100
    Range = 10
    Bou = np.array([- (Noise_Level * Range), Noise_Level * Range] * Dimensionality).reshape(Dimensionality, -1)
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = (constant * Dimensionality, Dimensionality))
    summation = np.array ([Model.predict (np.atleast_2d (Point + sample)) for sample in samples])
    
    mean = np.mean (summation)
    
    dispersion = np.sqrt ( np.sum(np.square(mean - summation)) / ((constant * Dimensionality) - 1))
    
    return mean + dispersion





    
    
    
    
    
    
    