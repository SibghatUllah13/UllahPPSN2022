import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats import norm
import doe

def __compute__ground__truth__ (test_problem = "F2", dim = 5, noise_level = 0.05, robustness = "mini-max"):
    
    """
    Compute ground truth for a particular test case
    Arguments:
    ------------
    test_problem: string
            Indicates the test problem/function chosen .
        dim: integer
            Determines the dimensionality of the problem.
        noise_level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
        robustness: string
            Gives the robustness formulation to be employed.
    """
    
    test_problem = getattr (doe, test_problem)
    seed = 0
    Range = 10
    best_x = None
    best_function_value = np.inf
    n_params = dim
    n_restarts = int (dim / 2) # fix the number of restarts
    Bou = np.array([-5 + (noise_level * Range), 5 - (noise_level * Range)] * dim).reshape(dim, -1)
    # Bounds to find the optimum, the optimum is found in a restricted search space of the original domain due to additive noise
    
    np.random.seed (seed)
    for starting_point in np.random.uniform (np.array(Bou)[:,0], 
                                            np.array(Bou)[:,1], size = (n_restarts, n_params)):
        
        res = minimize(fun = __effective__function__value__,
                       x0 = starting_point,
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 50 * n_params},
                       args = (test_problem, dim, noise_level, robustness))
        
        if res.fun < best_function_value:
            best_function_value = res.fun
            best_x = res.x
    
    
    return best_function_value, best_x


def __effective__function__value__ (Point, test_problem, dim, noise_level, robustness):
    
    """
    Compute Effective Function Value based on the Chosen Robustness Criterion.
    -----------
    For a Given point X in the search space, the Robustness Criterion, and an objective function, 
    Compute the Effective Function Value for X in the face of uncertainty.
    ----------
        Point: array-like
            The point in the search space for which the effective function value needs to be computed.
            
        test_problem: 
            Indicates the test problem/objective function.
            
        robustness: string
            Gives the robustness formulation to be employed.
            
        dim: integer
            Determines the dimensionality of the problem.
            
        noise_level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    
    
    if robustness == "mini-max":
        f = mini__max__robustness (Point, test_problem, dim, noise_level)
        
    elif robustness == "mini-max-regret":
        f = mini__max__regret__robustness (Point, test_problem, dim, noise_level)
        
    elif robustness == "expectation":
        f = expectation__based__robustness (Point, test_problem, dim, noise_level)
        
    elif robustness == "dispersion":
        f = dispersion__based__robustness (Point, test_problem, dim, noise_level)
        
    elif robustness == "composite":
        f = composite__robustness (Point, test_problem, dim, noise_level)
        
    return f 

def mini__max__robustness(Point, test_problem, Dimensionality, Noise_Level):
    
    """
    Compute Effective Function Value based on the Mini-Max Robustness
    ----------
       Point: array-like
            The point in the search space for which the effective function value needs to be computed.
            
        test_problem: 
            Indicates the test problem/objective function.
            
        
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
    n_restarts =  int (Dimensionality / 2) # fix the number of restarts
    n_params = Dimensionality
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(Bou)[:,0], np.array(Bou)[:,1], size = (n_restarts, n_params)):
        res = minimize (fun =  noisy_evaluation,
                       x0 = starting_point,
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 50 * n_params},
                       args = (Point , test_problem))
        
        if res.fun < worst_function_value:
            worst_function_value = res.fun
            worst_noise = res.x
    
    return test_problem (worst_noise + Point)


def noisy_evaluation (Noise, Point, test_problem):
    
    """
    For a Given point in the design space --- x ---
    Returns negative function value under additive Noise. Negative because the function is to be minimized
    """
    return -1  * test_problem (Point + Noise)


def mini__max__regret__robustness(Point, test_problem, Dimensionality, Noise_Level):
    
    """
    Compute Effective Function Value based on the Mini-Max Regret Robustness
    ----------
       Point: array-like
            The point in the search space for which the effective function value needs to be computed.
            
        test_problem: 
            Indicates the test problem/objective function.
            
        
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
        res = minimize(fun =  __regret__,
                       x0 = starting_point,
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 50 * n_params},
                       args = (Point , test_problem, Dimensionality, Noise_Level))
        
        if res.fun < worst_function_value:
            worst_function_value = res.fun
            worst_noise = res.x
    
    
    return test_problem (Point + worst_noise)

def evaluation (Point, Noise, test_problem):
    return test_problem (Point + Noise)
    
def __regret__ (Noise, Point, test_problem, Dimensionality, Noise_Level):
    
    """
    Compute the regret that may result from making non-optimal decision
    """
    A = test_problem (Point + Noise)
    
    
    seed = 0
    Range = 10
    best_x = None
    best_function_value = np.inf
    n_params = Dimensionality
    n_restarts = int (Dimensionality/2) # fix the number of restarts
    Bou = np.array([-5 + (Noise_Level * Range), 5 - (Noise_Level * Range)] * Dimensionality).reshape(Dimensionality, -1)
    # Bounds to find the robust optimum, the robust optimum is found in a restricted search space of the original domain
    
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(Bou)[:,0], 
                                            np.array(Bou)[:,1], size = (n_restarts, n_params)):
        res = minimize(fun = evaluation,
                       x0 = starting_point,
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 50 * n_params},
                       args = (Noise, test_problem))
        
        if res.fun < best_function_value:
            best_function_value = res.fun
            best_x = res.x
     
    B = best_function_value
    
    return -1 * (A-B)        


def expectation__based__robustness (Point, test_problem, Dimensionality, Noise_Level):
    
    """
    Compute Effective Function Value based on Expectation-based Robustness
    ----------
       Point: array-like
            The point in the search space for which the effective function value needs to be computed.
            
        test_problem: 
            Indicates the test problem/objective function.
            
        
        Dimensionality: integer
            Determines the dimensionality of the problem.
            
        Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    constant = 100
    Range = 10
    Bou = np.array([- (Noise_Level * Range), Noise_Level * Range] * Dimensionality).reshape(Dimensionality, -1)
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = (constant * Dimensionality, Dimensionality))
    summation = np.array([test_problem (Point + sample) for sample in samples])
    
    return np.mean(summation)


def dispersion__based__robustness (Point, test_problem, Dimensionality, Noise_Level):
    
    """
    Compute Effective Function Value based on Dispersion-based Robustness
    ----------
       Point: array-like
            The point in the search space for which the effective function value needs to be computed.
            
        test_problem: 
            Indicates the test problem/objective function.
            
        
        Dimensionality: integer
            Determines the dimensionality of the problem.
            
        Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    Range = 10
    constant = 100
    Bou = np.array([- (Noise_Level * Range), Noise_Level * Range] * Dimensionality).reshape(Dimensionality, -1)
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = (constant * Dimensionality, Dimensionality))
    summation = np.array([test_problem (Point + sample) for sample in samples])
    
    mean = np.mean (summation)
    
    dispersion = np.sqrt ( np.sum(np.square(mean - summation)) / ((constant * Dimensionality) - 1))
    
    return dispersion

def composite__robustness (Point, test_problem, Dimensionality, Noise_Level):
    
    """
    Compute Effective Function Value based on Composite Robustness
    ----------
       Point: array-like
            The point in the search space for which the effective function value needs to be computed.
            
        test_problem: 
            Indicates the test problem/objective function.
            
        
        Dimensionality: integer
            Determines the dimensionality of the problem.
            
        Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    Range = 10
    constant = 100
    Bou = np.array([- (Noise_Level * Range), Noise_Level * Range] * Dimensionality).reshape(Dimensionality, -1)
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = (constant * Dimensionality, Dimensionality))
    summation = np.array([test_problem (Point + sample) for sample in samples])
    
    mean = np.mean (summation)
    
    dispersion = np.sqrt ( np.sum(np.square(mean - summation)) / ((constant * Dimensionality) - 1))
    
    return dispersion + mean
    
    
    
    
    


    