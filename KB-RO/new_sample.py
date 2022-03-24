import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats import norm

def __sample__next__point__ (acquisition_func, Model, robust_optimum, Robustness = "mini-max", 
                             Dimensionality = 5, Noise_Level = 0.05):
    
    """ 
    Determine next location to compute function response based on augmented acquisition function
    
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to find the next sample point.
            
        Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated function responses.
            
        robust_optimum: double 
            Best-so-far known/estimated "robust" Kriging value of the function
            
        robustness: string
            Gives the robustness formulation to be employed.
            
        Dimensionality: integer
            Determines the dimensionality of the problem.
            
       Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    seed = 0
    Range = 10
    best_x = None
    best_function_value = np.inf
    n_params = Dimensionality
    n_restarts = int (Dimensionality / 2) # fix the number of restarts
    Bou = np.array([-5 + (Noise_Level * Range), 5 - (Noise_Level * Range)] * Dimensionality).reshape(Dimensionality, -1)
    
            
    np.random.seed(seed)
    for starting_point in np.random.uniform(np.array(Bou)[:,0], 
                                            np.array(Bou)[:,1], size = (n_restarts, n_params)):
        res = minimize(fun = acquisition_func,
                       x0 = np.atleast_2d(starting_point),
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 10 * n_params},
                       args = (robust_optimum, Model, Robustness, Dimensionality, Noise_Level))
        
        if res.fun < best_function_value:
            best_function_value = res.fun
            best_x = res.x
        
    return best_x

def __EIC__ (Point, robust_optimum, Model, Robustness, Dimensionality, Noise_Level):
    
    """
    Compute the augmented EIC for the Robust Scneario based on the robustness criterion chosen
    
    Arguments:
    ----------
        Point: array-like, 
            The point in the search domain for which the effective Kriging prediction needs to be computed.
            
        
        robust_optimum: double 
            Best-so-far known/estimated "robust" Kriging value of the function
            
        Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated function responses.
            
        robustness: string
            Gives the robustness formulation to be employed.
            
        Dimensionality: integer
            Determines the dimensionality of the problem.
            
       Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    mean, variance = __compute__effective__mean__and__variance__ (Point,  Model,
                                                                  Robustness, Dimensionality, Noise_Level)
    
    if variance == 0:
        expected_improvement = 0
        
    else:
        
        Z = ( robust_optimum - mean) / np.sqrt (variance)
        expected_improvement = (robust_optimum - mean) * norm.cdf(Z) + np.sqrt (variance) * norm.pdf(Z)
        
    return - expected_improvement

def __compute__effective__mean__and__variance__ (Point, Model, Robustness, Dimensionality, Noise_Level):
    
    """
    Compute effective Mean and Variance of the (Posterior) Stochastic Process based on the robustness criterion chosen
    
    Arguments:
    ----------
        Point: array-like, 
            The point in the search domain for which the effective Kriging prediction needs to be computed.
            
        Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated function responses.
            
        Robustness: string
            Gives the robustness formulation to be employed.
            
        Dimensionality: integer
            Determines the dimensionality of the problem.
            
       Noise_Level: double.
            Noise Level --- Determines the % maximum possible deviation in the search variables. 
    """
    if Robustness == "mini-max":
        mean, variance = mini__max__robustness (Point, Model, Dimensionality, Noise_Level)
        
    elif Robustness == "mini-max-regret":
        mean, variance = mini__max__regret__robustness (Point, Model, Dimensionality, Noise_Level)
        
    elif Robustness == "expectation":
        mean, variance = expectation__based__robustness (Point, Model, Dimensionality, Noise_Level)
        
    elif Robustness == "dispersion":
        mean, variance = dispersion__based__robustness (Point, Model, Dimensionality, Noise_Level)
        
    elif Robustness == "composite":
        mean, variance = composite__robustness (Point, Model, Dimensionality, Noise_Level)
        
    return mean, variance 


def mini__max__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute effective Mean and Variance of the (Posterior) Stochastic Process based on the Mini-Max Criterion
    
    Arguments:
    ----------
        Point: array-like, 
            The point in the search domain for which the effective Kriging prediction needs to be computed.
            
        Model: GaussianProcessRegressor object (from sklearn).
            Gaussian process trained on previously evaluated function responses.
        
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
        
        res = minimize(fun =  noisy_prediction,
                       x0 = np.atleast_2d (starting_point),
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 10 * n_params},
                       args = (Point , Model))
        
        if res.fun < worst_function_value:
            worst_function_value = res.fun
            worst_noise = res.x
           
    mean, sigma = Model.predict ( np.atleast_2d (Point + worst_noise) , return_std = True)    
    return mean, sigma **2


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
    
    prediction = Model.predict ( np.atleast_2d (Point + Noise))
    return -1 * prediction


def mini__max__regret__robustness(Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute effective Mean and Variance of the (Posterior) Stochastic Process based on the Mini-Max Regret Criterion
    
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
        
        res = minimize(fun =  __regret__,
                       x0 = np.atleast_2d (starting_point),
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 10 * n_params},
                       args = (Point , Model, Dimensionality, Noise_Level))
        
        if res.fun < worst_function_value:
            worst_function_value = res.fun
            worst_noise = res.x
            
    
    mean, sigma = Model.predict( np.atleast_2d (Point + worst_noise) , return_std = True)    
    return mean, sigma **2


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
    n_restarts = int (Dimensionality/2) # fix the number of restarts
    Bou = np.array([-5 + (Noise_Level * Range), 5 - (Noise_Level * Range)] * Dimensionality).reshape(Dimensionality, -1)
    
    
    np.random.seed(seed)
    
    for starting_point in np.random.uniform(np.array(Bou)[:,0], 
                                            np.array(Bou)[:,1], size = (n_restarts, n_params)):
        res = minimize(fun = prediction,
                       x0 = np.atleast_2d(starting_point),
                       bounds = Bou,
                       method = 'L-BFGS-B',
                       options = {'maxfun': 10 * n_params},
                       args = (Noise, Model))
        
        if res.fun < best_function_value:
            best_function_value = res.fun
            best_x = res.x
     
    B = best_function_value
    
    return -1 * (A-B)

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
    return Model.predict ( np.atleast_2d (Point + Noise))

def expectation__based__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute effective Mean and Variance of the (Posterior) Stochastic Process based on the Expectation-based Criterion
    
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
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = ( constant * Dimensionality, Dimensionality))
    
    # compute the sample mean of Kriging predictions given a point X in search space and samples from the p.d.f of noise 
    summation = np.array ([Model.predict (np.atleast_2d (Point + sample)) for sample in samples])
    mean = np.mean(summation) 
    
    # compute the Variance of Stochastic Process given a point X in search space and samples from the p.d.f of noise
    k = lambda x1, x2: Model.kernel_( np.atleast_2d (x1), np.atleast_2d (x2))
    variance = np.sum (k( Point + samples, Point + samples)) / (constant * Dimensionality) **2
    return mean, variance


def dispersion__based__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute effective Mean and Variance of the (Posterior) Stochastic Process based on the dispersion-based Criterion
    
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
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = ( constant * Dimensionality, Dimensionality))
    
    # compute the sample mean of Kriging predictions given a point X in search space and samples from the p.d.f of noise 
    summation = np.array ([Model.predict (np.atleast_2d (Point + sample)) for sample in samples])
    mean = np.mean(summation) 
    
    #compute dispersion
    dispersion = np.sqrt (np.sum (np.square (mean - summation)) / ((constant * Dimensionality) - 1))
    
    variance = np.var (np.square (mean - summation))
    
    return dispersion, variance


def composite__robustness (Point, Model, Dimensionality, Noise_Level):
    
    """
    Compute effective Mean and Variance of the (Posterior) Stochastic Process based on the Composite robustness Criterion
    
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
    samples = np.random.uniform (np.array(Bou)[:,0], np.array(Bou)[:,1], size = ( constant * Dimensionality, Dimensionality))
    
    # compute the sample mean of Kriging predictions given a point X in search space and samples from the p.d.f of noise 
    summation = np.array ([Model.predict (np.atleast_2d (Point + sample)) for sample in samples])
    mean = np.mean(summation) 
    
    #compute dispersion
    dispersion = np.sqrt (np.sum (np.square (mean - summation)) / ((constant * Dimensionality) - 1))
    
    variance = np.var (np.square (mean - summation))
    
    return mean + dispersion, variance







