# On the Issue of Computational Cost of Robustness
Contains the code for empirically evaluating and comparing five of the most common robustness formulation in Kriging-based Robust Optimization (KB-RO).
Robust solutions are solutions that are immune to the uncertainty/noise in the decision/search variables.
For finding robust solutions, the conceptual framework of sequential model-based optimization is utilized.

# Introduction
This code is based on our submitted paper (to PPSN 2022), titled `On the Issue of Computational Cost of Robustness in Model-Assisted Robust Optimization, and can be used to reproduce
the experimental setup and results mentioned in the paper. The code is produced in Python 3.7.0. The main packages utilized in this code are presented in the next section which deals with technical requirements. 

The code is present in the main directory as well as three other sub-directories. Within the main directory, the file `doe.py` contains the code to implement
the test functions and design of experiment (DoE) discussed in the paper.
The main directory also contains two csv files which contain the meta-data about the test scenarios.
Out of these two files, `Settings.csv` contains the meta-data about the test cases which have two or five dimensions.
The other file, namely the `Settings__10.csv` contains the meta-data about the test cases which have ten dimensions.


There are three main directories within the main folder, which are titled `Analysis`, `Compute Ground Truth`, and `KB-RO` respectively.
The first of these, namely `Analysis` contains six code files, namely `Avg. CPU Time Per Iteration.ipynb`, `ECDF.ipynb`, 
`Fixed__Iterations__Analysis.ipynb`, `Fixed__Target__Analysis.ipynb`, `Fixed__Time__Analysis.ipynb`, and `Utils.py` respectively.
The file `Utils.py` contains the methods necessary to run the `.ipynb` files, whereas all the other files except  `ECDF.ipynb`
utilize these methods to perform the corresponding analysis.
The `ECDF.ipynb` file generates the empirical cumulative distribution functions (ecdfs) plots utilized in the paper.

The directory `Compute Ground Truth` contains the code to compute the ground truth for each test scenario.
This directory has two files, namely `ground_truth.py` and `Parallel_Ground_Truth_MPI_Pieces.py`. The former is a helper file that
contains methods to compute the ground truth, whereas the latter actually runs the code based on parallel execution.

The directory `KB-RO` contains the actual implementation of Kriging-based Robust Optimization (Algorithm 1) discussed in the paper. 
This directory contains four python files, namely `lib.py`, `new_sample.py`, `robust_lib.py`, and `smbo.py` respectively.
The files `lib.py` and `robust_lib.py` contains the methods to find the robust optimum, whereas the file `new_sample.py`
contains the methods to find a new sample point based on augmented expected improvement criterion. 
Lastly, `smbo.py` runs the actual code in a parallel fashion. 

In the following, we describe the technical requirements as well the instructions to run the code in a sequential manner.

# Requirements

In this code, we make use of six python packages (among others), which are presented below in the table.
In particular, `smt` can be utilized for sampling plans and Design of Experiment (DoE).
We employ the so-called `Latin Hypercube Sampling` based on the `smt` package.  
For the purpose of numerical optimization in the code, e.g., to maximize the acquisition function, we utilize the famous `L-BFGS-B` algorithm based on `SciPy` package.
The package `mpi4py` is utilized for parallel execution of the code.
Finally, the main purpose of the `scikit-learn` package is to construct the Kriging surrogate, as well as data manipulation/wrangling in general. 
All six required packages can be installed by executing `pip install -r requirements.txt` from the main directory via the command line.

| Package | Description |
| --- | --- |
| mpi4py | For parallel execution of the code on DAS-5 server. |
| pickle | For saving and retreiving the results from our experimental setup.  |
| smt | For sampling plans and Design of Experiment (DoE).  |
| SciPy | For numerical optimization based on L-BFGS-B algorithm. |
| pandas | For data manipulation and transformation. |
| scikit-learn | For constructing the Kriging surrogate, as well as data manipulation. |

In the following, we describe how to reproduce the experimental setup and results mentioned in our paper.

## 1. Computing the Ground Truth
The first task in the experimental setup deals with the computation of ground truth/baseline, with which the quality of the robust optimal
solutions will be compared. The code for this task is given in the folder `Analysis`, which contains two files, namely 
`ground_truth.py` and `Parallel_Ground_Truth_MPI_Pieces.py`. The former is a helper file that
contains methods to compute the ground truth, whereas the latter actually runs the code based on parallel execution.

## 2. Kriging-based Robust Optimization
After computing the ground truth, we can run the implementation of KB-RO (Algorithm 1 in the paper) which is provided in directory, titled `KB-RO`.
Here, one should run the file `smbo.py` which parallely runs the KB-RO. Note that in this file, one has to specify
the meta-data (the cases) that needs to be run. We typically run 20 cases in a parallel fashion. These 20 cases are based on a particular choice
of robustness formulation and dimensionality.

## 3. Fixed budget and Fixed Target Analysis
The folder `Analysis` contains the jupyter notebooks for running the analysis based on the results from KB-RO and ground truth.
Here, each notebook implements a specific analysis, e.g., fixed cpu time analysis.
The notebook `ECDF.ipynb` generates the plots of ecdfs based on the analyses carried out.

# Acknowledgements
This research has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement number 766186 (ECOLE).
