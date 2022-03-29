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


There are three main directories within the main folder, which are titled `Accuracy`, `Hyper_Parameter_Optimization`, and `Results Compilation` respectively.
The first of these, namely `Accuracy` contains three further sub-directories, which are titled `NoiseLevel1`, `NoiseLevel2`, and `NoiseLevel3` respectively.
As the name suggests, these directories contain the code for a particular choice of noise level (the scale of uncertainty).
Within each of these directories, we further come up against six sub-directories which represent the test problem at hand.
If we further explore, these directories contain two jupyter notebooks, namely `Generate_Data_Sets.ipynb` and `Final_Comparison.ipynb`.
While the former contains the methods and routines for generating the training and testing data sets, the latter deals with constructing and 
appraising the surrogate models. Outside, the folder `Hyper_Parameter_Optimization` contains six sub-folders which represent the choice of modeling techniques.
Each of these sub-folders further contains a jupyter notebook, titled `*_Hyper.ipynb`, where `*` serves as the choice of modeling technique, e.g., Kriging, Random Forest.
This jupyter notebook contains the code for hyper parameter optimization for the chosen modeling technique.
Lastly, the folder `Results Compilation` contains several sub-folders which are named after the test problem considered (apart from folder `Graphs` which simply contains
all the plots). Each of those sub-folders contain the file `Graph.ipynb` which produces the figures and plots for the results achieved.
In the following, we describe the technical requirements as well the instructions to run the code in a sequential manner.

# Requirements

In this code, we make use of four python packages (among others), which are presented below in the table.
In particular, `pyDOE` can be utilized for sampling plans and Design of Experiment (DoE).
We employ the so-called `Latin Hypercube Sampling` based on the `pyDOE` package.  
For the purpose of numerical optimization in the code, e.g., to maximize the acquisition function, we utilize the famous `SLSQP` algorithm based on `SciPy` package.
For implementation AEs and VAEs, we utilize the `PyTorch` framework.
Finally, the main purpose of the `scikit-learn` package is to construct the Kriging surrogate, as well as data manipulation/wrangling in general. 
All four required packages can be installed by executing `pip install -r requirements.txt` from the main directory via the command line.

| Package | Description |
| --- | --- |
| pyDOE | For sampling plans and Design of Experiment (DoE).  |
| SciPy | For optimization based on Sequential Quadratic Programming. |
| pandas | For data manipulation and transformation. |
| scikit-learn | For constructing the Kriging surrogate, as well as data manipulation. |

In the following, we describe how to reproduce the experimental setup and results mentioned in our paper.

## 1. Generating the initial Data Set
The first task in the experimental setup deals with the generation of initial data sets. This refers to specifying the coordinates
of the sampling points to observe the functions responses. In our study, we alter the sample sizes to assess its impact on the performance of the
surrogate model. To generate the initial training data, we enter inside the folder `Accuracy`, which is present in the main directory.
Within this folder, we must select a particular sub-folder based on the noise level considered in the case. Note that each noise level affects the
problem landscape in a different way.
Inside the directory of the noise level, we have to select one of the six folders related to our test problem, e.g., ackley 2D, where we come across the notebook 
`Generate_Data_Sets.ipynb`, which contains the methods and routines to generate the initial training data sets based on a variety of sample sizes.
Note that this notebook will also generate the testing data set, which will be later utilized to assess the modeling asccuracy of the surrogate.

## 2. Hyper-parameter Optimization
After generating the initial data sets, we have to traverse back to the main directory, where we come across the folder `Hyper_Parameter_Optimization`.
Inside this, there are six others directories, which are named after the modeling techniques considered in our study.
These directories contain a jupyter notebook, titled `*_Hyper.ipynb`, where `*` serves as the choice of modeling technique, e.g., Kriging, Random Forest.
This jupyter notebook contains the code for hyper parameter optimization for the chosen modeling technique.
We have to utilize this code for each test scenario (noise level, problem, sample size) to find the best settings of hyper parameters, which will then be used when constructing the surrogate model. This is a crucial step, since it ensures the best quality surrogate model based on the sample size.

## 3. Constructing and Appraising the Surrogate Models
Once we retrieve the best configuration of hyper parameters, we can go back to the directory of our test scenario, i.e., by selecting relevant noise level and test problem.
There, we also find another jupyter notebook, titled `Final_Comparison.ipynb`. Inside this, we have to manually save the best settings of hyper parameters
for each test scenario. After that, we can run this notebook, which will construct the surrogate models, utilize them in the optimization, and appraise them
based on the modeling accuracy and optimality. We can save these results to later utilize them to generate figures and plots.

## 4. Plotting the Results
Inside the main directory, we can traverse to the directory titled `Results Compilation`, which further contains the directories named after the test problems.
Inside each of these sub-directories, there is a jupyter notebook, titled `Graph.ipynb`, which contains the necessary code to load the results and plot them.
Note that we have to manually specify the names/paths of the files, which were utilized for saving results in the last section.

# Citation
## Paper Reference
S. Ullah, H. Wang, S. Menzel, B. Sendhoff and T. Bäck, "An Empirical Comparison of Meta-Modeling Techniques for Robust Design Optimization," 2019 IEEE Symposium Series on Computational Intelligence (SSCI), 2019, pp. 819-828.
## BibTex Reference
`@inproceedings{ullah2019empirical`,\
  `title={An empirical comparison of meta-modeling techniques for robust design optimization},`\
  `author={Ullah, Sibghat and Wang, Hao and Menzel, Stefan and Sendhoff, Bernhard and Back, Thomas},`\
  `booktitle={2019 IEEE Symposium Series on Computational Intelligence (SSCI)},`\
  `pages={819--828},`\
  `year={2019},`\
  `organization={IEEE}`\
`}`

# Acknowledgements
This research has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement number 766186 (ECOLE).
