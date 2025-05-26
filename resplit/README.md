# Exploring the Whole Rashomon Set of Sparse Decision Trees (TreeFARMS)

This code creates *Rashomon sets* of decision trees. The Rashomon set is the set of all almost-optimal models. This code is able to enumerate the Rashomon set for sparse decision trees. In other words, instead of returning a single optimal decision tree, it returns a set of decision trees with an objective (misclassification loss with a penalty on the number of leaves) below a pre-defined threshold. To learn more about TreeFARMS, please read our [research paper](https://arxiv.org/abs/2209.08040) (published at NeurIPS'22). 

Given the Rashomon set enumerated by this code, an interactive tool [TimberTrek](https://github.com/poloclub/timbertrek) can be used to visualize and explore this set of trees. 

To make TreeFARMS run faster, please use the options to limit the depth of the tree, and increase the regularization parameter above 0.02. If you run the algorithm without a depth constraint or set the regularization too small, it will run more slowly.

TreeFARMS builds on a number of innovations for scalable construction of optimal tree-based classifiers: Scalable Bayesian Rule Lists[[8](#related-work)], CORELS[[2](#related-work)], OSDT[[4](#related-work)], and, most closely, GOSDT[[5](#related-work)]. 

**Note: this repository is built based on [Fast Sparse Decision Tree Optimization via Reference Ensembles](https://github.com/ubc-systopia/gosdt-guesses). Package is renamed to allow the installation of both packages. In this repository, all references to GOSDT refers to TreeFARMS algorithm.**

# Table of Content
- [Installation](#installation)
- [Compilation](#compilation)
- [Configuration](#configuration)
- [Example](#example)
- [Repository Structure](#structure)
- [License](#license)
- [FAQs](#faqs)

---

# Installation

You may use the following commands to install TreeFARMS along with its dependencies on macOS, Ubuntu and Windows.  
You need **Python 3.7 or later** to use the module `treefarms` in your project. If you encountered an error `ERROR: Could not find a version that satisfies the requirement treefarms`, try updating your pip to the latest version. 

```bash
pip3 install attrs packaging editables pandas scikit-learn sortedcontainers gmpy2 matplotlib
pip3 install treefarms
```

You can find a list of available wheels on [PyPI](https://pypi.org/project/treefarms/).  
Please feel free to open an issue if you do not see your distribution offered.

---

# Compilation

Please refer to the [manual](doc/build.md) to build the C++ command line interface and the Python extension module and run the experiment with example datasets on your machine.

---

# Configuration

The configuration is a JSON object and has the following structure and default values:
```json
{
  
    "regularization": 0.05,
    "rashomon": true, 
    "rashomon_bound_multiplier": 0.05,

    "rashomon_bound": 0,
    "rashomon_bound_adder": 0,
    "output_accuracy_model_set": false, 
    "output_covered_sets": [],
    "covered_sets_thresholds": [],
    "rashomon_model_set_suffix": "",
    "rashomon_ignore_trivial_extensions": true,
    "rashomon_trie": "",
  
    "depth_budget": 0,
    "uncertainty_tolerance": 0.0,
    "upperbound": 0.0,
    "precision_limit": 0,
    "stack_limit": 0,
    "tile_limit": 0,
    "time_limit": 0,
    "worker_limit": 1,
    "minimum_captured_points": 0,
    "cancellation": true,
    "look_ahead": true,
    "diagnostics": false,
    "verbose": true,
    "costs": "",
    "model": "",
    "profile": "",
    "timing": "",
    "trace": "",
    "tree": "",
    "datatset_encoding": "",
    "memory_checkpoints": [],
  }
```

## Key parameters

**regularization**
 - Values: Decimal within range [0,1]
 - Description: Used to penalize complexity. A complexity penalty is added to the risk in the following way.
   ```
   ComplexityPenalty = # Leaves x regularization
   ```
 - Default: 0.05
 - **Note: We highly recommend setting the regularization to a value larger than 1/num_samples. A small regularization could lead to a longer training time. If a smaller regularization is preferred, you must set the parameter `allow_small_reg` to true, which by default is false.**
 
**rashomon**
- Values: true or false
- Description: Enables extraction of Rashomon set. Setting it to false allows extracting only the optimal tree, allowing a more flexible way of setting Rashomon bound. Note: obtaining the optimal objective value is currently not supported in Python API.
- Default: true

**rashomon_bound_multiplier**
- Values: Decimal > 0
- Description: Used to set the Rashomon bound. Rashomon bound = (1 + rashomon_bound_multiplier) * optimal objective value. Mutually exclusive with `rashomon_bound` and `rashomon_bound_adder`.
- Default: 0.05
- **Warning: The size of Rashomon set increasing exponentially w.r.t. this argument.**

## More parameters
### Rashomon-specific configs
**rashomon_bound**
- Values: Decimal > 0
- Description: Used to set the Rashomon bound. Directly setting the Rashomon bound if it is known. Mutually exclusive with `rashomon_bound_multiplier` and `rashomon_bound_adder`.
- Default: 0
- **Warning: The size of Rashomon set increasing exponentially w.r.t. this argument.**

**rashomon_bound_adder**
- Values: Decimal > 0
- Description: Used to set the Rashomon bound. Rashomon bound = rashomon_bound_adder + optimal objective value. Mutually exclusive with `rashomon_bound` and `rashomon_bound_multiplier`.
- Default: 0
- **Warning: The size of Rashomon set increasing exponentially w.r.t. this argument.**

**output_accuracy_model_set**
 - Values: true or false
 - Description: Enables outputting the Rashomon set with accuracy metric in a file named by `model_set-accuracy-` with the suffix indicated in `rashomon_model_set_suffix`. Note that it is not required for Python API to obtain the accuracy Rashomon set.
 - Default: false

**output_covered_sets**
 - Values: an array of string in `['f1', 'bacc', 'auc']`.
 - Description: Enables outputting the Rashomon set with given metric in a file named by `model_set-[metric]-` with the suffix indicated in `rashomon_model_set_suffix`. 
 - Default: []

**covered_sets_thresholds**
 - Values: an array of Decimals.
 - Description: Sets the extraction threshold of given metric. 
 - Default: []

**rashomon_model_set_suffix**
 - Values: string representing a suffix of a file.
 - Description: Sets file suffix for outputted model sets.
 - Special Cases: None. Disable outputting to files by setting above config correspondingly. With an empty value the program still attempts to produce a file without suffix.
 - Default: Empty string

**rashomon_ignore_trivial_extensions**
 - Values: true or false
 - Description: Enables ignoring trivial splits, or terminal splits that does not improve accuracy.
 - Default: true

**rashomon_trie**
 - Values: string representing a path to a file.
 - Description: The output model set will be converted to trie representation, and be written to this file.
 - Special Case: When set to empty string, no trie will be stored.
 - Default: Empty string

### Flag

**balance**
 - Values: true or false
 - Description: Enables overriding the sample importance by equalizing the importance of each present class
 - Default: false

**cancellation**
 - Values: true or false
 - Description: Enables propagate up the dependency graph of task cancellations
 - Default: true

**look_ahead**
 - Values: true or false
 - Description: Enables the one-step look-ahead bound implemented via scopes
 - Default: true

**similar_support**
 - Values: true or false
 - Description: Enables the similar support bound implemented via the distance index
 - Default: true

**feature_exchange**
 - Values: true or false
 - Description: Enables pruning of pairs of features using subset comparison
 - Default: false

**continuous_feature_exchange**
 - Values: true or false
 - Description: Enables pruning of pairs continuous of feature thresholds using subset comparison
 - Default: false

**feature_transform**
 - Values: true or false
 - Description: Enables the equivalence discovery through simple feature transformations
 - Default: true

**rule_list**
 - Values: true or false
 - Description: Enables rule-list constraints on models
 - Default: false
 
**non_binary**
 - Values: true or false
 - Description: Enables non-binary encoding
 - Default: false

**diagnostics**
 - Values: true or false
 - Description: Enables printing of diagnostic trace when an error is encountered to standard output
 - Default: false

**verbose**
 - Values: true or false
 - Description: Enables printing of configuration, progress, and results to standard output
 - Default: false




### Tuners

**uncertainty_tolerance**
 - Values: Decimal within range [0,1]
 - Description: Used to allow early termination of the algorithm. Any models produced as a result are guaranteed to score within the lowerbound and upperbound at the time of termination. However, the algorithm does not guarantee that the optimal model is within the produced model unless the uncertainty value has reached 0.
 - Default: 0.0

**upperbound**
 - Values: Decimal within range [0,1]
 - Description: Used to limit the risk of model search space. This can be used to ensure that no models are produced if even the optimal model exceeds a desired maximum risk. This also accelerates learning if the upperbound is taken from the risk of a nearly optimal model.
 - Special Cases: When set to 0, the bound is not activated. 
 - Default: 0.0

### Limits

**depth_budget**
- Values: Integers >= 1
- Description: Used to set the maximum tree depth for solutions, counting a tree with just the root node as depth 1. 0 means unlimited.
- Default: 0
 
**time_limit**
 - Values: Decimal greater than or equal to 0
 - Description: A time limit upon which the algorithm will terminate. If the time limit is reached, the algorithm will terminate with an error.
 - Special Cases: When set to 0, no time limit is imposed.
 - Default: 0

**precision_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of significant figures considered when converting ordinal features into binary features.
 - Special Cases: When set to 0, no limit is imposed.
 - Default: 0

**stack_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of bytes considered for use when allocating local buffers for worker threads.
 - Special Cases: When set to 0, all local buffers will be allocated from the heap.
 - Default: 0

**worker_limit**
 - Values: Decimal greater than or equal to 1
 - Description: The maximum number of threads allocated to executing th algorithm.
 - Special Cases: When set to 0, a single thread is created for each core detected on the machine.
 - Default: 1

### Files

**costs**
 - Values: string representing a path to a file.
 - Description: This file must contain a CSV representing the cost matrix for calculating loss.
   - The first row is a header listing every class that is present in the training data
   - Each subsequent row contains the cost incurred of predicitng class **i** when the true class is **j**, where **i** is the row index and **j** is the column index
   - Example where each false negative costs 0.1 and each false positive costs 0.2 (and correct predictions costs 0.0):
     ```
     negative,positive
     0.0,0.1
     0.2,0.0
     ```
   - Example for multi-class objectives:
     ```
     class-A,class-B,class-C
     0.0,0.1,0.3
     0.2,0.0,0.1
     0.8,0.3,0.0
     ```
   - Note: costs values are not normalized, so high cost values lower the relative weight of regularization
 - Special Case: When set to empty string, a default cost matrix is used which represents unweighted training misclassification.
 - Default: Empty string

**model**
 - Values: string representing a path to a file.
 - Description: The output models will be written to this file.
 - Special Case: When set to empty string, no model will be stored.
 - Default: Empty string

**profile**
 - Values: string representing a path to a file.
 - Description: Various analytics will be logged to this file.
 - Special Case: When set to empty string, no analytics will be stored.
 - Default: Empty string

**timing**
 - Values: string representing a path to a file.
 - Description: The training time will be appended to this file.
 - Special Case: When set to empty string, no training time will be stored.
 - Default: Empty string

**trace**
 - Values: string representing a path to a directory.
 - Description: snapshots used for trace visualization will be stored in this directory
 - Special Case: When set to empty string, no snapshots are stored.
 - Default: Empty string

**tree**
 - Values: string representing a path to a directory.
 - Description: snapshots used for trace-tree visualization will be stored in this directory
 - Special Case: When set to empty string, no snapshots are stored.
 - Default: Empty string

---
# Example

Example code to run TreeFARMS with threshold guessing, lower bound guessing, and depth limit. The example python file is available in [treefarms/example.py](/treefarms/example.py). A tutorial ipython notebook is available in [treefarms/tutorial.ipynb](/treefarms/tutorial.ipynb).  

---

# Structure

This repository contains the following directories and files:
- **.github**: Configurations for GitHub action runners.
- **doc**: Documentation
- **experiments**: Datasets and their configurations to run experiments
- **treefarms**: Jupyter notebook, Python implementation and wrappers around C++ implementation
- **include**: Required 3rd-party header-only libraries
- **log**: Log files
- **src**: Source files for C++ implementation and Python binding
- **test**: Source files for unit tests
- **build.py**: Python script that builds the project automatically
- **CMakeLists.txt**: Configuration file for the CMake build system
- **pyproject.toml**: Configuration file for the SciKit build system
- **setup.py**: Python script that builds the wheel file

---

# FAQs

If you run into any issues when running TreeFARMS, consult the [**FAQs**](/doc/faqs.md) first. 

---

# License

This software is licensed under a 3-clause BSD license (see the LICENSE file for details). 

---

## Related Work
[1] Aglin, G.; Nijssen, S.; and Schaus, P. 2020. Learning optimal decision trees using caching branch-and-bound search. In _AAAI Conference on Artificial Intelligence_, volume 34, 3146–3153.

[2] Angelino, E.; Larus-Stone, N.; Alabi, D.; Seltzer, M.; and Rudin, C. 2018. Learning Certifiably Optimal Rule Lists for Categorical Data. _Journal of Machine Learning Research_, 18(234): 1–78.

[3] Breiman, L.; Friedman, J.; Stone, C. J.; and Olshen, R. A. 1984. _Classification and Regression Trees_. CRC press.

[4] Hu, X.; Rudin, C.; and Seltzer, M. 2019. Optimal sparse decision trees. In _Advances in Neural Information Processing Systems_, 7267–7275.

[5] Lin, J.; Zhong, C.; Hu, D.; Rudin, C.; and Seltzer, M. 2020. Generalized and scalable optimal sparse decision trees. In _International Conference on Machine Learning (ICML)_, 6150–6160.

[6] Quinlan, J. R. 1993. C4.5: _Programs for Machine Learning_. Morgan Kaufmann

[7] Verwer, S.; and Zhang, Y. 2019. Learning optimal classification trees using a binary linear program formulation. In _AAAI
Conference on Artificial Intelligence_, volume 33, 1625–1632.

[8] Yang, H., Rudin, C., & Seltzer, M. (2017, July). Scalable Bayesian rule lists. In _International Conference on Machine Learning (ICML)_ (pp. 3921-3930). PMLR.

---
