
# Short-term Memory Active Learning (SMAL)
As active learning campaigns progress, it has been observed that performance can have the potential to decrease after a "turning point" of maximum performance (Wen et al.). In standard active learning, data is continuously added in a linear fashion. It is hypothesized that this could be problematic since selection functions are likely imperfect, especially during early stages of active learning campaigns where data is limited leading to a poorer understanding of a particular domain. SMAL was developed to augment standard active learning approaches by implementing backward forgetting of training data based on various measures of sample utility. Forgetting data leads to restricted training set sizes making models more compact and less biased, while leading to equivalent or improved overall performance. Additionally, the re-integration of prior experimental data reduces labeling costs, and enhances training set diversity and quality.

<img width="975" height="758" alt="image" src="https://github.com/user-attachments/assets/75adbd91-9ca8-4329-a97b-594efad4a83d" />


This package was built on [MolALKit](https://github.com/RekerLab/MolALKit).

## Installation
```commandline
pip install numpy==1.22.3 git+https://gitlab.com/Xiangyan93/graphdot.git@feature/xy git+https://github.com/bp-kelley/descriptastorus git+https://github.com/Xiangyan93/chemprop.git@molalkit
pip install mgktools
```

## Capabilities and Utility
### Forgetting Protocols for SMAL:
- 'Random': Removes a data point randomly
- 'First': Removes a data point that has been in the training dataset the longest (first-in-first-out)
- 'MinE': Removes a data point with minimal OOB prediction error, targeting instances that can be correctly predicted using other training datapoints and thereby indicating redundance in the training data
- 'MaxE': Removes a data point with maximal OOB prediction error, thereby targeting instances that contradict the remaining training datapoints to clean the training dataset from outliers
- 'MinU': Removes a data point with minimal OOB uncertainty, focusing solely on prediction confidence, thereby targeting instances that the model deems already well-understood
- 'MaxU': Removes a data point with maximum uncertainty, targeting instances closest to the decision boundary to remove datapoints that might strongly impact model architectures to reduce risk for overfitting


### When to Forget:
- 'forget_size': In this study, this was implemented as the training set size associated with the "turning point" of observed standard active learning trajectories on the same dataset. Once this training set size is reached, data will be simultaneously added and forgotten at each iteration.
- 'forget_ratio': Start forgetting when the training set size reaches some percentage of total data from the original pool set.

### Perturbing Data:
- 'error_rate': Fraction of training data to perturb within dataset.

## Active Learning and Other Usage
More information can be found at [MolALKit](https://github.com/RekerLab/MolALKit).

## References
Wen, Y., Li, Z., Xiang, Y., & Reker, D. (2023). Improving Molecular Machine Learning Through Adaptive Subsampling with Active Learning.

