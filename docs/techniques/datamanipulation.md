How data is treated is really important in all Data Science projects and it is also the case for Quantum Machine Learning. Do to the techniques used we need all data to be informed, make it numerical and with few but informative features to be used.

That is why Data Manipulation is a relevant step in order to proceed with any other technique within the Falcondale package.

Trying to make it as easy as possible to the users, the creation of a Project requires indicating which feature is the target variable of our exercise.

```py
from falcondale import Project

myproject = Project(dataset, target="target")
```

It also allows to indicate if data profiling must be done. This feature helps future steps to count with more information when deciding on automatic steps but it is up to the user to decide on this step as it may add more computation to the whole pipeline and one might want to omit this additional computation initially.

```py
myproject.profile()
```

A report similar to the following one will be shown
```
Number of samples on the dataset: 569
Number of columns on the dataset: 31
target has been selected as target variable for supervised learning tasks.
Ratio of samples with missing values: 0.0
All data is numeric therefore little preprocessing might need to be done.
Some relevant alerts have been found:
	 * [mean concavity] has 13 (2.3%) zeros
	 * [mean concave points] has 13 (2.3%) zeros
	 * [concavity error] has 13 (2.3%) zeros
	 * [concave points error] has 13 (2.3%) zeros
	 * [worst concavity] has 13 (2.3%) zeros
	 * [worst concave points] has 13 (2.3%) zeros
	 * [target] has 212 (37.3%) zeros
```
This report is generated thanks to the YData colleagues and their powerful [ydata-profiling](https://github.com/ydataai/ydata-profiling) package.

## Preprocessing

In order to ease the work of the classifiers some preprocessing steps need to be implemented. Simple things like filling NAs or some more complex techniques like dimensionality reduction can be performed. This is an important step as filling all cells, standardizing the data and reducing dimensions will ease the execution when going into real hardware. Moreover, simulating systems beyond 20 variables (mapped to 20 qubits in general) would start to consume quite some resources on your machine. That is way, whenever possible, dimensionality reduction will help speed up our analysis.

```py
myproject.preprocess()
```

Is the most simple command we can send and will basically input missing data and standardized the features. This can be checked by calling to the following function.

```py
myproject.show_features()
```

## Dimensionality reduction

Falcondale supports to call [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) based dimensionality reduction at the moment by simply invoking the method indicating the amount of dimensions to be reduced to.

```py
myproject.preprocess(reduced_dimension=3)
```
This method will be extended in the future by adding more complex techniques for embedding data into lower dimensional spaces. Currently also [LDA]() based dimensionality reduction is being evaluated so that it gets included in following versions.