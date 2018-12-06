# AmesHousingPrices


The Ames Housing dataset was compiled by Dean De Cock for use in data science education. With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

  - [Numerical Feature Analysis](#numerical-feature-analysis)
  - [Categorical Feature Analysis](#categorical-feature-analysis)
  - [L2 Regularized Model](#l2-regularized-model)
  - [LASSO Feature Selection](#lasso-feature-selection)
  - [Linear Model With Backward Elimination](#linear-model-with-backward-elimination)
  - [Decision Tree Model](#decision-tree-model)
  - [Random Forest Model](#random-forest-model)
  - [Stacked Model](#stacked-model)
  
  
## Numerical Feature Analysis

**Notebook :** DataAnalysis_NumericFeatures.ipynb

**Key Steps:**
* Fill missing values for features by taking appropriate approach.

* Visualize distribution of SalePrice to check for skewness in data. If skew present, data has to be transformed since linear models require normal distribution.

* Find coorelation between numerical features and SalePrice. There are 2 correlation coefficients **Pearson's and Spearman's**.
  - Pearson's coefficient assumes linear relationship between variables.
  - Spearman's coefficient does not assume linear relationship between variables.
  
* Pick features that have high correlation with SalePrice. Then, check for multi collinearity between selected features and eliminate redundancy.


## Categorical Feature Analysis

**Notebook :** DataAnalysis_CategoricalFeatures.ipynb

**Key Steps:**
* Fill missing values for features by taking appropriate approach.

* Calculate Analysis of Variance (ANOVA) to find correlation between categorical features and SalePrice. Select features that has high correlation with Sale Price.
  
* Check for multi collinearity between selected features and eliminate redundancy.


## L2 Regularized Model

**Notebook :** L2RegularizedModel.ipynb

**Key Steps:**

* From Numerical Analysis, we found that Sale Price data is positively Skewed. Correct disribution by applying LOG transformation.

* Impute missing values. Check whether selected numerical features have normally distribution or not. If not, apply LOG transformation and correct them.
  
* Rescale independent variables by applying standardization (Refer https://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/)

* Split input data into training and test sets. Train basic linear model with default parameters on training data and establish baseline scores.

* Apply L2 Regularization model on data and compare scores with baseline model.

* SUMMARY


## LASSO Feature Selection

**Notebook :** LassoFeatureSelection.ipynb

**Key Steps:**

* Impute missing values. Correct the distribution of Sale Price and numerical features to follow GAUSSIAN distribution by applying LOG transformation. 

* Apply LASSO Model to eliminate irrelevant features.

* Apply L2 Regularization (RIDGE) model on LASSO selected features and compare scores with baseline model.

* SUMMARY


## Linear Model With Backward Elimination

**Notebooks :** BackwardFeatureElimination.ipynb, BackwardElimination_NumericFeatureOnly.ipynb

Backward Elimination is a feature selection technique that start with all the features and removes least significant feature (by comparing with preselected p-value) at each iteration thereby improving performance of the model. This is repeated until all features are more significant than preselected p-value

**Key Steps:**

* Impute missing values. Correct the distribution of Sale Price and numerical features to follow GAUSSIAN distribution by applying LOG transformation. 

* Select a p-value (generally p=0.05)

* Calculate p-value of each feature using statsmodels OLS technique. 

* Iterate though feature p-value list and eliminate a feature with highest p-value that is greater than preselected p-value.

* If feature with highest p-value is less than or equal to preselected p-value, break the loop.

* Apply L2 Regularization (RIDGE) model on remaining features and compare scores with baseline model.

* SUMMARY


## Decision Tree Model

**Notebooks :** DecisionTreeRegressor.ipynb

Decision Tree is a Non-Parametric model which means it do not make strong assumptions about the form of the mapping function. By not making assumptions, they are free to learn any functional form from the training data.

**Key Steps:**

* Impute missing values. Select features from Backward Elimination.

* Apply decision tree model with default parameters and establish baseline scores.

* Tune parameters of the decision tree such as max_depth, min_samples_split, min_samples_leaf, max_features and select value that give best accuracy. 

* Implement decision tree with selected best parameters and compare with baseline model.

* It is observed that Optimal parameter model is not performing as expected. Since I tuned paramters individually, optimal value of one parameters might not work best with optimal value of other parameter. Resolution is to tune parameters with Random Search or Grid Search CV.

* SUMMARY


## Random Forest Model

**Notebooks :** RandomForestRegressor.ipynb

**Key Steps:**

* Impute missing values. Select features from Backward Elimination.

* Apply random forest model with default parameters and establish baseline scores.

* Tune random forest parameters such as n_estimators, min_samples_split, min_samples_leaf, max_depth using RandomizedSearchCV

* Implement random forest with selected best parameters and compare with baseline model.

* SUMMARY


## Stacked Model

**Notebooks :** StackedModel.ipynb

In this notebook, I implemented 2 techniques.
* **Average Model** : This is a simple technique. I trained 4 models and averaged predictions of 4 models to give output.

* **Stacked Model** : Stacking (also called meta ensembling) is a model ensembling technique used to combine information from multiple predictive models to generate a new model. 

**Key Steps:**

* Impute missing values. Correct the distribution of Sale Price and numerical features to follow GAUSSIAN distribution by applying LOG transformation.

* Select features from Backward Elimination.

* Train 4 models i.e. LASSO, RIDGE, ElasticNet and Random Forest.

* Implement Average Model that averages output from above 4 models to predict output

* Implement Stacked model that combines output from 3 models to produce input for meta model.

* SUMMARY
