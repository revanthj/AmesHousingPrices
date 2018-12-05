# AmesHousingPrices


The Ames Housing dataset was compiled by Dean De Cock for use in data science education. With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

  - [Numerical Feature Analysis](#numerical-feature-analysis)
  - [Categorical Feature Analysis](#categorical-feature-analysis)
  - [L2 Regularized Model](#l2-regularized-model)
  
  
## Numerical Feature Analysis

**Notebook :** DataAnalysis_NumericFeatures.ipynb

**Steps Followed:**
* Fill missing values for features by taking appropriate approach.

* Visualize distribution of SalePrice to check for skewness in data. If skew present, data has to be transformed since linear models require normal distribution.

* Find coorelation between numerical features and SalePrice. There are 2 correlation coefficients **Pearson's and Spearman's**.
  - Pearson's coefficient assumes linear relationship between variables.
  - Spearman's coefficient does not assume linear relationship between variables.
  
* Pick features that have high correlation with SalePrice. Then, check for multi collinearity between selected features and eliminate redundancy.


## Categorical Feature Analysis

**Notebook :** DataAnalysis_CategoricalFeatures.ipynb

**Steps Followed:**
* Fill missing values for features by taking appropriate approach.

* Calculate Analysis of Variance (ANOVA) to find correlation between categorical features and SalePrice. Select features that has high correlation with Sale Price.
  
* Check for multi collinearity between selected features and eliminate redundancy.


## RIDGE Model

**Notebook :** DataAnalysis_CategoricalFeatures.ipynb

**Steps Followed:**
* Fill missing values for features by taking appropriate approach.

* Calculate Analysis of Variance (ANOVA) to find correlation between categorical features and SalePrice. Select features that has high correlation with Sale Price.
  
* Check for multi collinearity between selected features and eliminate redundancy.
