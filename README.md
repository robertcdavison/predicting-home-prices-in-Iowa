# Predicting Home Prices in Ames, Iowa

_submitted: January 17th, 2020_

### Executive Summary

Zillow is an online housing price estimation and reporting platform used across America. They frequently provide "Estimated home prices" to houses for sale, and not for sale, to help users follow housing market rates. Naturally, more accurate estimates of home values are central to increasing user confidence in Zillow's product. 

This report seeks to develop and refine a home sale price prediction tool that can help Zillow improve their current predicition tools. A Kaggle competition dataset containing various features from homes sold in Ames, Iowa was used to create four different prediction models. Linear Regression was the only machine learning method allowed for this competition and the OLS, ridge, and lasso methods were the only methods used. Root mean squared error in sale price prediction was used to score test predictions. 

### Report Structure:

    * Data cleaning were done in the data_cleaning.ipynb files
    * Modules folder contains the function used to run and compare key metrics for various models
    * Submissions folder contains all predictions submitted to kaggle
    * Remaining submission.ipynb files contain individual model iterations

---

### Primary findings

The data consists of 2,051 homes in the training set and 878 in the test set along with 80 unique features recorded for each home. The average sale price was $181,500 and minimum and maximum prices of $12,800 and $612,000 respectively. The two columns with the most null values were Lot Frontage and Garage Year Built, and the nulls were replaced with their average values. Moreover, 21 home traits with ordinal rankings were mapped to numeric values while 34 categorical home traits were converted to dummy variables. 

The modeling and evaluation workflow was carried out by iteratively selecting features, applying one of the three regression models allowed, and either adding or removing features depending on whether key metrics indicated high bias or high variance.

The first model iteration used OLS regression and only the top 13 variables most highly correlated with sale price to make predictions. This model had high bias and scored an rmse of $36,968 on the kaggle test. This rmse was used as a baseline for comparison going forward. 

Then, in order to get a sense of the upper limit of overfitting the model, the second iteration used an OLS regression on all 216 home features. The kaggle score improved to an rmse of $29,788 and was extremely overfit. Next, the third iteration attempted to minimize this variance by applying a ridge regularized regression to the same 216 variables. This model ended up doing worse than the second with a kaggle score rmse of $36,750. 

Finally, the fourth iteration applied the same Ridge model to a select group of 120 polynomial features. This produced the best kaggle score of $27,579. 

---

### Conclusions and Next Steps

This report set out to predict home prices from a very detailed and complete home feature dataset. Despite there being 80 features, a reasonable best rmse score of $27,579 was achieved by using the polynomial features that were derived from just 12 out of the 80 features. The Ames data offers more data than is necessary to predict sale price. The 12 features used in the fourth iteration consisted of key traits like overall quality, total square footage and year built. The simplicity of this model will likely translate to other towns and housing data sets. In an effort to improve the Zillow Offers program, more research could be done on identifying the bare minimum traits of a home that a home owner could plug into their home in order to get a quick price estimate and draw them in for further evaluation. Other next steps for improving the model would be to iteratively add more relevant categorical features such as neighborhood and house style in order to improve model accuracy. 