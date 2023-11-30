# Electic-Moped-Reviews
## Overview

Welcome to the Electric Moped Reviews Analysis project! 
**Background**
EMO is a manufacturer of electric motorcycles.
EMO launched its first electric motorcycle in India in 2019.

The product team has been asking website users to rate the motorcycles.Ratings from owners help the product team to improve the quality of the motorcycles.
Ratings from non-owners help the product team add new features. They hope the new features will increase the number of new customers.The product team wants to extend the survey. But, they want to be sure they can predict whether the ratings came from owners or non-owners.

This project focuses on predicting whether user ratings for EMO's electric motorcycles come from owners or non-owners. The analysis encompasses data validation, exploratory analysis, and model fitting using logistic regression and a random forest classifier.

## Table of Contents

1. [Background](#background)
2. [Data Validation](#data-validation)
3. [Exploratory Analysis](#exploratory-analysis)
4. [Model Fitting](#model-fitting)
   - [Logistic Regression](#logistic-regression)
   - [Random Forest Classifier](#random-forest-classifier)
5. [Feature Importance](#feature-importance)
   - [Logistic Regression](#logistic-regression-feature-importance)
   - [Random Forest Classifier](#random-forest-classifier-feature-importance)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Logistic Regression](#logistic-regression-hyperparameter-tuning)
   - [Random Forest Classifier](#random-forest-classifier-hyperparameter-tuning)
7. [Evaluation Metrics](#evaluation-metrics)

## Background

EMO, a prominent electric motorcycle manufacturer, launched its first model in India in 2019. To enhance their products, the product team collects user ratings. This analysis aims to predict ownership status based on these ratings, assisting in product improvement and feature additions.

## Data Validation

The dataset comprises 1500 rows and 8 columns, covering variables such as ownership, make model, review month, web browser, reviewer age, primary use, value for money, and overall rating. The dataset undergoes thorough validation, handling missing values, and converting categorical variables into a suitable numeric format.

## Exploratory Analysis

Exploratory analysis offers insights into ownership distribution, overall rating distributions, and relationships between ownership and factors like web browser and primary use.

## Model Fitting

The project utilizes two models for classification: Logistic Regression and Random Forest Classifier.

### Logistic Regression

Logistic Regression, chosen for its simplicity and efficiency, is employed to predict ownership. Evaluation metrics include F1 score and precision score.

### Random Forest Classifier

This model is selected for its ability to handle categorical variables and missing data. Similar to logistic regression, F1 score and precision score are used for model evaluation.

## Feature Importance

Understanding feature importance aids in interpreting model decisions.

### Logistic Regression Feature Importance

The coefficients of logistic regression features are analyzed to understand their impact on ownership prediction.

### Random Forest Classifier Feature Importance

The feature importances of the random forest model are examined to identify key contributors to ownership prediction.

## Hyperparameter Tuning

Fine-tuning model parameters is crucial for optimal performance.

### Logistic Regression Hyperparameter Tuning

Grid search is employed to find the best hyperparameters for logistic regression.

### Random Forest Classifier Hyperparameter Tuning

Similarly, grid search is utilized to find optimal hyperparameters for the random forest classifier.

## Evaluation Metrics

Precision score and F1 score are selected as evaluation metrics, providing insights into the models' ability to predict ownership. Results suggest that the Random Forest Classifier outperforms Logistic Regression in this context.

Feel free to delve into each section for detailed insights and adapt parameters as needed. Happy exploring!
