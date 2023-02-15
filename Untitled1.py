#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[ ]:


df = pd.read_csv("https://s3.amazonaws.com/talent-assets.datacamp.com/electric_bike_ratings_2212.csv")
df.head()

# In[ ]:





# owned
# Nominal. Whether the reviewer owns the moped (1) or not (0). Missing values should be removed.
# make_model
# Nominal. The make and model of the bike, one of six possible values (Nielah-Eyden, Nielah-Keetra, Lunna-Keetra, Hoang-Keetra, Lunna-Eyden, Hoang-Eyden).
# Replace missing values with “unknown”.
# review_month
# Nominal. The month the review was given in English short format (Jan, Feb, Mar, Apr etc.).
# Replace missing values with “unknown”
# web_browser
# Nominal. Web browser used by the user leaving the review, one of Chrome, IE, Firefox, Safari, Android, Opera
# Replace missing values with “unknown”.
# reviewer_age
# Discrete. Age of the user leaving the review. Integer values from 16. Replace missing values with the average age.
# primary_use
# Nominal. The main reason the user reports that they use the bike for. One of Commuting or Leisure
# Replace missing values with “unknown”.
# value_for_money
# Discrete. Rating given by the user on value for money of the bike. Rating from 1 to 10.
# Replace missing values with 0.
# overall_rating
# Continuous. Total rating score after combining multiple rating scores. Continuous values from 0 to 25 are possible.
# Replace missing values with the average rating.

# Data validation

# eg:
#     This data set has 6738 rows, 9 columns. I have validated all variables and I have not made any changes after validation. All the columns are just as described in the data dictionary:
# 
# model: character, 18 possible values
# year: numeric, from 1998 to 2020
# price: numeric
# transmission: character, four categories
# mileage: numeric
# fuelType: character, four categories
# tax: numeric
# mpg: numeric
# engineSize: numeric, 16 possible values

# In[3]:


#necessary cells for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score,f1_score


# In[ ]:


datasets.info()-> to chek the information how many rows plus columns is there any missing values


# In[ ]:


datasts.isnull().sum() - to see if there is any missing values


# In[ ]:


df['owned'].nunique()
df['make_model'].nunique()
df['review_month'].nunique()
df['web_browser'].nunique()
df['reviewer_age'].nunique()
df['primary_use'].nunique()
df['Leisure'].nunique()
df['overall_rating'].nunique()


# In[ ]:


#Repalce the missing value according to the description
#1-
df.dropna(subset=['owned'], inplace=True)
#2,3,4,5
df[['make_model', 'review_month', 'web_browser','primary_use']] 
= df[['make_model', 'review_month', 'web_browser','primary_use']].apply(lambda x: 
                                                                        x.fillna("unknown"))
#5,8
df[['reviewer_age', 'overall_rating']] 
= df[['reviewer_age', 'overall_rating']].apply(lambda x: x.fillna(x.mean()))
#7

df['value_for_money'].fillna(mean,inplace=True)


# In[ ]:


df.describe() - to see the full view


# DATA VISULIZATION / EXPLORATORY ANALYSIS

# In[ ]:


Target Variable - owned

Since The product team wants to extend the survey. 
But, they want to be sure they can predict
whether the ratings came from owners or non-owners.

we can then use a  barplot visualization 
to show the number of reviews from owners and non-owners: 


# In[ ]:


sns.countplot(x='owned', data=df)
plt.show()


# a. From the visualization, it is clear that the category of the variable "owned" with value 1 (or owners) has the most number of observations.
# 
# b. The observations are not balanced across categories of the variable "owned". The number of observations for owners is more than non-owners. This could potentially be a problem if the goal is to make predictions on new data since a model trained on this data might be biased towards predicting "owners". 

# To describe the distribution of overall rating, we can use a histogram visualization

# In[ ]:


data['overall_rating'].hist()
plt.xlabel('Overall Rating')
plt.ylabel('Count')
plt.title('Distribution of Overall Rating')
plt.show()


# We can also use a boxplot visualization to show the relationship 
# between ownership and overall rating:

# In[ ]:


sns.boxplot(x='owned', y='overall_rating', data=data)
plt.show()


# In[ ]:





# In[ ]:


Model Fitting & Evaluation


The business wants to predict whether a 
review came from an owner or not using the data provided and it is classification tasks
so i am choosing to use Logistic Regression and Random Forest Classifier 
For the evaluation,i am choosing score,confusion_matrix and classification report


# In[ ]:


Prepare Data for Modelling

To enable modelling, we chose 'make_model', 'review_month', 'web_browser', 'reviewer_age', 
'primary_use', 'value_for_money' as 
features, owned as target variables. I also have made the following changes:

standardize the numeric features
Convert the categorical variables into numeric features
Split the data into a training set and a test set


# In[ ]:


from sklearn.model_selection import train_test_split
X = df[['make_model', 'review_month', 'web_browser', 'reviewer_age', 'primary_use',
        'value_for_money']]
y = df['owned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


LOGISTIC REGRESSION MODEL


# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_acc = y_pred_log_reg.score(X_test,y_test)
print("logestic regression : {}".format(log_re_acc))
# Calculate the f1_score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
# Calculate the precision_score
precision = precision_score(y_test, y_pred)
print("Precision Score:", precision)


# Finding the feature importances

# In[ ]:


# Train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Get the feature importances
coefs = log_reg.coef_[0]

# Create a list of feature names
feature_names = X.columns

# Create a dataframe of feature importances
feature_importances = pd.DataFrame({'feature': feature_names, 'coef': coefs})

# Sort the dataframe by feature importance
feature_importances.sort_values(by='coef', ascending=False, inplace=True)

# Print the feature importances
print(feature_importances)


# In[ ]:


#visulizing the feature imporatance
# Plot the feature importances
plt.bar(feature_importances['feature'], feature_importances['coef'])
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Feature Importances')
plt.xticks(rotation=90)
plt.show()


# finding the best parameter
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10],
              'penalty': ['l1', 'l2']}

# Create the grid search object
grid_search = GridSearchCV(log_reg, param_grid, cv=5, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[ ]:


#visulizing it 
# Get the results of the grid search
results = grid_search.cv_results_

# Extract the mean test score for each combination of parameters
scores = results['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['penalty']))

# Create a heatmap of the test scores
plt.imshow(scores, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.xlabel('Penalty')
plt.ylabel('C')
plt.xticks(np.arange(len(param_grid['penalty'])), param_grid['penalty'])
plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
plt.title('Accuracy of Logistic Regression')
plt.show()


# You can see that the heatmap shows the accuracy of the model for different combinations of C and penalty. The darker the color, the higher the accuracy. By observing the heatmap, you can see which combinations of parameters performed better and which ones performed worse. The best combination of parameters will be the one with the highest accuracy.

# In[ ]:


RANDOMFOREST CLASSIFIER


# In[ ]:


rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train)


# In[ ]:


y_pred_rnd_clf = rnd_clf.predict(X_test)
rnd_clf_acc = y_pred_rnd_clf.score(X_test,y_test)
print("logestic regression : {}".format(rnd_clf_acc))


# In[ ]:


f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
# Calculate the precision_score
precision = precision_score(y_test, y_pred)
print("Precision Score:", precision)


# In[ ]:


finding feature importance


# In[ ]:


# Train the random forest classifier
rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train)

# Get the feature importances
importances = rnd_clf.feature_importances_

# Create a list of feature names
feature_names = X.columns

# Create a dataframe of feature importances
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort the dataframe by feature importance
feature_importances.sort_values(by='importance', ascending=False, inplace=True)

# Print the feature importances
print(feature_importances)


# In[ ]:



# Plot the feature importances
plt.bar(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


finding best paramters


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'n_estimators': [100, 200, 300],
              'max_depth': [5, 10, 15],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': [5, 10, 15]}

# Create the grid search object
grid_search = GridSearchCV(rnd_clf, param_grid, cv=5, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[ ]:


# visulizing with matplotlib or seaborn as u wish
import matplotlib.pyplot as plt

# Extract the results of the grid search
results = grid_search.cv_results_

# Extract the mean test scores
mean_test_scores = results['mean_test_score']

# Extract the standard deviation of the test scores
std_test_scores = results['std_test_score']

# Extract the parameters that were tested
params = results['params']

# Plot the mean test scores
plt.errorbar(range(len(params)), mean_test_scores, yerr=std_test_scores)
plt.xlabel('Parameter Combination')
plt.ylabel('Mean Test Score')
plt.show()
#This code will create a plot that shows the mean test score for each parameter 
#combination tested in the grid search, along with the standard deviation of
#the test scores. The x-axis shows the index of the parameter combination, and 
#the y-axis shows the mean test score.


# In[ ]:


import seaborn as sns

# Convert the results of the grid search to a dataframe
results_df = pd.DataFrame(grid_search.cv_results_)

# Create a heatmap of the test scores
sns.heatmap(results_df.pivot_table(values='mean_test_score', index='param_n_estimators', columns='param_max_depth'))
plt.xlabel('Max Depth')
plt.ylabel('Number of Estimators')
plt.show()
#This will create a heatmap where the x-axis shows the value of the max_depth parameter, 
#the y-axis shows the value of the n_estimators parameter and 
#the color of the cells represents the mean test score.


# why i choose them to be my evaluation
#  the precision_score metric  focus on the
# model's ability to correctly predict the positive class, specifically
# minimizing the number of false positives.
# it ranges between 0 and 1, where 1 represents a perfect score and 0 represents a poor score.
# Precision is a measure of how many of 
# the positive predictions were actually correct.
# 
# 
# 
# f1_score metric balance precision and recall and get a single number that
# represents the overall performance of the model.
#  It ranges between 0 and 1, where 1 represents a perfect score and 0 represents a poor score.
# F1 score is a better measure than accuracy, especially if you have 
# an uneven class distribution.
# 
# 

# In[ ]:


resulst
F1 score and precision score are both metrics used to evaluate the performance of a classifier. F1 score is a harmonic mean of precision and recall, while precision is the proportion of true positive predictions out of all positive predictions.

To compare the performance of logistic regression and random forest classifier,
you can calculate the F1 score and precision score for each model on a set of test data. 
If one model has a consistently higher F1 score and 
precision score than the other, it can be considered to be the better performing model.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




