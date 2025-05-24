# Python Code to build a Linear Regression Model with the Scikit Library

## Importing Library
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import sklearn library to perform L1, L2 & LR
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Import sklearn library to evaluate model using MAE, MSE & MAPE
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error, r2_score

pd.options.mode.chained_assignment = None # Disable warning from pandas library
# pd.options.mode.chained_assignment = 'warn' # enable warning from pandas library
```

## Read CSV file
```
dataDF = pd.read_csv('SNRE_Data_w_Dummy_for_SPSS.csv') 

#change the denomination of some variables e.g. Population, GDP, Land Supply
dataDF['Total Population'] = dataDF['Total Population']/10000 
dataDF['Real GDP (in $Sm)'] = dataDF['Real GDP (in $Sm)']/10000 
dataDF['Total Multiple-User Factory Space (m2)'] = dataDF['Total Multiple-User Factory Space (m2)']/10000 
dataDF.rename(columns={"Total Population": "Total Population (in 10,000)", 
                       "Real GDP (in $Sm)": "Real GDP (in $S10,000m)",
                       "Total Multiple-User Factory Space (m2)": "Total Multiple-User Factory Space (10,000m2)"},
              inplace = True)

x = dataDF.drop(['Unit Price ($ psf)','Floor Level','First Floor','Type of Sale', 'New Sale',
                 'Central Region','Region'], axis=1)

y = dataDF['Unit Price ($ psf)']
```

## Split the Dataset into Train/Test portion
```
# Split the dataset into the training and testing set.
input_train, input_test, target_train, target_test = train_test_split(x, y, 
                                                                      test_size = 0.2,
                                                                      random_state = 42,
                                                                      shuffle = True) 
```

## L2 Regression Model
```
#Building the Ridge Regression model with Training Data
l2Model = Ridge(alpha = 58.3) 
l2Model.fit(input_train, target_train)

l2Model_coef = [(x, y) for x, y in zip(l2Model.feature_names_in_, l2Model.coef_.round(3))]
l2Model_coef.append(('Intercept', l2Model.intercept_.round(2)))
display(l2Model_coef)

# Do prediction with l2Model with var_test (Testing set)
l2_pred = l2Model.predict(input_test)
l2_score = l2Model.score(input_test, target_test)

# Compute Error metrics for Ridge Regression Model
l2pred_mae = mean_absolute_error(target_test,l2_pred)
l2pred_mse = mean_squared_error(target_test,l2_pred)
l2pred_mape = mean_absolute_percentage_error(target_test,l2_pred)
l2pred_max = max_error(target_test,l2_pred)

print('Mae: {}'.format(l2pred_mae.round(2)))
print('Mse: {}'.format(l2pred_mse.round(2)))
print('Mape: {}'.format(l2pred_mape.round(2)))
print('Max: {}'.format(l2pred_max.round(2)))
print('R2 Score: {}'.format(l2_score.round(2)))
```

## Hyperparameter finetuning
```
from sklearn.model_selection import GridSearchCV

# Setup the Ridge Regression Model - ridgeModel
ridgeModel = Ridge()

# set the search space - hyperParameters
hyperParameters = {'alpha':np.arange(55,60.1,0.1)}

# Create the GridSearchCV object
l2GS = GridSearchCV(estimator = ridgeModel,  #This is the model i trying to optimise
                    param_grid = hyperParameters, # What is the Hyperparameter am I trying to finetune.
                    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error',
                               'neg_mean_absolute_percentage_error', 'r2'], # what is the metric I'm going to use.
                    cv = 10,
                    refit = 'neg_mean_absolute_error',
                    verbose = 0)

print(len(hyperParameters['alpha']))

l2GS.fit(input_train, target_train)

print(l2GS.best_params_)
print(l2GS.best_score_)
print(l2GS.best_estimator_)

l2_df = pd.DataFrame(l2GS.cv_results_)
l2error_df = l2_df[['param_alpha',
                    'mean_test_neg_mean_absolute_error']]
#df.to_csv('results.csv')

display(l2error_df.abs())

sns.lineplot(data=l2error_df.abs(), x="param_alpha", y="mean_test_neg_mean_absolute_error")
```
