# Python Code to build a Random Forest Regressor Model with the Scikit Library

## Note:
The current codebase does not adhere to the best practices of Software Engineering/Machine Learning Engineering.
Refactoring of this codebase is a WIP

## Importing Library
```
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import sklearn library to Build Tree - Based Model
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

#Import sklearn library to evaluate model using MAE, MSE & MAPE
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error, r2_score
```

## Data Preparation
```
dataDF = pd.read_csv('SNRE_Data_w_Dummy_for_SPSS.csv')

# Separate Input variables (x) and target variable (y)
x = dataDF.drop(['Unit Price ($ psf)','Floor Level','First Floor','Type of Sale', 'New Sale',
                 'Central Region','Region'], axis=1)
y = dataDF['Unit Price ($ psf)']

print(x.info())
print(y.info())

#Split the dataset into the training and testing set.
input_train, input_test, target_train, target_test = train_test_split(x, y, test_size=0.2, random_state=42,
                                                                  shuffle = True) 
print('Training Set has {} rows'.format(input_train.shape[0]))
print('Testing Set has {} rows'.format(input_test.shape[0]))

```

## RandomForestRegressor 
```
default_rfModel = RandomForestRegressor(n_estimators = 100, criterion = 'absolute_error', 
                                oob_score = True,random_state = 42) 

default_rfModel.fit(input_train, target_train)

default_rfModel_pred = default_rfModel.predict(input_test)
default_rfModel_score = default_rfModel.score(input_test, target_test)

# Compute Error metrics for RF Model
default_rfpred_mae = mean_absolute_error(target_test,default_rfModel_pred)
default_rfpred_mse = mean_squared_error(target_test,default_rfModel_pred)
default_rfpred_mape = mean_absolute_percentage_error(target_test,default_rfModel_pred)
default_rfpred_max = max_error(target_test,default_rfModel_pred)

print('Mae: {}'.format(default_rfpred_mae.round(2)))
print('Mse: {}'.format(default_rfpred_mse.round(2)))
print('Mape: {}'.format(default_rfpred_mape.round(2)))
print('Max: {}'.format(default_rfpred_max.round(2)))
print('R2: {}'.format(default_rfModel_score.round(2)))
```

## Ranking features based on importance - Get Top 5 important variables
```
def getInputRank(rfModel):
    rf_inputs_rank = pd.DataFrame({'Input':rfModel.feature_names_in_,
                                   'Rank' :rfModel.feature_importances_.round(5)*100})
    rf_inputs_rank.sort_values(by='Rank', ascending = False, inplace=True)
    rf_inputs_rank.reset_index(drop=True, inplace=True)
    display(rf_inputs_rank)
    plt.figure(figsize=(8, 3))  # Set the figure size
    sns.barplot(x=rf_inputs_rank["Rank"].head(5), y=rf_inputs_rank["Input"].head(5))
    plt.title('Top 5 Variables (Higher Score, Higher Importance)')
    plt.tight_layout()  # Adjust subplot spacing
    plt.show()
    return rf_inputs_rank

InputRankDF = getInputRank(default_rfModel)
```

## Tune Random Forest Model hyperparameter with GridSearch
```
# Setup the RF Model - randomForest
randomForest = RandomForestRegressor(random_state = 42)

# set the search space - hyperParameters
hyperParameters = {'n_estimators':np.arange(98,101,1),
                   'max_depth':np.arange(14,19,1)}    

# Create the GridSearchCV object
rfGS = GridSearchCV(estimator = randomForest,  #This is the model i trying to optimise
                    param_grid = hyperParameters, # What is the Hyperparameter am I trying to finetune.
                    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 
                               'neg_mean_absolute_percentage_error', 'r2'], # what is the metric I'm going to use.
                    cv = 10,
                    refit = 'neg_mean_absolute_error',
                    verbose = 1)
rfGS.fit(input_train, target_train)
print(rfGS.best_params_)
print(rfGS.best_score_)
print(rfGS.best_estimator_)

```

## Retrain Random Forest Model with updated hyperparameter and get Top 5 variables
```
rfModel_tuned = RandomForestRegressor(max_depth = 15, n_estimators = 100, criterion = 'absolute_error', 
                                oob_score = True,random_state = 42) 

rfModel_tuned.fit(input_train, target_train)

rfModel_tuned_pred = rfModel_tuned.predict(input_test)
rfModel_tuned_score = rfModel_tuned.score(input_test, target_test)

# Compute Error metrics for RF Model
rfpred_tuned_mae = mean_absolute_error(target_test,rfModel_tuned_pred)
rfpred_tuned_mse = mean_squared_error(target_test,rfModel_tuned_pred)
rfpred_tuned_mape = mean_absolute_percentage_error(target_test,rfModel_tuned_pred)
rfpred_tuned_max = max_error(target_test,rfModel_tuned_pred)

print('Mae: {}'.format(rfpred_tuned_mae.round(2)))
print('Mse: {}'.format(rfpred_tuned_mse.round(2)))
print('Mape: {}'.format(rfpred_tuned_mape.round(2)))
print('Max: {}'.format(rfpred_tuned_max.round(2)))
print('R2: {}'.format(rfModel_tuned_score.round(2)))

InputRankDF = getInputRank(rfModel_tuned)
top5Input = InputRankDF['Input'].head(5).tolist()
print(top5Input)
```

## Build Parimoney Random Forest Model with Top 5 variable
```
five_input_train = input_train[top5Input]
five_input_test = input_test[top5Input]

parsi_rfModel = RandomForestRegressor(n_estimators = 100, criterion = 'absolute_error', 
                                    oob_score = True,random_state = 42) 
parsi_rfModel.fit(five_input_train, target_train)

parsi_rfModel_pred = parsi_rfModel.predict(five_input_test)
parsi_rfModel_score = parsi_rfModel.score(five_input_test, target_test)

# Compute Error metrics for RF Model
parsi_rfModel_mae = mean_absolute_error(target_test,parsi_rfModel_pred)
parsi_rfModel_mse = mean_squared_error(target_test,parsi_rfModel_pred)
parsi_rfModel_mape = mean_absolute_percentage_error(target_test,parsi_rfModel_pred)
parsi_rfModel_max = max_error(target_test,parsi_rfModel_pred)

print('Mae: {}'.format(parsi_rfModel_mae))
print('Mse: {}'.format(parsi_rfModel_mse))
print('Mape: {}'.format(parsi_rfModel_mape))
print('Max: {}'.format(parsi_rfModel_max))
print('R2: {}'.format(parsi_rfModel_score))
print('Time: {} sec'.format(fit_time_3))
```
