# Import of all the libraries we will need
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

from os import listdir
from os.path import isfile, join
import ast
import datetime
import folium

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import chi2

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from category_encoders import TargetEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import math
import missingno as msno

# check xgboost version
import xgboost
#print(xgboost.__version__)

def preprocessing_co2(df):
# We create the dependant and independant features.
    x = df.drop(['TotalGHGEmissions'], axis =1)
    y_co2 = df.loc[:,'TotalGHGEmissions']
    return x,y_co2

def preprocessing(df):
# We create the dependant and independant features.
    x = df.drop(['SiteEnergyUse(kBtu)'], axis =1)
    y_enrg = df.loc[:,'SiteEnergyUse(kBtu)']
    return x,y_enrg

# feature selection
def correlation_select_features(x_train, y_train):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    x_train_fs = fs.transform(x_train)
    return x_train_fs, fs
    
def correlation(dataset,threshold):
    # set of all the name of correlated columns
    col_corr = set() 
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # we are interested to absolute value
            if abs(corr_matrix.iloc[i,j])>threshold:
                # getting the name of column
                col_name = corr_matrix.columns[i] 
                col_corr.add(col_name)
    return col_corr

# feature selection for mutual information
def mutual_select_features(x_train, y_train):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    x_train_fs = fs.transform(x_train)
    return x_train_fs, fs

# prepare input data
def prepare_inputs(x_train, y_train):
    te = TargetEncoder() #OneHotEncoder()
    te.fit(x_train, y_train)
    x_train_enc = te.transform(x_train)
    return x_train_enc

def best_score_model(model, Column_Norm_Trans, param_grid_model, x_train, y_train, x_test, y_test, score):
    pipe_model = make_pipeline(Column_Norm_Trans, model)
    pipe_grid_model = GridSearchCV(pipe_model, param_grid_model, cv=3, scoring=score, return_train_score=True)
    #pipe_grid_model = GridSearchCV(pipe_model, param_grid_model, cv=3, return_train_score=True)
    pipe_grid_model.fit(x_train, y_train)
    return pipe_grid_model

def prediction(model, x_train, y_train, x_test, y_test):
    # We extract the best estimator model and we can use it as a predictive model.
    best_model = model.best_estimator_
    print("best parameters: ", model.best_params_)
    print("best model: ", best_model)
    # We will fit again with train data and check the accuracy metrics.
    best_model.fit(x_train,y_train)
    ytr_pred = best_model.predict(x_train)
    mse = mean_squared_error(y_train, ytr_pred)
    r2 = r2_score(y_train, ytr_pred)
    print("MSE with train data: %.2f" % mse)
    print("R2  with train data: %.2f" % r2)

    #Next, we'll predict test data and check the accuracy metrics.
    ypred=best_model.predict(x_test)
    mse = mean_squared_error(y_test, ypred)
    r2 = r2_score(y_test, ypred)
    print("MSE with test data: %.2f" % mse)
    print("R2  with test data: %.2f" % r2)

    #Finally, we'll visualize the results in a plot.
    #x_ax = [np.exp(y_test).min(), np.exp(y_test).max()]
    x_ax = [y_test.min(), y_test.max()]
    
    plt.figure(figsize=(12,8))
    #plt.scatter(np.exp(y_test), np.exp(ypred), s=7, color="blue", label="Predicted")
    plt.scatter(y_test, ypred, s=7, color="blue", label="Predicted")
    plt.plot(x_ax, x_ax, lw=1, color='red', label="Real")
    plt.legend()
    plt.xlabel("Real values", fontsize=14)
    plt.ylabel("predict values", fontsize=14)
    plt.title("Predict values VS Real values",fontsize=14)   
    plt.show()

def feature_importance (model, x_train):
    plt.figure(figsize=(12,8))
    plt.barh(x_train.columns, np.sort(model.best_estimator_._final_estimator.feature_importances_))
    plt.xlabel("Scores", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.title("Feature importance",fontsize=14)   
    plt.show()
    
def x_y_split(trainset, testset, f_num, f_cat, y_enrgy_train, y_enrgy_test):
    x_train = trainset[f_num+f_cat]
    y_train =  np.log(y_enrgy_train +1)
    x_test = testset[f_num+f_cat]
    y_test = np.log(y_enrgy_test+1)
    return x_train, y_train, x_test, y_test
    
def column_trans(f_num, f_cat):
    #column_norm_trans = make_column_transformer(
    #                                             (StandardScaler(), f_num),
    #                                             (OneHotEncoder(use_cat_names=True, handle_unknown = "ignore"), f_cat))
    column_norm_trans = make_column_transformer(
                                                 (StandardScaler(), f_num),
                                                 (TargetEncoder(), f_cat))                                             
    return column_norm_trans
