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

from sklearn.preprocessing import OneHotEncoder

import math
import missingno as msno

def Get_Data_Path():
    """ Return le chemin des fichiers dans le répértoire Data
    Parameter: sans
    Return: le chemin des fichiers."""
    Dir = os.getcwd()
    parentDir = os.path.dirname(Dir)
    DataDir = os.path.join(parentDir, 'Data')
    return DataDir
    
def Get_Files(path):
    """ Return les fichiers dans le répértoire Data
    Parameter: Chemin des fichiers.
    Return: les fichiers."""
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files

def Read_Files(files, path):
    """ lit les fichiers et retourne dataFrame
    Parameter: nom de fichiers, chemin de fichiers.
    Return: liste de dataFrame."""
    df_list = []
    for file in files:
        file_path = os.path.join(path, file)
        df_file_name = pd.read_csv(file_path)
        df_list.append(df_file_name)    
    return df_list

def compare(df1, df2):
    L_15 = list(df1.columns)
    L_16 = list(df2.columns)

    print(10*"=","Elements in 2015 but absent in 2016", 10*"=")
    print(list(set(L_15)-set(L_16)))
    print()
    print(10*"=","Elements in 2016 but absent in 2015", 10*"=")
    print(list(set(L_16)-set(L_15)))

def nan_check(df):
    """ Function to list the percentage of NaNs in each column"""
    (df.isna().sum()/df.shape[0]).sort_values(ascending=True)
    print ((df.isna().sum()/df.shape[0]).sort_values(ascending=True))

def nan_drop(df, thres):
    """ Function to remove columns that contain more than threshold NaN"""
    df = df[df.columns[df.isna().sum()/df.shape[0] < thres]]
    return df

def search_componant(df, suffix=None):
    componant = []
    for col in df.columns:
        if suffix in col: 
            componant.append(col)
    return componant


def plot_bi_var (variable,target,df):
    # function to display (categorical variables and target variables)
    sns.set_style("ticks")
    plt.figure(figsize=(10,5))
    ax = sns.boxplot(x=variable, y=target, data=df, width=0.5,showfliers=False, showmeans=True)
    
    plt.xlabel(variable,size=14)
    plt.ylabel(target,size=14)
    plt.title(f"\n Distribution of {target} virsus {variable}",size=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    plt.show()

def plot_tar_num(var_num,target,df):
    # function to display (numerical variables and target variables)
    sns.set_style("ticks")
    plt.figure(figsize=(10,5))
    ax = sns.scatterplot(data = df, x = var_num, y = target, palette='bright')
    ax.set_xlabel(var_num, size=14)
    ax.set_ylabel(target, size=14)
    plt.title(f"\n Distribution of {target} virsus {var_num}",size=16)
    plt.show()

def detect_outliers_zscore(df,thres):
    outliers = []
    mean = df.mean()
    std = df.std()
    # print(mean, std)
    for i in df:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
    return outliers

def detect_outliers_iqr(df, thres):
    outliers = []
    df = sorted(df)
    q1 = np.percentile(df, 25)
    q3 = np.percentile(df, 75)
    #print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(thres*IQR)
    upr_bound = q3+(thres*IQR)
    #print(lwr_bound, upr_bound)
    for i in df: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers

def handling_outliers_percentile(df, col, mini, maxi):
    # Computing minith, maxith percentiles and replacing the outliers
    min_percentile = np.percentile(df[col], mini)
    max_percentile = np.percentile(df[col], maxi)
    df.loc[df[col]>max_percentile, col] = max_percentile
    df.loc[df[col]<min_percentile, col] = min_percentile