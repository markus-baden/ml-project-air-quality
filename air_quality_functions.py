
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import math
import gc

from pyts.preprocessing import InterpolationImputer
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import GridSearchCV


def calc_atmos_press(h):
    # atmospheric pressure equation (in dependence of altitude) (see also: https://en.wikipedia.org/wiki/Atmospheric_pressure)
    p0 = 101325.0       # Pa        Sea level standard atmospheric pressure
    g  = 9.80665        # m/s^2     Earth-surface gravitational acceleration
    M  = 0.02896968     # kg/mol    Molar mass of dry air
    R0 = 8.314462618    # J/(mol·K) Universal gas constant
    T0 = 288.16         # K         Sea level standard temperature
    #h                  # m         Height above Earth-surface
    return p0 * np.exp( -(g * h * M) / (T0 * R0))

def correct_pressure(data):
    for feature in data.columns:
        if ('atmos_press' in feature) and ('std' not in feature) and ('var' not in feature) and ('ptp' not in feature):
            data[feature] = data[feature] - data['std_pressure']
    return data.drop('std_pressure', axis=1)

def hazard_level(pm_2_5):
    if pm_2_5 <= 12.0:
        return 'good'
    elif 12.0 < pm_2_5 <= 35.0:
        return 'moderate'
    elif 35.0 < pm_2_5 <= 55.0:
        return 'unhealthy for sensitives'
    elif 55.0 < pm_2_5 <= 150.0:
        return 'unhealthy'
    elif 150.0 < pm_2_5 <= 250.0:
        return 'very unhealthy'  
    elif 250.0 < pm_2_5:
        return 'hazardous'

def agg_diffs(data, features, blocksize):
    for feature in features:
        for i in range(int(121/blocksize) -1):
            data[feature + '_' + str(i) + 'to' + str(i+1)] = data[feature + '_' + str(i+1)] - data[feature + '_' + str(i)]
    return data

def replace_nan(x):
    if x==" ":
        return np.nan
    else :
        return float(x)

def convert_str_to_list(data, features):
    for feature in features : 
        data[feature]=data[feature].apply(lambda x: [ replace_nan(X) for X in x.replace("nan"," ").split(",")])
    return data

def time_blocks(data, features, blocksize):
    for feat in features:
        for x in range(int(121/blocksize)):
            sum = pd.Series(np.zeros(len(data)))
            sum.index = data.index
            for j in range(blocksize):
                index = (x * blocksize + j)
                if blocksize > 1:
                    index += 1
                sum += data[feat].str[index]
            data[feat+ '_'+ str(x)] = sum / blocksize
    return data.drop(features,axis=1)

def impute_func(values, imputer):
    num_of_nans = np.isnan(values).sum()
    values = imputer.transform([values, np.arange(0, len(values))])
    return [list(values[0]), num_of_nans]


def impute_values(data, features, strategy='nearest'):
    imputer = InterpolationImputer(strategy=strategy)
    for feature in features:
        data[feature + '_nans'] = np.zeros(data.shape[0])

    for idx in data.index:
        for feature in features:
            data[feature][idx], num_of_nans =impute_func(data[feature][idx], imputer)
            data[feature + '_nans'][idx] = num_of_nans
    return data


def aggregate_features(x,col_name):
    x["max_"    + col_name] = x[col_name].apply(np.max)
    x["min_"    + col_name] = x[col_name].apply(np.min)
    x["mean_"   + col_name] = x[col_name].apply(np.mean)
    x["std_"    + col_name] = x[col_name].apply(np.std)
    x["var_"    + col_name] = x[col_name].apply(np.var)
    x["median_" + col_name] = x[col_name].apply(np.median)
    x["ptp_"    + col_name] = x[col_name].apply(np.ptp)
    return x  

def train_predict_evaluate(model, X_train, y_train, X_test, y_test):
    """Train model, make prediction and evaluate

    Args:
        model (_type_): Regressor model
        X_train (_type_): Train data features
        y_train (_type_): Train data target
        X_test (_type_): Test data features
        y_test (_type_): Test data target
    """
    # train model
    model.fit(X_train, y_train)

    # make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # print metrics of predictions
    print(f"Model: {type(model)}") 
    print(f"RMSE on train data: {mean_squared_error(y_train, y_train_pred, squared=False)}")
    print(f"RMSE on test data: {mean_squared_error(y_test, y_test_pred, squared=False)}")
    print('-----------------------------')

def grid_search_evaluate(model, hyperparameters, X_train, y_train, X_test, y_test):
    """make grid search for model parameters, get the best estimator, make prediction and evaluate

    Args:
        model (_type_): Regressor model
        hyperparameters (_type_): parameter list for the grid search
        X_train (_type_): Train data features
        y_train (_type_): Train data target
        X_test (_type_): Test data features
        y_test (_type_): Test data target
    """
    
    gs = GridSearchCV(model, hyperparameters, n_jobs=-1, scoring='neg_root_mean_squared_error', cv=3, verbose = 0)

    # Fit the grid search object to the training data and find the optimal parameters
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_

    # make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # print metrics of predictions
    print(f"Model: {type(best_model)}") 
    print(f"RMSE on train data: {mean_squared_error(y_train, y_train_pred, squared=False)}")
    print(f"RMSE on test data: {mean_squared_error(y_test, y_test_pred, squared=False)}")
    print('-----------------------------')

    return best_model
