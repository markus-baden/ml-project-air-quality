
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import math
import gc
from pyts.preprocessing import InterpolationImputer

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

def impute_func(values, imputer, max_num_of_nans=80):
    num_of_nans = np.isnan(values).sum()
    values = imputer.transform([values, np.arange(0, len(values))])
    return [list(values[0]), num_of_nans]


def impute_values(data, features, strategy='nearest'):
    imputer = InterpolationImputer(strategy=strategy)
    for feature in features:
        data[feature + '_nans'] = np.zeros(data.shape[0])

    for idx in data.index:
        for feature in features:
            data[feature][idx], num_of_nans =impute_func(data[feature][idx], imputer, max_num_of_nans=80)
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
