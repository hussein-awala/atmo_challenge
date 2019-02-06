# -*- coding: utf-8 -*-
from utils import load_data, get_data_day
import numpy as np
import random

import pickle
import datetime
import calendar
import time
import os

import utils
import pandas as pd
import logging
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'model++.sav')


#### Predict function for a given day

def predict(day, sites, chimeres_day, geops_day, meteo_day, concentrations_day, model=None):
    """
    day: day of the year (1-365)

    sites : Dataframe with columns "idPolair","nom_station","coord_x_l93","coord_y_l93","X_lamb2","Y_lamb2", "LON" ,"LAT",
"Département","Zone_EPCI","typologie","NO2_influence", "NO2_2012", "NO2_2013","NO2_2014","NO2_2015", "NO2_2016","NO2_2017","O3_influence","O3_2012",  "O3_2013", "O3_2014", "O3_2015","O3_2016" "PM10_2017","PM25_influence" "PM25_2012","PM25_2013","PM25_2014","O3_2017","PM10_influence" "PM10_2012","PM10_2013","PM10_2014","PM10_2015","PM10_2016","PM25_2015","PM25_2016","PM25_2017".
chimeres_day Dict on Pollutants, for each pollutant a Dataframe with columns 'date', 'val', 'idPolair', 'param'. Stopped at D0+72H

geops_day : Dict on sites, for each site a Dataframe with columns 'date', 'idPolair', 'geop_p_500hPa', 'geop_p_850hPa'. Stopped at D0+6H

meteo_day : Dataframe with columns "date", "idPolair", "T2", "Q2", "U10", "V10" "PSFC", "PBLH", "LH", "HFX", "ALBEDO", "SNOWC", "HR2", "VV10", "DV10", "PRECIP". Stopped at D0+6H

concentrations_day : Dict on Pollutants, for each pollutant, a dataframe with columns 'idPolair', 'Organisme', 'Station', 'Mesure', 'Date', 'Valeur'. Stopped at D0+6H
model : pretrained model data (e.g. saved learned sklearn model) if you have one. Change its default value with a relative path if you want to load a file
    """

    # Prediction step: up to you !

    # result format
    # dict[pol][site][horizon][array of size 24 (hourly prediction)]
    results = dict({})
    X = get_X(sites, chimeres_day, geops_day, meteo_day, concentrations_day)
    model = loaded_model = pickle.load(open(filename, 'rb'))
    for pol in ["PM10", "PM25", "O3", "NO2"]:
        results[pol] = dict({})
        for idPolair in sites.idPolair:
            results[pol][idPolair] = dict({})
        results[pol]['33374'] = dict({})
    for idPolair in sites.idPolair:
        if int(idPolair) in X:
            y_predict = model.predict(np.array([X[int(idPolair)]]))
            results["NO2"][idPolair]["D0"] = np.full((24), y_predict[0, 0])
            results["NO2"][idPolair]["D1"] = np.full((24),y_predict[0, 4])
            results["NO2"][idPolair]["D2"] = np.full((24),y_predict[0, 8])
            results["O3"][idPolair]["D0"] = np.full((24),y_predict[0, 1])
            results["O3"][idPolair]["D1"] = np.full((24),y_predict[0, 5])
            results["O3"][idPolair]["D2"] = np.full((24),y_predict[0, 9])
            results["PM10"][idPolair]["D0"] = np.full((24),y_predict[0, 2])
            results["PM10"][idPolair]["D1"] = np.full((24),y_predict[0, 6])
            results["PM10"][idPolair]["D2"] = np.full((24),y_predict[0, 10])
            results["PM25"][idPolair]["D0"] = np.full((24),y_predict[0, 3])
            results["PM25"][idPolair]["D1"] = np.full((24),y_predict[0, 7])
            results["PM25"][idPolair]["D2"] = np.full((24),y_predict[0, 11])
        else:
            rand=random.choice(list(X))
            y_predict = model.predict(np.array([X[rand]]))
            results["NO2"][idPolair]["D0"] = np.full((24), y_predict[0, 0])
            results["NO2"][idPolair]["D1"] = np.full((24), y_predict[0, 4])
            results["NO2"][idPolair]["D2"] = np.full((24), y_predict[0, 8])
            results["O3"][idPolair]["D0"] = np.full((24), y_predict[0, 1])
            results["O3"][idPolair]["D1"] = np.full((24), y_predict[0, 5])
            results["O3"][idPolair]["D2"] = np.full((24), y_predict[0, 9])
            results["PM10"][idPolair]["D0"] = np.full((24), y_predict[0, 2])
            results["PM10"][idPolair]["D1"] = np.full((24), y_predict[0, 6])
            results["PM10"][idPolair]["D2"] = np.full((24), y_predict[0, 10])
            results["PM25"][idPolair]["D0"] = np.full((24), y_predict[0, 3])
            results["PM25"][idPolair]["D1"] = np.full((24), y_predict[0, 7])
            results["PM25"][idPolair]["D2"] = np.full((24), y_predict[0, 11])
    results["NO2"]['33374']["D0"] = np.full((24), 0)
    results["NO2"]['33374']["D1"] = np.full((24), 0)
    results["NO2"]['33374']["D2"] = np.full((24), 0)
    results["O3"]['33374']["D0"] = np.full((24), 0)
    results["O3"]['33374']["D1"] = np.full((24), 0)
    results["O3"]['33374']["D2"] = np.full((24), 0)
    results["PM10"]['33374']["D0"] = np.full((24), 0)
    results["PM10"]['33374']["D1"] = np.full((24), 0)
    results["PM10"]['33374']["D2"] = np.full((24), 0)
    results["PM25"]['33374']["D0"] = np.full((24), 0)
    results["PM25"]['33374']["D1"] = np.full((24), 0)
    results["PM25"]['33374']["D2"] = np.full((24), 0)

    return results


#### Main loop (no need to be changed)

def run_predict(year=2016, max_days=10, dirname="../Data/training", list_days=None):
    """
    year : year to be evaluated
    max_days: number of past days allowed to predict a given day (set to 10 on the platform)
    dirname: path to the dataset
    list_days: list of days to be evaluated (if None the full year is evaluated)
    """

    overall_start = time.time()  # <== Mark starting time
    data = load_data(year=year, dirname=dirname)  # load all data files
    sites = data["sites"]  # get sites info
    day_results = dict({})
    if list_days is None:
        if calendar.isleap(year):  # check if year is leap
            list_days = range(366)
        else:
            list_days = range(365)
    for day in list_days:
        print(day)
        chimeres_day, geops_day, meteo_day, concentrations_day = get_data_day(day, data, max_days=max_days,
                                                                              year=year)  # you will get an extraction of the year datasets, limited to the past max_days for each day
        day_results[day] = predict(day, sites, chimeres_day, geops_day, meteo_day,
                                   concentrations_day)  # do the prediction

    overall_time_spent = time.time() - overall_start  # end computation time
    pickle.dump(day_results, open('submission/results.pk', 'wb'))  # save results
    pickle.dump(overall_time_spent, open('submission/time.pk', 'wb'))  # save computation time


def data_station_year(sites, chimeres_day, geops_day, meteo_day, concentrations_day):
    logger = logging.getLogger(__name__)

    t1 = chimeres_day['NO2']
    t1.drop(columns='param', inplace=True)
    t1['date'] = pd.to_datetime(t1['date'])
    t1.set_index(['idPolair', 'date'], inplace=True)
    t1.columns = ['NO2_chimere']
    t2 = chimeres_day['O3']
    t2.drop(columns='param', inplace=True)
    t2['date'] = pd.to_datetime(t2['date'])
    t2.set_index(['idPolair', 'date'], inplace=True)
    t2.columns = ['O3_chimere']
    t3 = chimeres_day['PM10']
    t3.drop(columns='param', inplace=True)
    t3['date'] = pd.to_datetime(t3['date'])
    t3.set_index(['idPolair', 'date'], inplace=True)
    t3.columns = ['PM10_chimere']
    t4 = chimeres_day['PM25']
    t4.drop(columns='param', inplace=True)
    t4['date'] = pd.to_datetime(t4['date'])
    t4.set_index(['idPolair', 'date'], inplace=True)
    t4.columns = ['PM25_chimere']
    t5 = pd.merge(t1, pd.merge(t2, pd.merge(t3, t4, left_index=True, right_index=True, how='outer'), left_index=True,
                               right_index=True, how='outer'), left_index=True, right_index=True, how='outer')
    t5.sort_index(inplace=True)

    meteo = meteo_day

    meteo['date'] = pd.to_datetime(meteo['date'])

    meteo.set_index(['idPolair', 'date'], inplace=True)
    meteo.sort_index(inplace=True)

    allData = pd.merge(meteo, t5, left_index=True, right_index=True, how='outer')
    geops = geops_day
    keys = np.fromiter(geops.keys(), dtype=float)

    z = pd.concat(geops.values(), keys=keys)
    z['date'] = pd.to_datetime(z['date'])
    z.set_index(['idPolair', 'date'], inplace=True)
    z.sort_index(inplace=True)

    allData = pd.merge(allData, z, left_index=True, right_index=True, how='outer')

    allData = allData.reset_index()
    allData.set_index(['idPolair', 'date'], inplace=True)

    allData.interpolate(inplace=True)

    if allData.isnull().any().any():
        allData.fillna(allData.mean(), inplace=True)
    if allData.isnull().any().any():
        allData.fillna(0, inplace=True)
    return allData


def to_sequances(data,t):
    data.reset_index(inplace=True)
    data.set_index(['idPolair', 'date'], inplace=True)
    g = data.groupby('idPolair')
    sites = np.array(['15013', '15017', '15018', '15031', '15038', '15039', '15043',
                           '15045', '15046', '15048', '15049', '20004', '20013', '20017',
                           '20019', '20029', '20031', '20037', '20045', '20046', '20047',
                           '20048', '20049', '20061', '20062', '20063', '20065', '20069',
                           '20070', '27002', '27003', '27004', '27005', '27007', '27008',
                           '27010', '29421', '29423', '29424', '29426', '29428', '29429',
                           '29439', '29440', '29441', '33101', '33102', '33111', '33114',
                           '33120', '33121', '33122', '33201', '33202', '33211', '33212',
                           '33220', '33232', '33302', '33305', '33414', '33105', '33203',
                           '33233', '33235', '33367', '33213', '33303', '15114', '36001',
                           '36002', '36003', '36005', '36019', '36021', '07009', '07004',
                           '07001', '07034', '07039', '07042', '07051', '07053', '07032',
                           '07045', '07028', '07013', '07016', '07010', '07049', '07048',
                           '07017', '07014', '07011', '07054', '07012', '07018', '07015',
                           '07020', '07022', '07031', '07029', '07043', '07052', '07056',
                           '07057', '07058']).astype(int)
    groups = g.groups
    result = dict({})
    for site in sites:
        group = g.get_group(site)
        group.reset_index(inplace=True)
        group['date'] = pd.to_datetime(group['date'])
        group.set_index('date', inplace=True)
        group = group.resample('D').mean()
        X = to_window(group,t)
        result[site] = X

    return result


def to_window(data,t):
    matrix = data.values[:, 1:]
    matrix = matrix[::-1]
    matrix = matrix[:t]
    i = matrix.shape[0]
    m = matrix.mean(0)
    while (i - t) < 0:
        matrix = np.append(matrix, m)
        i = i + 1
    vector = np.append(matrix, [])
    return vector


def get_X(sites, chimeres_day, geops_day, meteo_day, concentrations_day):
    data = data_station_year(sites, chimeres_day, geops_day, meteo_day, concentrations_day)
    return to_sequances(data,2)
