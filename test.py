from utils import load_data, get_data_day
import numpy as np
import pandas as pd

import pickle
import datetime
import calendar
import time
import os

import utils
import logging
import pickle


logging.basicConfig(level=logging.INFO)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Masking
from keras.models import load_model
from keras import backend as K



dirname = os.path.dirname(__file__)




def predict(day, sites, chimeres_day, geops_day, meteo_day, concentrations_day, model=None):

    X=toPredcit(concentrations_day)
    
    #define a dictionary to return the result
    results = dict({})
    
    #add the 4 pol to the dict
    for pol in ["PM10", "PM25", "O3", "NO2"]:
        results[pol] = dict({})
        #add each idPolair to each pol in the dict
        #sites=np.load('models/sites.npy')
        for idPolair in sites.idPolair:
            results[pol][''+str(int(idPolair))] = dict({})
        
    #results[pol]['33374'] = dict({})
    
    #for each pol predict and add the result
    for index,row in X.iterrows():
        
        #remove old model from GPU to speed up the model
        K.clear_session()
        
        #load the model of this site
        filename = os.path.join(dirname,'models/'+ str(int(index))+'.model')
        model=load_model(filename)
        
        #predict the 12 values
        y_predict = model.predict(row.values.reshape(1,10,4))
        
        #assign each value to the right place
        idPolair=str(int(index))
        print(idPolair)
        results["NO2"][idPolair]["D0"] = np.full((24), y_predict[0, 1])
        results["NO2"][idPolair]["D1"] = np.full((24),y_predict[0, 5])
        results["NO2"][idPolair]["D2"] = np.full((24),y_predict[0, 9])
        results["O3"][idPolair]["D0"] = np.full((24),y_predict[0, 0])
        results["O3"][idPolair]["D1"] = np.full((24),y_predict[0, 4])
        results["O3"][idPolair]["D2"] = np.full((24),y_predict[0, 8])
        results["PM10"][idPolair]["D0"] = np.full((24),y_predict[0, 2])
        results["PM10"][idPolair]["D1"] = np.full((24),y_predict[0, 6])
        results["PM10"][idPolair]["D2"] = np.full((24),y_predict[0, 10])
        results["PM25"][idPolair]["D0"] = np.full((24),y_predict[0, 3])
        results["PM25"][idPolair]["D1"] = np.full((24),y_predict[0, 7])
        results["PM25"][idPolair]["D2"] = np.full((24),y_predict[0, 11])
        
    return results
        
        
        
        


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



def toPredcit(concentrations_day):
    O3=pd.DataFrame(concentrations_day['O3'].set_index(['idPolair','date'])['Valeur'])
    O3.columns=['O3']
    NO2=pd.DataFrame(concentrations_day['NO2'].set_index(['idPolair','date'])['Valeur'])
    NO2.columns=['NO2']
    PM10=pd.DataFrame(concentrations_day['PM10'].set_index(['idPolair','date'])['Valeur'])
    PM10.columns=['PM10']
    PM25=pd.DataFrame(concentrations_day['PM25'].set_index(['idPolair','date'])['Valeur'])
    PM25.columns=['PM25']
    X=pd.merge(O3,pd.merge(NO2,pd.merge(PM10,PM25,left_index=True,right_index=True,how='outer'),left_index=True,right_index=True,how='outer'),left_index=True,right_index=True,how='outer')
    X=X.reset_index()
    X['idPolair']=pd.to_numeric(X['idPolair'])
    X['date']=pd.to_datetime(X['date'])
    X=X.set_index(['idPolair','date']).sort_index()
    i=0


    means=X.reset_index().set_index('date').resample('D').mean()
    del means['idPolair']
    meanOfmeans=pd.DataFrame((np.array(means.mean().tolist()).reshape(1,4)),columns=['O3', 'NO2', 'PM10', 'PM25'])
    meanOfmeans=pd.concat([meanOfmeans,meanOfmeans,meanOfmeans,meanOfmeans,meanOfmeans,meanOfmeans,meanOfmeans,meanOfmeans,meanOfmeans,meanOfmeans],axis=1)
    newX=None
    groups=X.groupby(level=0)
    i=0
    for name,g in groups:
        gg=g.reset_index().set_index('date').resample('D').mean().reset_index().set_index(['idPolair','date']).fillna(method='ffill').fillna(method='bfill').fillna(means).fillna(0)
        shape=gg.shape[0]
        gg=pd.DataFrame(np.array(pd.concat([gg.shift(10),gg.shift(9),gg.shift(8),gg.shift(7),gg.shift(6),gg.shift(5),gg.shift(4),gg.shift(3),gg.shift(2),gg.shift(1)],axis=1).iloc[shape-1,:].tolist()).reshape(1,40),index=[name],columns=['O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25','O3', 'NO2', 'PM10', 'PM25'])
        meanOfmeans.index=[name]
        gg=gg.fillna(meanOfmeans)
        if(i==0):
            newX=gg
        else:
            newX=pd.concat([newX,gg])
        i=i+1
    return newX