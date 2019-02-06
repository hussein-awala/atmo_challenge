# -*- coding: utf-8 -*-
from utils import load_data,get_data_day
import numpy as np
import pickle
import datetime
import calendar
import time
import os

import utils
import pandas as pd
import logging
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
import utils
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle
from utils import load_data,get_data_day
import numpy as np
import pickle
import datetime
import calendar
import time
import os

import utils
import pandas as pd
import logging
#### Predict function for a given day

dir = os.path.dirname(__file__)

def predict(day, sites, chimeres_day, geops_day, meteo_day,
                                   concentrations_day):
    start=time.time()
    print(day)
    X=toPredcit(concentrations_day)
    #define a dictionary to return the result
    results = dict({})
    
    #add the 4 pol to the dict
    for pol in ["PM10", "PM25", "O3", "NO2"]:
        results[pol] = dict({})
        #add each idPolair to each pol in the dict
        sites=np.load('models/sites.npy')
        for idPolair in sites:
            results[pol][''+str(int(idPolair))] = dict({})
        
    #results[pol]['33374'] = dict({})
    
    #for each pol predict and add the result
    for index,row in X.iterrows():
        
        #remove old model from GPU to speed up the model
        
        #load the model of this site
        model = pickle.load(open(dir+'/forests/'+str(int(index))+'.model', 'rb'))
        
        #predict the 12 values
        y_predict = model.predict(np.array([row.values]))
        
        #assign each value to the right place
        idPolair=str(int(index))
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
    end=time.time()
    print(end-start)
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
        chimeres_day,geops_day,meteo_day,concentrations_day = get_data_day(day,data,max_days=max_days,year=year)  # you will get an extraction of the year datasets, limited to the past max_days for each day
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