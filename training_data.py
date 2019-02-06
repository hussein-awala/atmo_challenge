import utils
import pandas as pd
import numpy as np
import logging
import datetime

def data_station_year(dirname='Data/training/',year=2016):
    logger = logging.getLogger(__name__)
    logger.info("load all data")
    x=utils.load_data(dirname=dirname,year=year)

    print("     start loading chimere")
    t1 = x['chimeres']['NO2']
    t1.drop(columns='param', inplace=True)
    t1['date'] = pd.to_datetime(t1['date'])
    t1.set_index(['idPolair', 'date'], inplace=True)
    t1.columns = ['NO2_chimere']
    t2 = x['chimeres']['O3']
    t2.drop(columns='param', inplace=True)
    t2['date'] = pd.to_datetime(t2['date'])
    t2.set_index(['idPolair', 'date'], inplace=True)
    t2.columns = ['O3_chimere']
    t3 = x['chimeres']['PM10']
    t3.drop(columns='param', inplace=True)
    t3['date'] = pd.to_datetime(t3['date'])
    t3.set_index(['idPolair', 'date'], inplace=True)
    t3.columns = ['PM10_chimere']
    t4 = x['chimeres']['PM25']
    t4.drop(columns='param', inplace=True)
    t4['date'] = pd.to_datetime(t4['date'])
    t4.set_index(['idPolair', 'date'], inplace=True)
    t4.columns = ['PM25_chimere']
    t5 = pd.merge(t1,pd.merge(t2, pd.merge(t3, t4, left_index=True, right_index=True,how='outer'), left_index=True, right_index=True,how='outer'),left_index=True, right_index=True,how='outer')
    t5.sort_index(inplace=True)
    #t5.interpolate(inplace=True)


    #print("     remove nan")
    #t5 = t5.groupby(t5.columns, axis=1).transform(lambda x: x.fillna(x.mean()))
    #t5.fillna(t5.mean(), inplace=True)
    #print("     set zeros")
    #t5.fillna(0, inplace=True)

    print("     start loading meteo")
    meteo = x['meteo']

    meteo['date'] = pd.to_datetime(meteo['date'])

    meteo.set_index(['idPolair', 'date'], inplace=True)
    meteo.sort_index(inplace=True)
    #meteo.interpolate(inplace=True)

    #print("     remove nan")
    #meteo = meteo.groupby(meteo.columns, axis=1).transform(lambda x: x.fillna(x.mean()))
    #meteo.fillna(meteo.mean(), inplace=True)

    #print("     set zeros")
    #meteo.fillna(0, inplace=True)

    allData = pd.merge(meteo, t5, left_index=True, right_index=True,how='outer')
    print("     start loading geops")
    geops = x['geops']
    keys = np.fromiter(geops.keys(), dtype=float)

    z = pd.concat(geops.values(), keys=keys)
    z['date'] = pd.to_datetime(z['date'])
    z.set_index(['idPolair', 'date'], inplace=True)
    z.sort_index(inplace=True)
    #z.interpolate(inplace=True)

    #print("     remove nan")
    #z = z.groupby(z.columns, axis=1).transform(lambda x: x.fillna(x.mean()))
    #z.fillna(z.mean(), inplace=True)

    #print("     set zeros")
    #z.fillna(0, inplace=True)



    allData = pd.merge(allData, z, left_index=True, right_index=True,how='outer')

    print("     start loading target values")
    t6 = x['concentrations']['NO2']
    t6.drop(columns=['Organisme','Station','Mesure'], inplace=True)
    t6['date'] = pd.to_datetime(t6['date'])
    t6['idPolair']=pd.to_numeric( t6['idPolair'])
    t6.set_index(['idPolair', 'date'], inplace=True)
    t6.columns = ['NO2']
    t7 = x['concentrations']['O3']
    t7.drop(columns=['Organisme','Station','Mesure'], inplace=True)
    t7['date'] = pd.to_datetime(t7['date'])
    t7['idPolair'] = pd.to_numeric(t7['idPolair'])
    t7.set_index(['idPolair', 'date'], inplace=True)
    t7.columns = ['O3']
    t8 = x['concentrations']['PM10']
    t8.drop(columns=['Organisme','Station','Mesure'], inplace=True)
    t8['date'] = pd.to_datetime(t8['date'])
    t8['idPolair'] = pd.to_numeric(t8['idPolair'])
    t8.set_index(['idPolair', 'date'], inplace=True)
    t8.columns = ['PM10']
    t9 = x['concentrations']['PM25']
    t9.drop(columns=['Organisme','Station','Mesure'], inplace=True)
    t9['date'] = pd.to_datetime(t9['date'])
    t9['idPolair'] = pd.to_numeric(t9['idPolair'])
    t9.set_index(['idPolair', 'date'], inplace=True)
    t9.columns = ['PM25']
    t10 = pd.merge(t6,pd.merge(t7, pd.merge(t8, t9, left_index=True, right_index=True,how='outer'), left_index=True, right_index=True,how='outer'),
                  left_index=True, right_index=True,how='outer')
    t10.sort_index(inplace=True)
    #t10.interpolate(inplace=True)

    #print("     remove nan")
    #t10 = t10.groupby(t10.columns, axis=1).transform(lambda x: x.fillna(x.mean()))
    #t10.fillna(t10.mean(), inplace=True)

    #print("     set zeros")
    #t10.fillna(0, inplace=True)

    allData=pd.merge(allData, t10, left_index=True, right_index=True,how='outer')

    allData=allData.reset_index()
    #allData.set_index("date")\
     #   .drop(str(year+1) + "-01-01 00:00:00")
    allData.set_index(['idPolair', 'date'],inplace=True)
    return allData

def data_station(dirname='Data/training/'):
    logger = logging.getLogger(__name__)
    sDate=datetime.datetime.now()
    print("start loading 2012")
    data_2012 = data_station_year(dirname,  2012)
    data_2012.to_csv('data_'+str(2012) + '.csv')
    eDate=datetime.datetime.now()
    print("end in {}".format((eDate-sDate)))
    sDate = datetime.datetime.now()
    print("start loading 2013")
    data_2013 = data_station_year(dirname,  2013)
    data_2013.to_csv('data_'+str(2013) + '.csv')
    eDate = datetime.datetime.now()
    print("end in {}".format((eDate - sDate)))
    sDate = datetime.datetime.now()
    print("start loading 2014")
    data_2014 = data_station_year(dirname,  2014)
    data_2014.to_csv('data_'+str(2014) + '.csv')
    eDate = datetime.datetime.now()
    print("end in {}".format((eDate - sDate)))
    sDate = datetime.datetime.now()
    print("start loading 2015")
    data_2015 = data_station_year(dirname,  2015)
    data_2015.to_csv('data_'+str(2015) + '.csv')
    eDate = datetime.datetime.now()
    print("end in {}".format((eDate - sDate)))
    sDate = datetime.datetime.now()
    print("start loading 2016")
    data_2016 = data_station_year(dirname,  2016)
    data_2016.to_csv('data_'+str(2016) + '.csv')
    eDate = datetime.datetime.now()
    print("end in {}".format((eDate - sDate)))
    print("appending")
    allData=pd.concat([data_2012,data_2013,data_2014,data_2015,data_2016])

#     print("interpolate all data")
#     allData.interpolate(inplace=True)

#     if allData.isnull().any().any():
#         print("nan->mean")
#         allData.fillna(allData.mean(), inplace=True)
#     if allData.isnull().any().any():
#         print("nan->0")
#         allData.fillna(0, inplace=True)

    allData.to_csv( 'data_with_nan.csv')

def join_all_sites(path):
    print('loading data')
    data=pd.read_csv(path)
    data.set_index(['idPolair','date'],inplace=True)
    g=data.groupby('idPolair')
    del data
    sites_info=pd.read_csv('sites.csv')
    sites_info.set_index('idPolair')
    groups=g.groups
    sites=groups.keys()
    resultX=None
    resultY=None
    for site in sites:
        print(site)
        group=g.get_group(site)
        group.reset_index(inplace=True)
        del group['idPolair']
        group['date'] = pd.to_datetime(group['date'])
        group.set_index('date', inplace=True)
        group = group.resample('D').mean()
        X=group.iloc[:,:group.shape[1]-4]
        y=group.iloc[:,group.shape[1]-4:]
        to_window(X,10,-1)
        to_window(y,3,1)
        data=pd.concat([X,y],axis=1, join='inner')
        data.dropna(inplace=True)
        X = data.iloc[:, :data.shape[1] - 12]
        y = data.iloc[:, data.shape[1] - 12:]
        X.reset_index(inplace=True)
        del X['date']

        y.reset_index(inplace=True)
        del y['date']
        if resultX is None:
            resultX=X
            resultY=y
        else:
            resultX=pd.concat([resultX,X])
            resultY=pd.concat([resultY,y])
        del group
    return resultX,resultY
def to_window(data,t,sig=-1):
    cols=data.columns
    i=1
    while(i<abs(t)):
        for col in cols:
            data[col+'_t+'+str(i)]=data[col].shift(sig*i)
        i=i+1
    return data

def exec(path=""):
    data_station(path)
    join_all_sites("data.csv").to_csv('final_data.csv')