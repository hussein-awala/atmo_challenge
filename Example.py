# coding: utf-8

import prediction
from utils import load_data,get_data_day

import logging
logging.basicConfig(level=logging.INFO)

###################################"
## Use functions separately if you want to test your predict function

# get data from files 
# CHANGE DIRNAME TO WHERE YOU STORE YOUR DATA
all_data=load_data(dirname="../Data/training") 

# returns a dictionnary {"sites":sites,"chimeres":chimeres,"geops":geops,"meteo":meteo,"concentrations":concentrations}
# see utils.load_data function for more details

# from all_data, extract only  allowed data for day=3 (! january 4 as python starts with 0)
data_day=get_data_day(3,all_data) 

## apply predict function for day=3, using 
prediction.predict(3,sites=all_data['sites'],chimeres_day=data_day[0],geops_day=data_day[1],meteo_day=data_day[2],concentrations_day=data_day[3])

##############################################"""

## OR run the run_predict function that will call your daily predict function, here on the first 50 days of year

prediction.run_predict(list_days=range(50),dirname = "../Data/training")

## And get score in a file score.txt
run(scoring)

