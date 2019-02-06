import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from glob import glob
from os.path import join
import pandas
import datetime
import numpy as np
import logging
import os.path



def load_training_data(dirname="../Data/training", f_geop="Geop_d02", f_meteo="meteoWRF"):
    logger = logging.getLogger(__name__)

    ## sites
    ## Dataframe with columns
    logger.info("Loading sites files")
    ro.r['load'](join(dirname, 'Description_Stations.RData'))
    sites = ro.r['Description_Stations']
    sites = pandas2ri.ri2py(sites)

    ## chimere
    ## Dict on Pollutants
    ## for each pollutant
    ## Dataframe with columns 'date', 'val', 'idPolair', 'param'
    logger.info("Loading Chimere files")
    chimeres = dict({})
    for pol in ["NO2", "O3", "PM10", "PM25"]:
        year=2012
        fname = join(dirname, f"CHIMERE/CHIMERE_{pol}_{year}.rds")
        data = loadFile(fname)
        chimeres[pol] = data
        for year in range(2013,2017):
            fname = join(dirname, f"CHIMERE/CHIMERE_{pol}_{year}.rds")
            data = loadFile(fname)
            chimeres[pol] = pandas.concat([chimeres[pol],data])

    ## Geop
    ## Dict on sites
    ## Dataframe with columns 'date', 'idPolair', 'geop_p_500hPa', 'geop_p_850hPa'
    logger.info("Loading Geop files")
    geops = dict({})
    for site in sites.idPolair:
        year=2012
        fname = join(dirname, f"WRF/Geop_02/Geop.{site.lstrip('0')}.{year}.d02.rds")
        if os.path.isfile(fname):
            data = loadFile(fname)
            geops[site] = data
        else:
            logger.info(f"Site {site} does not have a geop file")
        for year in range(2013,2017):
            fname = join(dirname, f"WRF/Geop_02/Geop.{site.lstrip('0')}.{year}.d02.rds")
            if os.path.isfile(fname):
                data = loadFile(fname)
                geops[site] = pandas.concat([geops[site],data])
            else:
                logger.info(f"Site {site} does not have a geop file")

    ## meteo
    ## Dataframe with columns
    year=2012
    logger.info("Loading Meteo files")
    fname = join(dirname, f"WRF/{f_meteo}_{year}.RData")
    meteo = loadFile(fname, "wrfData")
    for year in range(2013,2017):
        fname = join(dirname, f"WRF/{f_meteo}_{year}.RData")
        meteo = pandas.concat([meteo,loadFile(fname, "wrfData")])

    ## concentrations
    ## Dict on Pollutants
    ## for each pollutant, a dataframe with columns 'idPolair', 'Organisme', 'Station', 'Mesure', 'Date', 'Valeur'
    logger.info("Loading concentrations measure files")
    concentrations = dict({})
    for pol in ["NO2", "O3", "PM10", "PM25"]:
        year=2012
        fname = join(dirname, f"measures/Challenge_Data_{pol}_{year}.rds")
        data = loadFile(fname)
        concentrations[pol] = data
        for year in range(2013, 2017):
            fname = join(dirname, f"measures/Challenge_Data_{pol}_{year}.rds")
            data = loadFile(fname)
            concentrations[pol] = pandas.concat([concentrations[pol],data])
    return {"sites": sites, "chimeres": chimeres, "geops": geops, "meteo": meteo, "concentrations": concentrations}

def loadFile(fname,varname=None):
    """
    fname :  rdata or rds filename to be loaded
    varname : variable name inside rdata
    """
    if varname is not None:
        ro.r['load'](fname)
        full_data =ro.r[varname]
    else: #assume it is in rds format
        full_data = pandas2ri.ri2py(ro.r['readRDS'](fname))
    return full_data