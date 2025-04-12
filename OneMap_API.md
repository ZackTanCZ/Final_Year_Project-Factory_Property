# OneMap_API

## Import Python Library

```
import numpy as np
import pandas as pd
from datetime import datetime
import re
import json
import requests
from sklearn.metrics.pairwise import haversine_distances
from math import radians
```

## Define custom functions

```
def geoCode(projNameList):
    latList, longList = [],[]
    for i in range(len(projNameList)):
        property = projNameList[i]
        url_proj = "https://www.onemap.gov.sg/api/common/elastic/search?searchVal={}&returnGeom=Y&getAddrDetails=Y".format(property)
        proj_response = requests.request("GET", url_proj)
        projDict = json.loads(proj_response.text)
        latList.append([i, projDict["results"][0]["LATITUDE"]])
        longList.append([i, projDict["results"][0]["LONGITUDE"]])
    latDF = pd.DataFrame(latList, columns = ["index", "lat"])
    latDF.set_index('index', inplace = True)
    longDF = pd.DataFrame(longList, columns = ["index", "long"])
    longDF.set_index('index', inplace = True)
    projCoordDF = pd.DataFrame(projNameList, columns = ["Name"])
    # Merge with latDF
    projCoordDF = projCoordDF.merge(latDF, left_index =  True, right_index = True)
    # Merge with longDF
    projCoordDF = projCoordDF.merge(longDF, left_index =  True, right_index = True)

    return projCoordDF
```
```
def getStationName(df):
    # Categorise each station to 'MRT Station' or 'LRT Station'
    stnTypeDF = df["mrt_line_english"].drop_duplicates().reset_index(drop = True)
    stnTypeList = []
    for name in stnTypeDF:
        if "LRT" in name:
            stnTypeList.append("LRT Station")
        else:
            stnTypeList.append("MRT Station")
            
    stnTypeDF = stnTypeDF.to_frame()
    stnTypeDF["stn_type"] = stnTypeList
    
    # Merge back to df
    df = df.merge(stnTypeDF, left_on = "mrt_line_english" , right_on = "mrt_line_english")
    df["full_mrt_stn"] = df["mrt_station_english"] + " " + df["stn_type"]

    return df

```

## Prepare Project Name & Street Name and get the respective coordinates
Outcome: Properties with their coordinates
```
dbDF = pd.read_csv('PBIDB_Data.csv')

# Check for null project names in the dataframe
print("There're {} transactions with NaN Names".format(len(dbDF[dbDF["Project Name"].isnull() == True])))

# Subset the DataFrame to only include 'Project Name' & 'Street Name' 
projNameDF = dbDF[["Project Name","Street Name"]]

# Remove any duplicates and reset the DataFrame index
# This ensures that if the project name is not available, the street name is used for geoencoding
projNameDF = projNameDF.drop_duplicates(subset=["Project Name","Street Name"]).reset_index(drop = True)

# Replace NaN Project Name with their Street Name
projAllNamesDF = projNameDF["Project Name"].fillna(projNameDF["Street Name"])

# Convert to a List
projAllNameList = projAllNamesDF.to_list()

# Call 'geocode' function to geocode property
projCoordDF = geoCode(projAllNameList)

# Map back to original DF
projNameDF = projNameDF.merge(projCoordDF, left_index =  True, right_index = True)

# Drop 'Name' Column
projNameDF.drop(columns = ["Name"], inplace = True)

```

## Get coordinates of all MRT/LRT stations
Dataset available from LTA Datamall - [Train Station Codes and Chinese Names](https://datamall.lta.gov.sg/content/datamall/en/static-data.html#Public%20Transport)
Outcome: DataFrame with all MRT/LRT station and their coordinates
```
mrtDF = pd.read_csv('Train Station Codes.csv')
mrtDF = mrtDF[["mrt_station_english","mrt_line_english"]]

# Fully encode Station Name
mrtDF = getStationName(mrtDF)

# Convert the station names into a list
stationList = mrtDF["full_mrt_stn"].to_list()

# geoCode MRT Stations
mrtCoordDF = geoCode(stationList)

mrtDF = mrtDF.merge(mrtCoordDF, left_index = True, right_index = True)
mrtDF.drop(columns = ["Name"], inplace = True)

```

## Find the minimum distance to nearest MRT/LRT station

```
distList, minDistList = [], []
# Outer loop to loop through all the properties
for projindex in range(projNameDF.shape[0]):
    propCoord = projNameDF.loc[projindex,['lat','long']]
    propCoord_in_rad = [radians(_) for _ in propCoord]   
    for mrtindex in range(mrtDF.shape[0]):
        mrtCoord = mrtDF.loc[mrtindex,['lat','long']]
        mrtDist_in_rad = [radians(_) for _ in mrtCoord] 
        # Compute distance(in KM) between mrt and property 
        result = haversine_distances([mrtDist_in_rad, propCoord_in_rad])
        resultInKM = (result * 6371)
        distList.append(resultInKM[0,1].round(5))
    minDistList.append([projindex, min(distList)])
    distList.clear()

minDistDF = pd.DataFrame(minDistList, columns = ["index", "minDistToNearestStation"])
minDistDF.set_index('index', inplace = True)

projMRTDF = projNameDF.merge(minDistDF, left_index = True, right_index = True )
```
