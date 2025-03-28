# Data Wrangling for FYP Dataset 

## Processing Industrial Property Data

### Importing Python Libraries
```
import pandas as pd
pd.options.mode.chained_assignment = None # Disable warning from pandas library
# pd.options.mode.chained_assignment = 'warn' # enable warning from pandas library
from datetime import datetime
import re
```

### Defining Custom Functions
```
#Custom function to remove metadata
def removeMeta(dfName):
    firstColName = dfName.columns[0]
    startIndex, endIndex = 0,0
    startIndex = dfName.index[dfName[firstColName]=='Data Series'].tolist()[0]
    endIndex = dfName.index[dfName[firstColName]=='Footnotes:'].tolist()[0]-1
    dfName = dfName.iloc[startIndex:endIndex] #remove metadata
    return dfName

def checkNull(dfName):
    null_counts = dfName.isnull().sum()
    null_counts = pd.Series(null_counts, name = 'Null Count').to_frame()
    print("Rows: {}, Cols: {}".format(dfName.shape[0],dfName.shape[1]))
    print('')
    print(dfName.info())
    print('')
    display(null_counts[null_counts['Null Count'] > 0])
```

### Read CSV files
```
factoryDF = pd.read_csv('Data/Factory_Data_By_District.csv', na_values=['NA', 'N/A','na','-']) 

checkNull(factoryDF)
display(factoryDF.tail(5))

```

### Filter Dataset to specific use case
```
# Using Boolean masking to remove 'Freehold' and non-'Strata' first
tenureMask = factoryDF['Tenure'] !='Freehold'
areaTypeMask = factoryDF['Type Of Area'] == 'Strata'
leaseholdFactoryDF = factoryDF[tenureMask & areaTypeMask].reset_index(drop=True)

# Using Boolean masking to remove 'Tenure' containing '999 yrs'
longLeasePattern = r'999 yrs'
longLeaseMask = leaseholdFactoryDF['Tenure'].str.contains(longLeasePattern, case=False, na=False)
leaseholdFactoryDF = leaseholdFactoryDF[~longLeaseMask].reset_index(drop=True)

checkNull(leaseholdFactoryDF)

#Using Regex and Boolean Masking
leaseholdFactoryDF = leaseholdFactoryDF.dropna(subset=['Tenure', 'Floor Level']) 
noDatePattern = r'yrs from \d{2}\/\d{2}\/\d{4}'
noDateMask = leaseholdFactoryDF['Tenure'].str.contains(noDatePattern, case=False, na=False)
leaseholdFactoryDF = leaseholdFactoryDF[noDateMask]
leaseholdFactoryDF = leaseholdFactoryDF.reset_index(drop=True) #reset the index, just in case.

checkNull(leaseholdFactoryDF)

```

### Adding new columns - quarter, month and year
```
quarterList,monthList ,yearList = [], [], []

for i in leaseholdFactoryDF['Contract Date']:
    month = int(i.split("/")[1])
    year = int(i.split("/")[2])
    monthinquarter = str((month-1)//3+1)
    quarter = str(year) + " " + monthinquarter +"Q"  
    quarterList.append(quarter)
    monthList.append(month)
    yearList.append(year)

#Convert list into pd Series    
quarterSeries = pd.Series(quarterList, name='Quarter')
monthSeries = pd.Series(monthList, name='Month')
yearSeries = pd.Series(yearList, name='Year')

# Add quarterSeries, monthSeries & yearSeries as new columns into leaseholdFactoryDF
leaseholdFactoryDF['Quarter'] = quarterSeries
leaseholdFactoryDF['Month'] = monthSeries
leaseholdFactoryDF['Year'] = yearSeries

# print(leaseholdFactoryDF.shape)
checkNull(leaseholdFactoryDF)
display(leaseholdFactoryDF.tail(3))
```

### Adding new columns - Remaining Tenure
```
endTenureList = []
for i in leaseholdFactoryDF['Tenure']:
    tenureYears = int(i.split(" ")[0])

    if re.search(r'/', i.split(" ")[-1]): # check for 'ETC' in the 'Tenure' Column
        startTenure = int(i.split(" ")[-1].split("/")[-1])
    else:
        startTenure = int(i.split(" ")[-2].split("/")[-1])    
    endTenure = tenureYears + startTenure
    endTenureList.append(endTenure)

endTenureSeries = pd.Series(endTenureList, name='endTenure')
leaseholdFactoryDF['Remaining Tenure'] = endTenureSeries - leaseholdFactoryDF['Year']
leaseholdFactoryDF.reset_index(drop=True,inplace=True)

checkNull(leaseholdFactoryDF)
display(type(leaseholdFactoryDF['Month'][0]))

display(leaseholdFactoryDF.head(5))
```

### Processing Population Data
```
sgPOPDF = pd.read_csv('Data/2024/population.csv') 
headerStart, headerEnd = 0,0

sgPOPDF = removeMeta(sgPOPDF)
sgPOPDF = sgPOPDF.set_index('Unnamed: 0')

headerStart = int(sgPOPDF.loc['Data Series'][0])
headerEnd = int(sgPOPDF.loc['Data Series'][-1])

sgPOPDF.columns = pd.Series(range(headerStart, headerEnd-1, -1))

sgPOPDF = sgPOPDF.iloc[1,0:] #is now a series
sgPOPDF = sgPOPDF.reset_index()
sgPOPDF = sgPOPDF.rename(columns={'index': 'Year'})

sgPOPDF = sgPOPDF.astype({'Year': 'int', 
                          'Total Population (Number)': 'float64'})

display(type(sgPOPDF['Total Population (Number)'][0]))
display(sgPOPDF)
```

### Processing SORA Data
```
soraDF = pd.read_csv('Data/2024/SORA.csv',na_values=['NA', 'N/A','na'])
soraYear, soraMonth = [],[]
soraDF = removeMeta(soraDF)
soraDF = soraDF.iloc[[0,-1],:].T.reset_index(drop=True)
soraDF.columns = soraDF.iloc[0]
soraDF = soraDF.iloc[1:].reset_index(drop=True)
soraDF = soraDF.rename(columns={'Data Series': 'Quarter', 'Singapore Overnight Rate Average':'SORA'})


#convert the "Quarter" column to date object so we can translate the month to number. 
#so we can map back to transactional dataset
soraDF['Quarter'] = pd.to_datetime(soraDF['Quarter'])

for i in soraDF['Quarter']:
    soraYear.append(i.year)
    soraMonth.append(i.month)
    
soraYearSeries = pd.Series(soraYear)
soraMonthSeries = pd.Series(soraMonth)
soraDF['Year'] = soraYearSeries 
soraDF['Month'] = soraMonthSeries

#Remove rows with na SORA Values and reset index
soraDF = soraDF.dropna(subset=['SORA'])
soraDF = soraDF.reset_index(drop=True)

display(type(soraDF['Month'][0]))
display(type(soraDF['Year'][0]))
display(soraDF.head(3))
```

### Processing GDP Data
```
yearlyNominalGDPDF = pd.read_csv('Data/2024/Annual_GDP.csv')
deflatorDF = pd.read_csv('Data/2024/Annual_GDP_Deflator.csv')
yearlyNominalGDPDF = removeMeta(yearlyNominalGDPDF)
deflatorDF = removeMeta(deflatorDF)

yearlyNominalGDPDF = yearlyNominalGDPDF.iloc[[0,1],:].T.reset_index(drop=True)
yearlyNominalGDPDF.columns = yearlyNominalGDPDF.iloc[0]
yearlyNominalGDPDF = yearlyNominalGDPDF.iloc[1:].reset_index(drop=True)
yearlyNominalGDPDF = yearlyNominalGDPDF.rename(columns={'Data Series': 'Year', 
                              'GDP At Current Market Prices':'Nominal Annual GDP (in $Sm)'})


deflatorDF = deflatorDF.iloc[[0,1],:].T.reset_index(drop=True)
deflatorDF.columns = deflatorDF.iloc[0]
deflatorDF = deflatorDF.iloc[1:].reset_index(drop=True)
deflatorDF = deflatorDF.rename(columns={'Data Series': 'Year', 'GDP':'GDP Deflator'})

gdpDF = deflatorDF.merge(yearlyNominalGDPDF, on='Year')

gdpDF = gdpDF.astype({'Year': 'int', 
                      'GDP Deflator': 'float64', 
                      'Nominal Annual GDP (in $Sm)': 'float64'})

#Change Deflator to decimal point
gdpDF['GDP Deflator'] = gdpDF['GDP Deflator']/100

# Calculate Real GDP
gdpDF['Real GDP (in $Sm)'] = (gdpDF['Nominal Annual GDP (in $Sm)'] / gdpDF['GDP Deflator'])
gdpDF['Real GDP (in $Sm)'] = gdpDF['Real GDP (in $Sm)'].round(2)

display(gdpDF)
```

### Adding Land Supply Data
```
landSupplyDF = pd.read_csv('Data/2024/Supply Of Commercial And Industrial Properties In The Pipeline By Development Status.csv', na_values=['NA', 'N/A','na'])
landSupplyDF = removeMeta(landSupplyDF)
landSupplyFirstColName = landSupplyDF.columns[0]

landSupplyQuarterIndex = landSupplyDF.index[landSupplyDF[landSupplyFirstColName]=='Data Series'].tolist()[0]
warehouseIndex = landSupplyDF.index[landSupplyDF[landSupplyFirstColName]=='Total Warehouse Space'].tolist()[0]
singleFactoryIndex = landSupplyDF.index[landSupplyDF[landSupplyFirstColName]=='Total Single-User Factory Space'].tolist()[0]
multiFactoryIndex = landSupplyDF.index[landSupplyDF[landSupplyFirstColName]=='Total Multiple-User Factory Space'].tolist()[0]
bizParkIndex = landSupplyDF.index[landSupplyDF[landSupplyFirstColName]=='Total Business Park Space'].tolist()[0]

landSupplyDF = landSupplyDF.loc[[landSupplyQuarterIndex, warehouseIndex, singleFactoryIndex, multiFactoryIndex,bizParkIndex],:].T.reset_index(drop=True)
landSupplyDF.columns = landSupplyDF.iloc[0]
landSupplyDF = landSupplyDF.iloc[1:].reset_index(drop=True)
landSupplyDF = landSupplyDF.rename(columns={'Data Series': 'Quarter'})

#Remove rows with na Values and reset index
landSupplyDF = landSupplyDF.dropna(subset=['Total Warehouse Space','Total Single-User Factory Space',
                                           'Total Multiple-User Factory Space','Total Business Park Space'])
landSupplyDF = landSupplyDF.reset_index(drop=True)

landSupplyDF = landSupplyDF.astype({'Quarter': 'str', 
                                    'Total Warehouse Space': 'float64', 
                                    'Total Single-User Factory Space': 'float64',
                                    'Total Multiple-User Factory Space': 'float64', 
                                    'Total Business Park Space': 'float64'})

#remove ghost spaces so that we can merge the DFs
landSupplyDF['Quarter'] = landSupplyDF['Quarter'].apply(lambda x: x.strip())

landSupplyDF['Total Warehouse Space'] = landSupplyDF['Total Warehouse Space'] * 1000
landSupplyDF['Total Single-User Factory Space'] = landSupplyDF['Total Single-User Factory Space'] * 1000
landSupplyDF['Total Multiple-User Factory Space'] = landSupplyDF['Total Multiple-User Factory Space'] * 1000
landSupplyDF['Total Business Park Space'] = landSupplyDF['Total Business Park Space'] * 1000

landSupplyDF['Total Supply of Industrial Space (m2)'] = landSupplyDF['Total Warehouse Space'] + landSupplyDF['Total Single-User Factory Space'] + landSupplyDF['Total Multiple-User Factory Space'] + landSupplyDF['Total Business Park Space'] 

display(landSupplyDF.head(3))
```

### Adding Foreign Exchange Rate (Forex) Data
```
forexDF = pd.read_csv('Data/2024/Exchange Rates (Average For The Year).csv', na_values=['NA', 'N/A','na'])
forexYear, forexMonth = [],[]
forexDF = removeMeta(forexDF)
forexFirstColName = forexDF.columns[0]

forexYearIndex = forexDF.index[forexDF[forexFirstColName]=='Data Series'].tolist()[0]
usdIndex = forexDF.index[forexDF[forexFirstColName]=='US Dollar (Singapore Dollar Per US Dollar)'].tolist()[0]

forexDF = forexDF.loc[[forexYearIndex,usdIndex],:].T.reset_index(drop=True)
forexDF.columns = forexDF.iloc[0]
forexDF = forexDF.iloc[1:].reset_index(drop=True)
forexDF = forexDF.rename(columns={'Data Series': 'Year'})

forexDF = forexDF.astype({'Year': 'int',
                          'US Dollar (Singapore Dollar Per US Dollar)': 'float64'})

display(forexDF)
```

### Adding Tender Price Index (TPI) Data
```
bcaTPIDF = pd.DataFrame({'Year':[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
                         'TPI' :[119.1, 101.3, 100.0, 99.7, 99.8, 104.6, 106.8, 104.0, 98.0, 96.7, 98.6, 99.9, 102.8,
                                 117.1, 130.7, 136.0]})

bcaTPIDF['TPI % Change'] = bcaTPIDF['TPI'] - 100

bcaTPIDF = bcaTPIDF.astype({'Year': 'int', 
                            'TPI': 'float64', 
                            'TPI % Change': 'float64'})

display(bcaTPIDF)
```

### Merge all the DataFrames into one DataFrame
```
leaseholdFactoryDF = leaseholdFactoryDF.drop(columns=['Property Type','Type Of Area'])
finalDF1 = leaseholdFactoryDF.merge(sgPOPDF,how='inner', on='Year')
finalDF2 = finalDF1.merge(soraDF,how='inner', on=['Month','Year'])
finalDF3 = finalDF2.merge(gdpDF,how='inner', on='Year')
finalDF4 = finalDF3.merge(landSupplyDF,how='inner', on='Quarter')
finalDF5 = finalDF4.merge(forexDF,how='inner', on=['Year'])
finalDF = finalDF5.merge(bcaTPIDF,how='inner', on=['Year'])
```

### Adding Lagged psf
```
def addLaggedpsf(df):
    listofProject = df['Project Name'].unique()
    print('Number of Projects: {}'.format(len(listofProject)))
    
    fullDF = pd.DataFrame()
    for i in listofProject:
        projFullDF = df[df['Project Name'] == i].reset_index(drop = True) #Creates a df for each project
        projFullDF['Contract Date'] = pd.to_datetime(projFullDF['Contract Date'], dayfirst=True) # convert 'Contract Date' to datetime
        projFullDF.sort_values(by = 'Contract Date', ascending = True, inplace=True) # sort Date in ascending order
        projFullDF.reset_index(drop=True, inplace=True)
        projSubDF = projFullDF[['Contract Date','Unit Price ($ psf)']] # subset the full project DF, join back later.
        projSubDF.set_index('Contract Date', inplace = True) #set the contractDate as df index
        projSubDF = projSubDF.rolling(3, closed= "left", min_periods= 1).max() # get the lagged psf and choose the max
        projSubDF.fillna(method='bfill', inplace=True) # replace NA values with the next row's psf
        projSubDF.rename(columns={'Unit Price ($ psf)':'lagged psf'}, inplace=True)
        projSubDF.reset_index(inplace=True)
        projFullDF = projFullDF.merge(projSubDF,left_index = True, right_index=True, sort=False)
        projFullDF.rename(columns={'Contract Date_x':'Contract Date'}, inplace=True)
        fullDF = pd.concat([fullDF, projFullDF], sort=False)
    return fullDF

# Apply the custom function to the dataset, finalDF
laggedDF = addLaggedpsf(finalDF)
laggedDF.dropna(subset=['lagged psf'], inplace=True)
display(laggedDF.shape)
final_laggedDF = laggedDF.drop(columns=['Contract Date_y'])
```

### Adding Rental Index
```
# Custom function to add Regional Rental Index % change
def mergeRentalIndex(list):
    allRentalIndexDF = pd.DataFrame()
    for i in list:
        fileName = 'Data/Rental Index of Multiple-User Factory (' + i + ').csv'
        rentalIndexDF = pd.read_csv(fileName,na_values=['NA', 'N/A','na','-'])
        rentalIndexDF.columns = ['Quarter', 'Rental Index']
        rentalIndexDF['Region'] = i
        allRentalIndexDF = pd.concat([allRentalIndexDF, rentalIndexDF])
    allRentalIndexDF.dropna(inplace=True)
    allRentalIndexDF.reset_index(drop=True, inplace=True)
    allRentalIndexDF = allRentalIndexDF[['Quarter', 'Region', 'Rental Index']]
    allRentalIndexDF['Quarter'] = allRentalIndexDF['Quarter'].apply(lambda x: x[:4] + " " + x[5] + x[4])
    return allRentalIndexDF

regionList = ['Central Region','East Region','North Region','North-East Region','West Region']
allRentalIndexDF = mergeRentalIndex(regionList)
allRentalIndexDF['Rental Index % change'] = allRentalIndexDF['Rental Index'] - 100

final_RICDF = final_laggedDF.merge(allRentalIndexDF ,how='inner', on=['Quarter','Region'])
final_RICDF.reset_index(drop = True, inplace=True)
```

### Adding Geographic Data
```
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import requests
import json

# https://github.com/xkjyeah/MRT-and-LRT-Stations/blob/master/mrt_lrt.csv
mrtDF = pd.read_csv('Data\mrt_lrt.csv') #Read MRT Coords

# Custom Function to get the lat and long coordinates of each building, identified by the project name
# Using API from https://www.onemap.gov.sg/
def geocodeProperty(list):
    nameList,latList, longList = [], [], []
    for i in list:
        proj_name= i[0]
        nameList.append(proj_name)
        url_proj = "https://www.onemap.gov.sg/api/common/elastic/search?searchVal={}&returnGeom=Y&getAddrDetails=Y".format(proj_name)
        proj_response = requests.request("GET", url_proj)
        projDict = json.loads(proj_response.text)
        
        street_name = i[1]
        url_street = "https://www.onemap.gov.sg/api/common/elastic/search?searchVal={}&returnGeom=Y&getAddrDetails=Y".format(street_name)
        street_response = requests.request("GET", url_street)
        streetDict = json.loads(street_response.text)
               
        if projDict['found'] == 0:
            latList.append(streetDict['results'][0]['LATITUDE'])
            longList.append(streetDict['results'][0]['LONGITUDE'])
        else:
            latList.append(projDict['results'][0]['LATITUDE'])
            longList.append(projDict['results'][0]['LONGITUDE'])
    geoCodeDF = pd.DataFrame({'Project Name':nameList,
                              'latitude':pd.Series(latList,name = 'latitude'),
                              'longitude':pd.Series(longList,name = 'longitude')})    
    return geoCodeDF

# Get the unique building names from the dataset and geocode them
nameDF = final_RICDF[['Project Name','Street Name']].drop_duplicates('Project Name').reset_index(drop=True)
nameList = nameDF.values.tolist() # a Total of 138 unique building names, caution of "Woodlands east industrial estate"
display(len(nameList))

# Invoke the custome function to geocode the properties
propertyGeoCodeDF = geocodeProperty(nameList)
propertyGeoCodeDF = propertyGeoCodeDF.astype({'latitude':'float64',
                                              'longitude':'float64'})

display(propertyGeoCodeDF)
display(propertyGeoCodeDF.info())

# Prepare the MRT dataset from csv file
mrtDF = mrtDF[['Name','Latitude','Longitude']]
mrtDF.rename(columns={'Name':'mrt_station',
                      'Latitude':'latitude',
                      'Longitude':'longitude'}, inplace=True)

# Use a loop structure to compute the nearest mrt distance
distList, minDistList = [], []
for i in range(propertyGeoCodeDF.shape[0]): # Outer loop to loop through all the buildings, from propertyGeoCodeDF
    propCoord = propertyGeoCodeDF.loc[i,['latitude','longitude']]
    propCoord_in_rad = [radians(_) for _ in propCoord]
    for i in range(mrtDF.shape[0]): # Inner loop to loop through all the mrt exits
        mrtCoord = mrtDF.loc[i,['latitude','longitude']]
        mrtDist_in_rad = [radians(_) for _ in mrtCoord] #Convert to radian according to docs
        result = haversine_distances([mrtDist_in_rad, propCoord_in_rad])
        resultInKM = (result * 6371)
        distList.append(resultInKM[0,1].round(5))
    minDistList.append(min(distList))
    distList = [] # Clear the list after each iteration
display(len(minDistList))

minDistToMRT = pd.Series(minDistList, name = 'minDistToMRT')
propertyGeoCodeDF = pd.concat([propertyGeoCodeDF, minDistToMRT], axis = 1)
propertyGeoCodeDF['minDistToMRT'] = (propertyGeoCodeDF['minDistToMRT'] * 1000)

Final_Dataset = final_RICDF.merge(propertyGeoCodeDF, how = 'inner', left_on='Project Name', 
                             right_on='Project Name', sort=False)
```
