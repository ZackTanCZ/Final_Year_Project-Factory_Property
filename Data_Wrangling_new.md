# Data Cleaning and Data Transformation

## Python Libraries

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from datetime import datetime
import re
import json
from urllib.request import Request,urlopen
from urllib.parse import quote
from functools import reduce
```

## Cleaning and Transformation of JTC Data

### Read the CSV file

```
jtcDF = pd.read_csv('FYP_JTC.csv')
```
### Filter and retain properties with 'Multiple User Factory', 'Strata' & 'Leasehold'

```
tenureMask = jtcDF['Tenure'] !='Freehold'
areaTypeMask = jtcDF['Type Of Area'] == 'Strata'
propertyTypeMask = jtcDF['Property Type'] == 'Multiple-User Factory'
lhfDF = jtcDF[tenureMask & areaTypeMask & propertyTypeMask].reset_index(drop=True)
```

```
longLeasePattern = r'999 yrs'
longLeaseMask = lhfDF['Tenure'].str.contains(longLeasePattern, case=False, na=False)
lhfDF = lhfDF[~longLeaseMask].reset_index(drop=True)


```

```
quarterList,monthList ,yearList = [], [], []

for i in lhfDF['Contract Date']:
    month = int(i.split("/")[1])
    year = int(i.split("/")[2])
    # Find which quarter does this month belong to
    monthinquarter = str((month-1)//3+1)
    quarter = str(year) + " " + monthinquarter + "Q"   
    quarterList.append(quarter)
    monthList.append(month)
    yearList.append(year)

#Convert list into a Pandas Series    
quarterSeries = pd.Series(quarterList, name='Quarter')
monthSeries = pd.Series(monthList, name='Month')
yearSeries = pd.Series(yearList, name='Year')

# Add quarterSeries, monthSeries & yearSeries as new columns into lhfDF
lhfDF['Quarter'] = quarterSeries
lhfDF['Month'] = monthSeries
lhfDF['Year'] = yearSeries

```

```
endTenureList = []
for i in lhfDF['Tenure']:
    tenureYears = int(i.split(" ")[0])

    if re.search(r'/', i.split(" ")[-1]): # check for 'ETC' in the 'Tenure' Column
        startTenure = int(i.split(" ")[-1].split("/")[-1])
    else:
        startTenure = int(i.split(" ")[-2].split("/")[-1])    
    endTenure = tenureYears + startTenure
    endTenureList.append(endTenure)

endTenureSeries = pd.Series(endTenureList, name='endTenure')
lhfDF['Remaining Tenure'] = endTenureSeries - lhfDF['Year']
# Remove 'ETC' from the 'Tenure' Column
lhfDF['Tenure'] =  lhfDF['Tenure'].str.replace('ETC', '', regex=False)
lhfDF.reset_index(drop=True,inplace=True)

```

```
lhfDF["YearMonthKey"] = (lhfDF["Year"]*100) + lhfDF["Month"]
```

## Compile, clean and transform Economic Data
Here, we use the API provided by Singstat to retrieve the necessary data.

### Get the ResourceID for each variable from Singstat API 

```
# Get the ResourceID for each variable from Singstats
varDict = {"SORA":{"title":"Current Banks Interest Rates (End Of Period)"},
           "pop":{"title":"Total Population By Broad Age Group And Sex, At End June, Annual"},
           "gdp":{"title":"Gross Domestic Product At Current Prices, By Industry (SSIC 2020), Annual"},
           "forex":{"title":"Exchange Rates (Average For Period), Monthly"},
           "land_supply":{"title":"Supply Of Commercial And Industrial Properties In The Pipeline By Development Status, (Private And Public Sectors)"}
          }

for key,value in varDict.items():
    # Replace blank spaces with %20
    title = value["title"].replace(" ","%20")
    header = {'User-Agent': 'Mozilla/5.0', "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8"}
    url = "https://tablebuilder.singstat.gov.sg/api/table/resourceid?keyword=" + title + "&searchOption=title"
    request = Request(url, headers = header)
    # Returns as a JSON object
    info = json.loads(urlopen(request).read())
    id = info["Data"]["records"][0]["id"]
    varDict[key]["id"] = id
```

### Adding 'Land_Supply' Variable

```
resourceID,seriesNo = varDict["land_supply"]["id"], 5
header = {'User-Agent': 'Mozilla/5.0', "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8"}
url = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/{}?seriesNoORrowNo={}".format(resourceID,seriesNo)
request = Request(url,headers = header)
data = json.loads(urlopen(request).read())
listOfland = data["Data"]["row"][0]["columns"]
# in '000 msq
landDF = pd.DataFrame(listOfland)

# Renaming columns
landDF.rename(columns={'key': 'Quarter', 'value': 'Land_Supply'}, inplace=True)

# Casting variables
landDF = landDF.astype({'Quarter':'str', 'Land_Supply':'int64'})
```

### Adding 'Population' variable

```
resourceID,seriesNo = varDict["pop"]["id"], 1
header = {'User-Agent': 'Mozilla/5.0', "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8"}
url = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/{}?seriesNoORrowNo={}".format(resourceID,seriesNo)
request = Request(url,headers = header)
data = json.loads(urlopen(request).read())
listOfpop = data["Data"]["row"][0]["columns"]
popDF = pd.DataFrame(listOfpop)

# Renaming columns
popDF.rename(columns={'key': 'Year', 'value': 'Pop'}, inplace=True)

# Casting variables
popDF = popDF.astype({'Year':'int64', 'Pop':'int64'})
```

### Adding '3M-SORA' variable

```
resourceID, seriesNo = varDict["SORA"]["id"], 23
header = {'User-Agent': 'Mozilla/5.0', "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8"}
url = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/{}?seriesNoORrowNo={}".format(resourceID, seriesNo)
request = Request(url,headers = header)
data = json.loads(urlopen(request).read())
listOfsora = data["Data"]["row"][0]["columns"]
soraDF = pd.DataFrame(listOfsora)

# Renaming columns
soraDF.rename(columns={'key': 'date', 'value': '3M-SORA'}, inplace=True)

# Split 'date' into 'year' & 'mth'
soraDF[["Year", "mth"]] = soraDF["date"].str.split(" ", expand = True)

# convert 'mth' into numerical month
dt_series = pd.to_datetime(soraDF["mth"], format='%b')
soraDF["mth_num"] = dt_series.dt.month

# Add 'yearmthkey'
soraDF["YearMonthKey"] = ((soraDF["Year"].astype(int)) * 100) + (soraDF["mth_num"])

# Casting variables
soraDF = soraDF.astype({'3M-SORA':'float64'})

# Remove unneeded col
soraDF = soraDF[['YearMonthKey','3M-SORA']]
```


### Adding 'GDP' variable

```
resourceID, seriesNo = varDict["gdp"]["id"], 1
header = {'User-Agent': 'Mozilla/5.0', "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8"}
url = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/{}?seriesNoORrowNo={}".format(resourceID,seriesNo)
request = Request(url,headers = header)
data = json.loads(urlopen(request).read())
listOfgdp = data["Data"]["row"][0]["columns"]
gdpDF = pd.DataFrame(listOfgdp)

# Renaming columns
gdpDF.rename(columns={'key': 'Year', 'value': 'gdp'}, inplace=True)

# Casting Variable
gdpDF = gdpDF.astype({'Year':'int','gdp':'float'})
```

### Adding 'Forex' variable

```
dateList, forexList = [],[]
resourceID,seriesNo = varDict["forex"]["id"], 1
header = {'User-Agent': 'Mozilla/5.0', "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8"}
url = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/{}?seriesNoORrowNo={}".format(resourceID,seriesNo)
request = Request(url,headers = header)
data = json.loads(urlopen(request).read())
listOfDateForex = data["Data"]["row"][0]["columns"]
forexDF = pd.DataFrame(listOfDateForex)

# Renaming columns
forexDF.rename(columns={'key': 'date', 'value': 'forex'}, inplace=True)

# Split 'date' into 'year' & 'mth'
forexDF[["Year", "mth"]] = forexDF["date"].str.split(" ", expand = True)

# convert 'mth' into numerical month
dt_series = pd.to_datetime(forexDF["mth"], format='%b')
forexDF["mth_num"] = dt_series.dt.month

# Add 'yearmthkey'
forexDF["YearMonthKey"] = ((forexDF["Year"].astype(int)) * 100) + (forexDF["mth_num"])

# Casting Variable
forexDF = forexDF.astype({'forex':'float'})

# Remove unneeded col
forexDF = forexDF[['YearMonthKey','forex']]
```

## Mapping of economic variables to JTC dataset

```
dfList = [
    {'df': lhfDF, 'on': 'Year'}, # Main DF from JTC
    {'df': popDF, 'on': 'Year'}, # Population Dataset
    {'df': soraDF, 'on': 'YearMonthKey'}, # SORA Dataset
    {'df': forexDF, 'on': 'YearMonthKey'}, # Forex Dataset
    {'df': gdpDF, 'on': 'Year'}, # GDP Datase
    {'df': landDF, 'on': 'Quarter'} # Land Supply Dataset
]

merged_df = dfList[0]['df']
display(len(merged_df.dtypes))
display(merged_df.head(3))

for i in range(1, len(dfList)):
    merged_df = pd.merge(
        merged_df,
        dfList[i]['df'],
        how="inner"
    )
    display(len(merged_df.dtypes))
    display(merged_df.head(3))

finalDF = merged_df
```

## Adding Tertiary variables

### Rental Index Varible
#### Custom Function - Resequence '2024Q1'  to '2024 1Q' 

```
def proper_q(quarter):
    quarter = quarter[0:4] + " " + quarter[-1] + "Q"
    return quarter
```

#### 

```
regionList = ["North","West","East","North-East"]

rentalDF = pd.read_csv('Rental_Index_Central.csv')
rentalDF["Region"] = "Central"
rentalDF.rename(columns = {'Rental Index of Multiple-User Factory (Central Region)':'rental_index'}, inplace = True)

for r in regionList:
    rentDF = pd.read_csv("Rental_Index_{}.csv".format(r))
    rentDF["Region"] = r
    rentDF.rename(columns = {'Rental Index of Multiple-User Factory ('+ r +' Region)':'rental_index'}, inplace = True)
    rentalDF = pd.concat([rentalDF, rentDF])

# Remove all NaN rows
rentalDF.dropna(inplace = True)

# Apply custom function
rentalDF["Period"] = rentalDF["Period"].apply(lambda x: proper_q(x)) 
```

### Geographical Data

Refer to [OneMap_API.md](https://github.com/ZackTanCZ/FactoryPBI_DB/blob/main/OneMap_API.md)

### Lagged Values

####
