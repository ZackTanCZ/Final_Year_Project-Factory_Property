#

## Custom Function

```
def addLaggedpsf(df):
    
    # Subset the DataFrame to only include 'Contract Date','Unit Price ($ psf)','Project Name' & 'Street Name' 
    dbDF_nonan = df[["ID","Project Name","Street Name","Contract Date","Unit Price ($ psf)"]]
    
    # Remove any duplicates and reset the DataFrame index
    projNameDF = dbDF.drop_duplicates(subset=["Project Name","Street Name"]).reset_index(drop = True)
    
    # Replace NaN Project Name with their Street Name
    projNameDF.loc[:,"Project Name"] = projNameDF["Project Name"].fillna(projNameDF["Street Name"])
    
    # Get all the unique Project Names or Street Names (if Project Name is not available)
    listofProject = projNameDF['Project Name'].unique()
    print('Number of Projects: {}'.format(len(listofProject)))

    # main df with nan project names replaces with street name
    dbDF_nonan.loc[:,"Project Name"] = dbDF_nonan["Project Name"].fillna(dbDF_nonan["Street Name"])
    
    fullDF = pd.DataFrame()
    for i in listofProject:
        
        # Creates a df for each project
        projFullDF = dbDF_nonan[dbDF_nonan['Project Name'] == i].reset_index(drop = True) 
        
        # convert 'Contract Date' to datetime
        projFullDF['Contract Date'] = pd.to_datetime(projFullDF['Contract Date'], dayfirst=True) 
        
        # sort Date in ascending order
        projFullDF.sort_values(by = 'Contract Date', ascending = True, inplace=True) 
        
        # reset the DataFrame index
        projFullDF.reset_index(drop=True, inplace=True)

        # subset the full project DF, join back later.
        projSubDF = projFullDF[['Unit Price ($ psf)']] 

        # Get the max value from a window of past three transactions
        projSubDF = projSubDF.rolling(3, closed= "left", min_periods= 1).max() 

        # replace NA values with the previous row's psf
        projSubDF.bfill(inplace=True) 
        
        projSubDF.rename(columns={'Unit Price ($ psf)':'lagged psf'}, inplace=True)
        projSubDF.reset_index(drop=True, inplace=True)        
        projFullDF = projFullDF.merge(projSubDF,left_index = True, right_index=True, sort=False)
        fullDF = pd.concat([fullDF, projFullDF], sort=False)
        fullDF.reset_index(drop=True, inplace=True)
    return fullDF
```

## Creating the 'lagged psf' column

```
# Read csv file for dataset
dbDF = pd.read_csv('PBIDB_Data.csv')

# Create ID column for mapping
dbDF.reset_index(inplace = True)
dbDF.rename(columns={'index':'ID'}, inplace = True)

# So unit price is considered a string
dbDF["Unit Price ($ psf)"] = dbDF["Unit Price ($ psf)"].apply(lambda s: float(s.replace(",", "")) if isinstance(s, str) and 
                                                              s.replace(",", "").replace(".", "").isdigit() 
                                                              else float(s) if not isinstance(s,str) and pd.notna(s) else
                                                              None)
# Create a new column 'lagged psf'
lagDF = addLaggedpsf(dbDF)

# Replace NaN values
lagDF.loc[:,"lagged psf"] = lagDF["lagged psf"].fillna(lagDF["Unit Price ($ psf)"])

# Merge both DataFrame
dbDF = dbDF.merge(lagDF, on = "ID")

# Remove unnessary columns and rename column
dbDF = dbDF[["ID","Contract Date_x","Project Name_x","Street Name_x","Unit Price ($ psf)_x","lagged psf"]]
dbDF.rename(columns = {"Contract Date_x":"Contract Date",
                       "Project Name_x":"Project Name",
                       "Street Name_x":"Street Name",
                       "Unit Price ($ psf)_x":"Unit Price ($ psf)"})


```
