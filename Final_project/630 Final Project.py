#!/usr/bin/env python
# coding: utf-8

# #### By exploring the bike ride share data, we will try to find answers for the below questions through thorough exploratory data analysis and predictive modelling.
# 
# For this project, we have used the data from Metro Bike Share in the city of Los Angeles, California that makes bikes available 24/7, 365 days a year in Downtown LA, Central LA, Port of LA and Westside. They started their operations during the second quarter of 2016.
# Based on this scenario, the project answers key questions e.g. a review of the quality of the available data, assumptions made, implications from revenue, tickets and pass-types, trip duration, number of trips by region and weather.
# 
# Data Files:
# Trip and station data is collected from [LA Metro Bike Share website](https://bikeshare.metro.net/about/data/). The station details are updated as of 1/4/2019 and the trip data ranges from 2018 Q1 to 2019 Q1.
# * Station data 
#     metro-bike-share-stations-2019-04-01.csv
# * Trip data
#     metro-bike-share-trips-2018-q1.csv, 
#     metro-bike-share-trips-2018-q2.csv,
#     metro-bike-share-trips-2018-q3.csv,
#     metro-bike-share-trips-2018-q4.csv,
#     metro-bike-share-trips-2019-q1.csv
#              
# Weather data is collected from [NOAA website](https://www.noaa.gov/weather). The date range overlaps the trip data, i.e. 1/1/2018 - 1/31/2019.
# * Weather data
#     WeatherLA.csv
#     
# ## Analysis and model building

# In[1]:


# standard library imports
import pandas as pd
import numpy as np
import warnings
import sys
#import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats

# third party imports

# suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
# graph output
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Preparation: Data Load and merging

# In[2]:


# read in the csv data files

def getdata_csv(file):
    '''
    Args:
        csv file name
    Output:
        dataframe of the csv data
    '''

    outdf = pd.read_csv(file)
    return outdf

# read station data
station = getdata_csv('data/metro-bike-share-stations-2019-04-01.csv')

# read quarterly trip data
quarter1 = getdata_csv('data/metro-bike-share-trips-2018-q1.csv')
quarter2 = getdata_csv('data/metro-bike-share-trips-2018-q2.csv')
quarter3 = getdata_csv('data/metro-bike-share-trips-2018-q3.csv')
quarter4 = getdata_csv('data/metro-bike-share-trips-2018-q4.csv')
quarter5 = getdata_csv('data/metro-bike-share-trips-2019-q1.csv')

# concatenate the quarterly trip data into one dataframe
trip = quarter5.append(quarter4, ignore_index=True, sort=False) .append(quarter3, ignore_index=True, sort=False) .append(quarter2, ignore_index=True, sort=False) .append(quarter1, ignore_index=True, sort=False)


# In[3]:


# check the station data
station.head()


# In[4]:


# check the combined trip data
trip.head()


# #### Merge trip and station data
# 
# Add important station information in trip data to create a unified dataset for further analysis

# In[5]:


# unnecessary columns from station data
columns = ['Station_ID','Station_Name','Go_live_date']

# add start station info
result = pd.merge(trip,
                  station,
                  left_on = 'start_station', 
                  right_on = 'Station_ID',
                  how='left').rename(index=str, columns={'Region ': 'start_region', 'Status': 'start_status'})

# remove unnecessary columns of station data
result.drop(columns, inplace=True, axis=1)

# add end station info
result = pd.merge(result,
                  station,
                  left_on = 'end_station', 
                  right_on = 'Station_ID',
                  how='left').rename(index=str, columns={'Region ': 'end_region', 'Status': 'end_status'})

# remove unnecessary columns of station data
result.drop(columns, inplace=True, axis=1)

# final result of combined station and trip data
result.head()


# #### Merge weather data
# 
# Weather data of LA weather stations for the 2018Q1 - 2019Q1 timeframe has been collected from National Oceanic and Atmospheric Administration (NOAA) website.

# In[6]:


# load the LA weather data
weather = getdata_csv('data/LATemp.csv')

# check the first five rows of the weather data
weather.head()


# In[7]:


# drop unnecessary columns
weather = weather.drop(columns=['Unnamed: 0'])

# standerdize the column names
weather = weather.rename(columns={'DATE': 'date', 'TMIN': 'tmin', 'TAVG': 'tavg', 'TMAX': 'tmax'})

# Checking the data types for the result data frame
result.dtypes


# In[8]:


# convert the start_time and end_time into date object
result['start_time'] = pd.to_datetime(result['start_time'])
result['end_time'] = pd.to_datetime(result['end_time'])

# check the data types to make sure its converted into date time object
result.dtypes


# In[9]:


# converting the date in the weather data frame to datetime object
weather['date'] = pd.to_datetime(weather['date'])

# checking to make sure the date is in date time object
weather.dtypes


# In[10]:


# filter the weather data frame to contain the values only upto the firts quarter of the year 2019
weather_to2019Q1 = weather[weather['date'] <= '2019-03-31']

# check the head of the Date column in weather
print('Head:') 
print(weather_to2019Q1.sort_values(by ='date').date.head())

# check the tail of the Date column in weather
print('Tail:') 
print(weather_to2019Q1.sort_values(by ='date').date.tail())


# In[11]:


# extract the date from the date time object and creating a DATE column 
result['date'] = pd.to_datetime(result['start_time']).dt.date

# check the head of the Date column in result
result.sort_values(by = 'date').date.head()


# In[12]:


# check the head of the Date column in result
result.sort_values(by = 'date').date.tail()


# In[13]:


result['date'] = pd.to_datetime(result['date'])

# since the two data frames have the same column 'DATE', we will join the data frame by DATE
trip_merged = pd.merge(result, weather_to2019Q1)

# Check the head of the top five rows of the combined data frame
trip_merged[['trip_id','date','tmin', 'tavg', 'tmax']].head()


# In[14]:


# check the tail of the last five rows of the combined data frame
trip_merged[['trip_id','date','tmin', 'tavg', 'tmax']].tail()


# In[15]:


# checkpoint - the analysis could be resumed from here
# write the final dataset to a csv file for future load
trip_merged.to_csv('data/Bikesharing-Weather-Combined.csv')


# We faced two roadblocks while integrating the wethaer data.
# 
# 1. Weather data was not at the same grain with station location in terms of zip code.
# 
# 2. Zip code level weather data is available but we could not find zip code level trip data.
# 
# Resolution:
# We took average of the daily meteorological attributes and applied to the trip data. 

# In[16]:


trip_merged.info()


# ### Exploratory Data Analysis (EDA)
# 
# In the EDA, we decided to discover various causal relationship between the attributes. This will include:
# 
# 1. Initial Data Profiling
# 
# 2. Plotting correlation matrix
# 
# 3. Total rides by membership type
# 
# 4. Average trip duration by membership type
# 
# 5. Top 5 stations with the most starts (showing # of starts)
# 
# 6. Plot of active stations
# 
# 7. Most Popular Trip
# 
# 8. Bike traffic throughout the year
# 
# #### 1. Initial Data Profiling
# 
# Looking for basic data distribution.

# In[17]:


round(trip_merged.describe(),2)


# #### 2. Plotting correlation matrix
# 
# Find correlation between the numeric fields to have idea between existing relationship in the data. 

# In[18]:


plt.figure(figsize = (12,12))
# Correlation Matrix
sns.heatmap(trip_merged             .drop(columns=['passholder_type','bike_type','start_region','start_status','end_region','end_status'])             .corr(), annot=True)

# since we need to correlation graph a bit larger for easy interpretation, change the output size
plt.rcParams["figure.figsize"] = (12,12)

plt.show()


# Some of the attributes show strong correlation e.g. `end_lat` vs `start_lat`, `end_lon` vs `start_lon` etc. But those are mostly because of the trips those started and ended at the same station. We were looking for correlation between trip attributes and weather attributes but no such strong correlation exist. Therefore the data needs further investigation.

# #### 3. Total rides by membership type

# In[19]:


trip_merged.columns


# In[20]:


countsbypasstype = trip_merged.groupby(['passholder_type']).trip_id.count().reset_index(name='counts')
countsbypasstype.columns


# In[21]:


countsbypasstype


# In[22]:


#plot the bar chart

# plot the graph
p = sns.barplot('passholder_type', 'counts', data = countsbypasstype)
plt.Figure(figsize=(8,8))
plt.title("Trips by passholder type")

plt.show(p)


# Monthly pass and Walk-up are usually the most common trip types. One Day Pass, Flex Pass and Annual Pass are not used that much when compared to the Monthly Pass.

# #### 4. Average trip duration by membership type
# 
# Going forward, let us calculate duration in minutes for easy plotting.

# In[23]:


# add minutes column for trip duration
trip_merged['minutes'] = round(trip_merged['duration']/60).astype(int)

# calculate trip duration based on minutes
avg_dur_membership = pd.DataFrame()
avg_dur_membership['avg_trip_duration'] = round(trip_merged.groupby('passholder_type')['minutes'].mean(),1)
avg_dur_membership = avg_dur_membership.reset_index()
avg_dur_membership['passholder_type'] = avg_dur_membership['passholder_type'].astype('object')

# plot the graph
g = sns.barplot('passholder_type', 'avg_trip_duration', data = avg_dur_membership)
plt.Figure(figsize=(20,20))
plt.title("Average Trip Duration by Membership Type based on Minutes")

plt.show(g)


# #### 5. Top 5 stations with the most starts (showing # of starts)
# 
# We want to check which station should be adequately stuffed based on usage.

# In[24]:


# data for Top 5 Stations visual
top5 = pd.DataFrame()
top5['Station'] = trip_merged['start_station'].value_counts().head().index
top5['Number of Starts']=trip_merged['start_station'].value_counts().head().values
top5['Station'] = top5['Station'].astype('category')
top5['Station'] = top5.Station.cat.remove_unused_categories()

# plot the top 5 stations
sns.barplot('Station', 'Number of Starts', data = top5)
plt.xticks(rotation=40, ha = 'right')
plt.title("Top 5 LA Metro Bike Stations by Number of Starts")
plt.show()


# #### 6. Plot of active stations
# 
# Geoplotting the locations of active stations. Since we do not have lat-long for the unique stations in the station, we will get the active stations from the station data and lat-long from the trip data.

# In[25]:


# get active stations
station_act = pd.DataFrame()
station_act = station[station.Status == 'Active'][['Station_ID']] .rename(index=str, columns={'Station_ID': 'station_id'})

# add the lat-long from trip data
# first try the start stations
start_geo = trip_merged[['start_station','start_lat','start_lon']] .rename(index=str, columns={'start_station': 'station_id', 'start_lat': 'lat', 'start_lon': 'lon'})

end_geo = trip_merged[['end_station','end_lat','end_lon']] .rename(index=str, columns={'end_station': 'station_id', 'end_lat': 'lat', 'end_lon': 'lon'})

station_geo = start_geo.append(end_geo).drop_duplicates().dropna()


# #### 7. Most Popular Trip
# 
# The intention is to find the most popular trip in the LA bike route so that the promotion is effectively directed.

# In[26]:


# calculate the number of trips between each stations
trips_df = pd.DataFrame()
trips_df = trip_merged.groupby(['start_station','end_station']).size().reset_index(name = 'Number of Trips')
trips_df = trips_df.sort_values('Number of Trips', ascending = False)
trips_df['Starting Station ID'] = trips_df['start_station'].astype('str')
trips_df['Ending Station ID'] = trips_df['end_station'].astype('str')
trips_df['Trip'] = trips_df['Starting Station ID'] + ' to ' + trips_df['Ending Station ID']
trips_df = trips_df[:10]
trips_df = trips_df.drop(['Starting Station ID', 'Ending Station ID'], axis = 1)
trips_df = trips_df.reset_index()

# find the most popular trips
g = sns.barplot('Number of Trips','Trip', data = trips_df)
plt.title("Most Popular Trips")
for index, row in trips_df.iterrows():
    g.text(row['Number of Trips']-50,index,row['Number of Trips'], 
             color='white', ha="center",fontsize = 10)
plt.show()


# #### 8. Bike traffic through the year

# Check the trend of trips (trip counts) by summarizing trip counts for each date

# In[27]:


countbydates = trip_merged.groupby(['date']).trip_id.count().reset_index(name='counts')
countbydates.head()


# In[28]:


sns.lineplot(x="date", y="counts",
                data=countbydates)

plt.title("Trips throughout the year - Trend analysis")


# Looks like there are some interesting peaks around October 2018 and December 2018

# #### 9. Exploring trips by dates through scatter plot

# In[29]:


# Plot
plt.scatter(countbydates.date, countbydates.counts,alpha=0.5)
plt.title('Scatter plot for Date Vs Trip Counts')
plt.xlabel('Dates')
plt.ylabel('Trip Counts')
plt.show()


# #### 10. Exploring trips interms of weekdays

# In[30]:



trip_merged.columns


# In[31]:


# derving 'day of week' attribute from date
trip_merged['day_of_week'] = trip_merged['date'].dt.day_name()
trip_merged.head()


# In[32]:


# summarizing trips with dates and day of weeks
countbywkday = trip_merged.groupby(['date', 'day_of_week']).trip_id.count().reset_index(name='counts')
countbywkday.head()


# In[33]:


sns.scatterplot(x = countbywkday.date,y = countbywkday.counts, hue = countbywkday.day_of_week, alpha=0.5)

# control x and y limits
plt.xlim('2017-12-01', '2019-05-01')


# In[34]:


sns.lineplot(x = countbywkday.date,y = countbywkday.counts, hue = countbywkday.day_of_week)


# From the above chart, looks the most trips between 2018-01-01 to 2019-03-31 happend on Sundays

# In[35]:


pd.get_dummies(trip_merged)


# In[36]:


trip_merged.shape


# # Predictive Analytics

# # 1) Predictive modelling for predicting usage rate for a given station

# ### Data Preparation for the model

# In[37]:


trip_merged.columns


# In[38]:


trip_merged.shape


# In[39]:


trip_merged.head()


# In[40]:


#outstation_df = unique(trip_merged['start_station', 'date'])


# In[41]:


outstation_df = pd.DataFrame()
outstation_df['Station'] = trip_merged['start_station'].value_counts().head().index
outstation_df['Number of Starts']=trip_merged['start_station'].value_counts().head().values


# In[42]:


outstation_df.head()


# In[43]:


outstation_df = pd.DataFrame()
outstation_df = trip_merged.groupby(['start_station', 'date','day_of_week'])['trip_id'].count().reset_index()
outstation_df.rename(columns = {'trip_id': 'outgoingbikes'}, inplace = True)
outstation_df.head()


# In[44]:


instation_df = pd.DataFrame()
instation_df = trip_merged.groupby(['end_station', 'date','day_of_week'])['trip_id'].count().reset_index()
instation_df.rename(columns = {'trip_id': 'incomingbikes'}, inplace = True)
instation_df.head()


# In[45]:


#trip_merged.groupby(['date','start_station'])['trip_id'].count().unstack()


# In[46]:


station.columns
station.head()


# In[47]:


# unique list of stations
len(outstation_df.start_station.unique())


# In[48]:


# unique list of stations
len(instation_df.end_station.unique())
#instation_df.end_station.value_counts()


# In[49]:


# unique list of stations
len(station.Station_ID.unique())


# In[50]:


# merging all data sets for modelling
print(instation_df.shape)
print(outstation_df.shape)


# In[51]:


merge1df = pd.merge(station,outstation_df, left_on = 'Station_ID', right_on='start_station', how = 'outer' )
merge1df.shape


# In[52]:


# unique list of stations
len(merge1df.Station_ID.unique())


# In[53]:


print(len(merge1df[merge1df.Station_ID.isnull()]))


# In[54]:


merge1df[merge1df.Station_ID.isnull()]


# In[55]:


# looks like station 4108 is not present in station data. So, lets add it
merge1df.Station_ID.fillna(merge1df.start_station, inplace = True)


# In[56]:


print(len(merge1df[merge1df.Station_ID.isnull()]))
merge1df[merge1df.Station_ID.isnull()]


# In[57]:


print(len(merge1df.start_station.unique()))
print(len(merge1df[merge1df.start_station.isnull()]))


# In[58]:


merge1df[merge1df.start_station.isnull()]


# In[59]:


merge1df.head()


# In[60]:


#merge1df.rename(columns = {'date':'date1'}, inplace = True)
#merge1df.columns


# In[61]:


merge2df = pd.merge(merge1df, instation_df, left_on = ['Station_ID','date','day_of_week'], 
                                           right_on=['end_station', 'date','day_of_week'], how = 'outer' )
merge2df.shape


# In[62]:


merge2df.columns


# In[63]:


merge2df.head()


# In[64]:


# Check for unmatched stations (end stations)
print(len(merge2df[merge2df.Station_ID.isnull()]))


# In[65]:


merge2df[merge2df.Station_ID.isnull()].head()


# In[66]:


# looks like stations and date combination records doesnt exist, lets add those to the station_ID
merge2df.Station_ID.fillna(merge2df.end_station, inplace = True)


# In[67]:


# Now check for unmatched stations (end stations)
print(len(merge2df[merge2df.Station_ID.isnull()]))


# In[68]:


weather.columns


# In[69]:


# joining weather data
merge3df = pd.merge(merge2df,weather, left_on = ['date'], right_on = ['date'] , how = 'outer')
merge3df.shape


# In[70]:


merge3df.head()


# In[71]:


summarydf = merge3df.copy()
summarydf.to_excel("data/summarydfv1.xlsx")
summarydf = summarydf.drop(['start_station', 'end_station'], axis=1) #, inplace = True)

summarydf['year'] = pd.DatetimeIndex(summarydf['date']).year
summarydf['month'] = pd.DatetimeIndex(summarydf['date']).month

summarydf.head()


# In[72]:


summarydf.shape


# In[73]:


# Check for how many nulls station id are there
print(len(summarydf[summarydf.Station_ID.isnull()]))


# In[74]:


# Check for  nulls station id in detail
summarydf[summarydf.Station_ID.isnull()]


# In[75]:


# we can drop these records if there are no bikes trips on such dates
summarydf = summarydf.dropna(axis=0, subset=['Station_ID'])


# In[76]:


# Check for  nulls station id
print(len(summarydf[summarydf.Station_ID.isnull()]))


# In[77]:


summarydf.shape


# ### Consdering impact of hoildays of bike rentals

# In[78]:


from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidaydf = pd.DataFrame()
holidaydf['holiday_date'] = cal.holidays('2018', '2020')
holidaydf['holidayflag'] = 1
print(type(holidaydf))


# In[79]:


holidaydf


# In[80]:


# merging holiday calendar and summary data set
summarydf = pd.merge(summarydf,holidaydf, left_on = ['date'], right_on = ['holiday_date'] , how = 'outer')
summarydf.head()


# In[81]:


summarydf.shape


# In[82]:


# Filling 0's for NaN values - representing not a holiday on that day
summarydf['holidayflag'].fillna(0, inplace=True)


# In[83]:


summarydf[summarydf['holidayflag'] ==0].head()


# In[84]:


# check number of records on holidays and non holidays
summarydf.groupby(['holidayflag'])['Station_ID'].count()


# In[85]:


# converting saturdays and sundays into holidays as well
summarydf.loc[summarydf['day_of_week'] == 'Sunday', 'holidayflag'] = 1
summarydf.loc[summarydf['day_of_week'] == 'Saturday', 'holidayflag'] = 1


# In[86]:


# check number of records on holidays including saturdays and sundays and non - holidays
summarydf.groupby(['holidayflag'])['Station_ID'].count()


# In[87]:


summarydf.describe()


# In[88]:


summarydf.info()


# In[89]:


# checking for null records
summarydf[summarydf['Station_ID'].isna()]


# In[90]:


# deleting the above null records as we dont have any bike trip details for those dates
summarydf = summarydf.dropna(axis=0, subset=['Station_ID'])


# In[91]:


summarydf.info()


# In[92]:


# checking for null records
print(len(summarydf[summarydf['date'].isna()]))
summarydf[summarydf['date'].isna()]


# In[93]:


# dropping the above records
summarydf = summarydf.dropna(axis=0, subset=['date'])
# checking for null records
print(len(summarydf[summarydf['date'].isna()]))


# In[94]:


# StationId is in float format, converting that ID into categorical variables
summarydf['Station_ID'] = summarydf['Station_ID'].astype(dtype=np.int64)
summarydf['Station_ID'] = summarydf['Station_ID'].astype('category')

summarydf['year'] = summarydf['year'].astype(dtype = np.int64)
summarydf['month'] = summarydf['month'].astype(dtype = np.int64)
summarydf.head()


# In[95]:


#taking backup of dataframe
summarydfcopy = summarydf.copy()
summarydf.shape


# ## Data Pre-processing

# ### Feature Engineering

# In[96]:


summarydf = summarydfcopy.copy()


# In[97]:


summarydf.columns


# In[98]:


#Removing unnecessary attributes
summarydf = summarydf.drop(['Station_Name','Go_live_date','Region ', 'date', 'holiday_date'] , axis = 1 )
summarydf.head()


# In[99]:


#checking for nulls in temperature data
print(len(summarydf[summarydf['tavg'].isna()]))


# In[100]:


summarydf[summarydf['tavg'].isna()]


# In[101]:


# dropping the above null records as there no trips associated with those stations on those days
summarydf = summarydf.dropna(axis=0, subset=['tavg'])

#checking for nulls in temperature data after deleting rows to confirm
print(len(summarydf[summarydf['tavg'].isna()]))


# In[102]:


summarydf['totaltrips'] =  summarydf['outgoingbikes'] + summarydf['incomingbikes']

#checking for nulls in temperature data after deleting rows to confirm
print(len(summarydf[summarydf['totaltrips'].isna()]))

# dropping the above null records as there no trips associated with those stations on those days
summarydf = summarydf.dropna(axis=0, subset=['totaltrips'])
summarydf.head()


# In[103]:


#checking for nulls in temperature data after deleting rows to confirm
print(len(summarydf[summarydf['totaltrips'].isna()]))


# In[104]:


#checking for nulls in temperature data after deleting rows to confirm
print(len(summarydf[summarydf['totaltrips'] < 1]))


# In[105]:


# converting holiday flag into integer
summarydf['holidayflag'] = summarydf['holidayflag'].astype(dtype = np.int64)


# In[106]:


summarydf.to_excel("data/summary.xlsx")


# In[107]:


# Create dummy variables for categorical variables - Station_Id, day of week and status
summarydf_features = pd.DataFrame()
summarydf_features = pd.get_dummies(summarydf)
summarydf_features.head()


# In[108]:


summarydf_features.columns


# In[109]:


# Removing one category out of each categorical dummy variable
summarydf_features = summarydf_features.drop(['Station_ID_4385','Status_Inactive','day_of_week_Tuesday','totaltrips'] , axis = 1 )
summarydf_features.head()


# ## Building Model

# #### We are trying to build a model to predict the number of bike trips from that station and number of trips ending at that station. So we have 2 dependant variables and rest are independant variables.

# ### Model Training and Testing

# In[120]:


# Import required modules for modelling
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_absolute_error # Mean Absolute Error
from sklearn.metrics import mean_squared_error # Mean Squared Error
from sklearn.metrics import r2_score #R² Score

import matplotlib.pyplot as plt
import seaborn as sns


# In[121]:


# checking Nans in outgoingbikes columns
print(len(summarydf_features[summarydf_features['outgoingbikes'].isna()]))


# In[122]:


# checking Nans in incomingbikes columns
print(len(summarydf_features[summarydf_features['incomingbikes'].isna()]))


# In[123]:


# creating 2 data sets for 2 dependant variables
summarydf_features1 = summarydf_features.dropna(axis=0, subset=['outgoingbikes'])
summarydf_features2 = summarydf_features.dropna(axis=0, subset=['incomingbikes'])


# In[124]:


print(len(summarydf_features1))
print(len(summarydf_features2))


# In[125]:


# Defining independant variables
features1 = summarydf_features1.iloc[ : ,2:200]
features1.head()


# In[126]:


# Defining targets

target1 = summarydf_features1['outgoingbikes'] # Trips that started from that station


# In[127]:


# ensuring lenghts of the features and targets
print(len(features1))
print(len(target1))


# In[128]:


# splitting data into train (75%) and test (25%)
# for first dependant variable - target1
X1_train, X1_test, y1_train, y1_test = train_test_split(features1, target1, test_size = 0.25, random_state = 0)


# In[129]:


# get list of columns having NaN values
X1_train.loc[ : , X1_train.isna().any()]


# In[130]:


# get list of columns having NaN values
print(len(y1_train.isna()))


# #### Feature Scaling

# In[131]:


# Feature Scaling
sc_x = StandardScaler()
X1_train_std = sc_x.fit_transform(X1_train)
X1_test_std = sc_x.transform(X1_test)

sc_y = StandardScaler()
y1_train_std = sc_y.fit_transform(y1_train[:, np.newaxis]).flatten()
y1_test_std = sc_y.fit_transform(y1_test[:, np.newaxis]).flatten()
#(y1_train.reshape(-1, 1) )


# #### Since the variables are not linear in nature and the predicting variables are continuous variables, we choose to select non- linear regression analysis. 

# ### Implementing Decision Tree Regressor

# #### Fitting Model

# In[132]:


# Fitting Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
dtmodel = regressor.fit(X1_train, y1_train)

# Predicting results for training data set
y1_pred_train = regressor.predict(X1_train)

# Predicting results for test data
y1_pred_test = regressor.predict(X1_test)

residuals_dt_train = y1_train - y1_pred_train # actual training value - predicted value for training data set
residuals_dt_test = y1_test - y1_pred_test # actual test value - predicted value for testing data set


# #### Model Evaluation on Training Data

# In[133]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error is : ", mean_absolute_error(y1_train, y1_pred_train))

#Mean Squared Error
from sklearn.metrics import mean_squared_error
print("Mean Squared Error : ", mean_squared_error(y1_train, y1_pred_train))

#R² Score
from sklearn.metrics import r2_score
print("R-square value : ", r2_score(y1_train, y1_pred_train))


# #### Model Evaluation on Training Data

# In[134]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error is : ", mean_absolute_error(y1_test, y1_pred_test))

#Mean Squared Error
from sklearn.metrics import mean_squared_error
print("Mean Squared Error : ", mean_squared_error(y1_test, y1_pred_test))

#R² Score
from sklearn.metrics import r2_score
print("R-square value : ", r2_score(y1_test, y1_pred_test))


# In[135]:


plt.scatter(y1_pred_test,residuals_dt_test)
plt.hlines(y=0, xmin=-10, xmax=300, color='black', lw=2)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Decision Tree Regressor - Fitted Values Vs Residuals")
plt.show()


# In[136]:


# plotting residuals to see the error pattern

plt.figure(figsize = (10,8))

plt.scatter(y1_pred_train,residuals_dt_train,
               c='steelblue',
                edgecolor='white',
                marker='o',
                s=35,
                alpha=0.9,
                label='Train data')


plt.scatter(y1_pred_test,residuals_dt_test,
                c='limegreen',
                edgecolor='white',
                marker='s',
                s=35,
                alpha=0.9,
                label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title("Decision Tree Regressor - Fitted Values Vs Residuals")
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=500, lw=2, color='black')
#plt.xlim([-10, 50])
plt.show()


# #### Decision Tree Regressor Evaluation:

# As the errors are randomly distributed and do not see a pattern behind errors, decision tree algorithms seems to explain the variance in the model.

# ###  Implementing Random Forest Regressor

# In[137]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000,
                                    criterion='mse',
                                    random_state=1,
                                    n_jobs=-1)


rfmodel = forest.fit(X1_train, y1_train)

# Predicting a new result for test data 
y1_pred_train_rf = forest.predict(X1_train)


# Predicting a new result for test data 
y1_pred_test_rf = forest.predict(X1_test)


# #### Evaluating Random Forrest Regressor on training data set

# In[138]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error is : ", mean_absolute_error(y1_train, y1_pred_train_rf))

#Mean Squared Error
from sklearn.metrics import mean_squared_error
print("Mean Squared Error : ", mean_squared_error(y1_train, y1_pred_train_rf))

#R² Score
from sklearn.metrics import r2_score
print("R-square value : ", r2_score(y1_train, y1_pred_train_rf))


# #### Evaluating Random Forrest Regressor on test data set

# In[139]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error is : ", mean_absolute_error(y1_test, y1_pred_test_rf))

#Mean Squared Error
from sklearn.metrics import mean_squared_error
print("Mean Squared Error : ", mean_squared_error(y1_test, y1_pred_test_rf))

#R² Score
from sklearn.metrics import r2_score
print("R-square value : ", r2_score(y1_test, y1_pred_test_rf))


# As expected, Mean Absolute error on test data is more than that of prediction on training data.Similarly the model could explain the 94% of the variance on training data set but could explain only 65.5% of variation on test data. 
# 
# Let us visualize the errors with respective the predicted value.

# In[140]:


# Predicting a new result for training data 

residuals_train_rf = y1_train - y1_pred_train_rf # actual test value - predicted value

residuals_test_rf = y1_test - y1_pred_test_rf # actual test value - predicted value


# In[141]:


plt.figure(figsize = (12,12))

plt.scatter(y1_pred_train_rf,residuals_train_rf,
               c='steelblue',
                edgecolor='white',
                marker='o',
                s=35,
                alpha=0.9,
                label='Train data')


plt.scatter(y1_pred_test_rf,residuals_test_rf,
                c='limegreen',
                edgecolor='white',
                marker='s',
                s=35,
                alpha=0.9,
                label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title("Random Forrest Regressor - Fitted Values Vs Residuals")
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=500, lw=2, color='black')
#plt.xlim([-10, 50])
plt.show()


# ### With Standardised Train and test data

# In[142]:


# Random Forest Regressor

forest2 = RandomForestRegressor(n_estimators=1000,
                                    criterion='mse',
                                    random_state=1,
                                    n_jobs=-1)


rfmodel2 = forest2.fit(X1_train_std, y1_train_std)

# Predicting a new result for test data 
y1_pred_train_rf2 = forest2.predict(X1_train_std)


# Predicting a new result for test data 
y1_pred_test_rf2 = forest2.predict(X1_test_std)


# #### Evaluating Random Forest Regression on scaled version of trained dataset

# In[143]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error is : ", mean_absolute_error(y1_train_std, y1_pred_train_rf2))

#Mean Squared Error
from sklearn.metrics import mean_squared_error
print("Mean Squared Error : ", mean_squared_error(y1_train_std, y1_pred_train_rf2))

#R² Score
from sklearn.metrics import r2_score
print("R-square value : ", r2_score(y1_train_std, y1_pred_train_rf2))


# #### Evaluating Random Forest Regression on scaled version of test dataset

# In[144]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error is : ", mean_absolute_error(y1_test_std, y1_pred_test_rf2))

#Mean Squared Error
from sklearn.metrics import mean_squared_error
print("Mean Squared Error : ", mean_squared_error(y1_test_std, y1_pred_test_rf2))

#R² Score
from sklearn.metrics import r2_score
print("R-square value : ", r2_score(y1_test_std, y1_pred_test_rf2))


# When I used the scaled data set, the evaluation metric - Mean Absolute Error is much better, without loosing the variance power (R-sqaure) - almost 66%

# In[145]:



# plotting residuals to see a pattern
residuals_train_rf2 = y1_train_std - y1_pred_train_rf2 # actual test value - predicted value

# plotting residuals to see a pattern
residuals_test_rf2 = y1_test_std - y1_pred_test_rf2 # actual test value - predicted value


plt.figure(figsize = (10,10))

plt.scatter(y1_pred_train_rf2,residuals_train_rf2,
               c='steelblue',
                edgecolor='white',
                marker='o',
                s=35,
                alpha=0.9,
                label='Train data')


plt.scatter(y1_pred_test_rf2,residuals_test_rf2,
                c='limegreen',
                edgecolor='white',
                marker='s',
                s=35,
                alpha=0.9,
                label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title("Random Forrest Regressor - Fitted Values Vs Residuals (Standardized data)")
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
#plt.xlim([-10, 50])
plt.show()


# The residual plot still explains the variance of the Random Forest Regressor. 
# 
# Final selection of the model is - Random Forest Regressor with scaled data set

# ## Predicting the values for the 2nd dependant variable -  Incoming bikes 

# #### Feature Scaling (for 2nd dependant variable)

# In[146]:


# Defining targets
target2 = summarydf_features1['incomingbikes'] # Trips that started from that station

# splitting data into train (75%) and test (25%)
# for second dependant variable - target2
X1_train, X1_test, y2_train, y2_test = train_test_split(features1, target2, test_size = 0.25, random_state = 0)

# Feature Scaling
sc_x = StandardScaler()
X1_train_std = sc_x.fit_transform(X1_train)
X1_test_std = sc_x.transform(X1_test)

sc_y = StandardScaler()
y2_train_std = sc_y.fit_transform(y2_train[:, np.newaxis]).flatten()
y2_test_std = sc_y.fit_transform(y2_test[:, np.newaxis]).flatten()


# #### Building Model

# In[147]:


# Random Forest Regressor

forest3 = RandomForestRegressor(n_estimators=1000,
                                    criterion='mse',
                                    random_state=1,
                                    n_jobs=-1)


rfmodel3 = forest3.fit(X1_train_std, y2_train_std)

# Predicting a new result for test data 
y2_pred_train_rf3 = forest3.predict(X1_train_std)


# Predicting a new result for test data 
y2_pred_test_rf3 = forest3.predict(X1_test_std)


# #### Evaluating model for 2nd variable with training data set

# In[148]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error is : ", mean_absolute_error(y2_train_std, y2_pred_train_rf3))

#Mean Squared Error
from sklearn.metrics import mean_squared_error
print("Mean Squared Error : ", mean_squared_error(y2_train_std, y2_pred_train_rf3))

#R² Score
from sklearn.metrics import r2_score
print("R-square value : ", r2_score(y2_train_std, y2_pred_train_rf3))


# #### Evaluating model for 2nd variable with test data set

# In[149]:


#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error is : ", mean_absolute_error(y2_test_std, y2_pred_test_rf3))

#Mean Squared Error
from sklearn.metrics import mean_squared_error
print("Mean Squared Error : ", mean_squared_error(y2_test_std, y2_pred_test_rf3))

#R² Score
from sklearn.metrics import r2_score
print("R-square value : ", r2_score(y2_test_std, y2_pred_test_rf3))


# The above evaluation confirms our expectation that the model fits almost perfectly for training data set with Mean Absolute Error or 0.106, while test data set has MAE of 0.32. The model explains the variance in dependant variable upto 58%

# In[150]:



# plotting residuals to see a pattern
residuals_train_rf3 = y2_train_std - y2_pred_train_rf3 # actual test value - predicted value

# plotting residuals to see a pattern
residuals_test_rf3 = y2_test_std - y2_pred_test_rf3 # actual test value - predicted value


plt.figure(figsize = (10,10))

plt.scatter(y2_pred_train_rf3,residuals_train_rf3,
               c='steelblue',
                edgecolor='white',
                marker='o',
                s=35,
                alpha=0.9,
                label='Train data')


plt.scatter(y2_pred_test_rf3,residuals_test_rf3,
                c='limegreen',
                edgecolor='white',
                marker='s',
                s=35,
                alpha=0.9,
                label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title("Random Forrest Regressor - Fitted Values Vs Residuals (Standardized data)")
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
#plt.xlim([-10, 50])
plt.show()


# As no pattern can be observed from the above residual vs predicted value for 2nd dependant variable (Incoming bikes), we can safely assume that the model explains well

# # 2) Unsupervied - Clustering of the stations

# #### Problem Statement
# 
# Using unsupervised learning, cluster the stations and find the key influencers for the trip count.
# 
# We will start with the `summarydf` data frame and refine the features. 
# 
# For clustering, we decided to exclude the status column. Since status is specific to the time of acquiring the data, dropping record based on status or using status as a feature might lead to data loss or inaccuracy.

# In[151]:


# create mapping for days

numdays = {'day_of_week': {'Monday': 1,
                           'Tuesday': 2,
                           'Wednesday': 3,
                           'Thursday': 4,
                           'Friday': 5,
                           'Saturday': 6,
                           'Sunday': 7}}

# drop column that will not be used

basedf = summarydf.drop(['Status'], axis=1)


# replace day names by numbers (weeks starts on Monday - 1)

basedf.replace(numdays, inplace=True)

basedf.head()


# We plan to use unsupervised learning and need to convert the columns into integer columns. For that purpose, the following adjustments are needed in the data.
# 
# * `totaltrips` needed to be converted to integer.
# * `tavg` will be encoded and used as the only feature for temperature.
# * `Station_ID` will be dropped

# In[152]:


# Drop unnecessary columns

dropcols = ['Station_ID','outgoingbikes','incomingbikes','tmin','tmax']
featuredf = basedf.copy()
featuredf.drop(dropcols,inplace=True,axis=1)

# Convert to int
featuredf = featuredf.astype('int64', copy=False)

featuredf.head()


# For the purpose of predicting sation cluster, we would apply unsupervised learning, _k-means_ to be precise.

# In[153]:


# keep only relevant columns

dropcols = ['tmin','tmax','year','month','holidayflag','day_of_week','outgoingbikes','incomingbikes']
featuredf = basedf.copy()
featuredf.drop(dropcols,inplace=True,axis=1)

featuredf.head()


# In[154]:


# encode totaltrips to traffic - high, medium, low

prepdf = featuredf[["Station_ID", "totaltrips","tavg"]].groupby(['Station_ID'], as_index=False) .mean().sort_values(by='totaltrips', ascending=False)

# light traffic
prepdf.loc[prepdf.totaltrips < 50.0, 'traffic' ] = 0
# medium traffic
prepdf.loc[(prepdf.totaltrips >= 50.0) & (prepdf.totaltrips < 100.0), 'traffic'] = 1
# high traffic
prepdf.loc[prepdf.totaltrips >= 100.0, 'traffic' ] = 2

prepdf.head()


# In[155]:


# test train split

msk = np.random.rand(len(prepdf)) < 0.8
train = prepdf[msk]
test = prepdf[~msk]
test.drop('totaltrips',inplace=True,axis=1)

print(len(prepdf))
print(len(test))
print(len(train))


# In[156]:


# create dependent and independent varibales

X = np.array(train.drop(['traffic','totaltrips','tavg'], 1).astype(float))
y = np.array(train['traffic'])

# we want to cluster the stations into 3: low, medium and high traffic
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)


# In[157]:


# look at the number of stations clusterd correctly

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# To improve the performance of the model we are tweaking some parameters. Following parameters will be tweaked and the model will be fit again:
# algorithm, max_iter and n_jobs.

# In[158]:


# fit model with altered parameters

kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
kmeans.fit(X)


# In[159]:


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# As a practice, we would like to scale the values in the dataset so that changes in feature values have least influence on each other.

# In[161]:


# scale feature values
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans.fit(X_scaled)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))


# # 3) Classifying the trips based on passholder Type

# In[162]:


# Visualizing the first five rows of the trip_merged dataset

trip_merged.head()


# In[163]:


# Checking the column names for the trip merged dataset

trip_merged.columns


# In[164]:


# Checking teh number of different types of pass holder types

trip_merged.passholder_type.unique()


# In[165]:


# The goal of the experiment is to classify the dataset based on the passholder type 
# i.e. in other words predicting pass holder tyype

trip_merged.passholder_type.value_counts()


# In[166]:


# Since the dataset is imbalanced between different pass holder type, I am going to collect the data corresponding to the highest two
# I am going to call the new data frame with only two pass holder types - Monthly_WalkUp dataframe

monthly_walkup = trip_merged[(trip_merged['passholder_type']=='Monthly Pass') | (trip_merged['passholder_type']== 'Walk-up')]


# In[167]:


# Checking the shape of the new dataframe i.e monthly_walkup

monthly_walkup.shape


# In[168]:


# Checking the unique types of pass holder type
# As we can see, now the data frame has only two types of data

monthly_walkup.passholder_type.value_counts()


# In[169]:


# Model Building
# Problem Statement: Predicting whether a pass holder has a monthly pass or walk up type
# Classification Model- Random Forests


# In[170]:


# Mapping the pass holder type variable into numerics

monthly_walkup['passholder_type'].replace(['Monthly Pass','Walk-up'],[0,1],inplace=True)


# In[171]:


# Checking the monthly pass holder type to see if it is mapped correctly

monthly_walkup.passholder_type.unique()


# In[172]:


# Attribute selection for the model
# dropping the columns that may not effect teh classification

monthly_walkup.drop(['trip_id', 'start_time', 'end_time', 'start_lat', 'start_lon', 'end_station', 
                     'end_lat', 'bike_id', 'start_status', 'end_region', 'end_status', 'date'], axis=1, inplace=True)


# In[173]:


# Visualizing the current state of the dataset
monthly_walkup.drop(['end_lon'],axis=1, inplace=True)
monthly_walkup.dtypes


# In[174]:


# Checking for nan values

monthly_walkup.isnull().sum()


# In[175]:


# Based on the percentage of null values, dropping the bike type column

monthly_walkup.drop(['bike_type'],axis=1, inplace=True)


# In[176]:


# dropping rows with null values

monthly_walkup.dropna(axis=0, how='any', inplace = True)


# In[177]:


# Checking for nan values

monthly_walkup.isnull().values.any()


# In[178]:


# Converting the object variables and get dummies for the model

monthly_walkup = pd.get_dummies(monthly_walkup, columns = ['trip_route_category','start_region'], drop_first = True)


# In[179]:


# converting start station to categorical

monthly_walkup['start_station'] = monthly_walkup['start_station'].astype('category')


# In[180]:


# Converting the object variables and get dummies for the model

monthly_walkup = pd.get_dummies(monthly_walkup, columns = ['day_of_week'], drop_first = True)


# In[181]:


# Shape of the final dataset

monthly_walkup.shape


# In[182]:


# Visualizing the final dataset before fitting the model

monthly_walkup.head()


# In[183]:


# Visualizing the final dataset before fitting the model

monthly_walkup.head()


# In[184]:


# Getting the X and y values from the dataframe

X = monthly_walkup.loc[:, monthly_walkup.columns != 'passholder_type']
y = monthly_walkup.loc[:,'passholder_type']


# In[185]:


# Importing the test and train split library

from sklearn.model_selection import train_test_split

# Splitting the dataset into test and train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[186]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, random_state = 42)
classifier.fit(X_train, y_train)


# In[187]:


classifier.fit(X_train, y_train)


# In[188]:


# Getting the predictions for the X_test
y_pred = classifier.predict(X_test)


# In[189]:


# Calculating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[190]:


# Calculating area under the curve
from sklearn import metrics
metrics.roc_auc_score(y_test, y_pred)


# In[191]:


# Calculating the accuracies from 3 fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train,  y = y_train, n_jobs = -1, cv = 5)
accuracies.mean()

