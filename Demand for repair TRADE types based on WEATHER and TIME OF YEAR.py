#!/usr/bin/env python
# coding: utf-8

# #### Loading neccessary libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from datetime import date
import holidays
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, recall_score, classification_report
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
import joblib
from sklearn.base import clone


# #### Loading datasets needed to execute this project objective

# In[2]:


job = pd.read_excel(r'..\UPDATED DATA\04. Repairs\Job.xlsx')
pty_codes = pd.read_excel(r'..\UPDATED DATA\04. Repairs\Pty.xlsx')
sor = pd.read_excel(r'..\UPDATED DATA\04. Repairs\SORTrd.xlsx')
# weather data in Gloucester from 27 Feb 1996 to 15 June 2023
gl = pd.read_excel(r'..\UPDATED DATA\Weather Data\Gloucester.xlsx')
# weather codes from WMO - for states of sky - CLOUDY, CLEAR, RAIN, LIGHT DRIZZLE, SNOW etc
wmo = pd.read_csv(r'..\UPDATED DATA\wmo_codes.csv', header = None)


# In[3]:


# select country for importing holiday dates from holiday library in python
uk_holidays = holidays.UnitedKingdom()
# printing all the holidays in UnitedKingdom in year 2023 for demonstration
for ptr in holidays.UnitedKingdom(years = 2023).items():
    print(ptr)


# ##### Cleaning Job frame

# In[4]:


# converting job_report_date to datetime 
job['job_report_date'] = pd.to_datetime(job['reported-dat']).dt.date

# mapping priority code of each report with priority classifcation of repair (routine, cyclical, emergency, void, planned, otherm, inspection)
#pty_type = pty_codes[['pty_cde','pty_type']]
pty_map = dict(pty_codes[['pty_cde','pty_classification']].values)
job['priority'] = job['pty-cde'].map(pty_map)

# replacing OLD SOR trade codes with NEW ones
job['sortrd-cde-1'].replace(['BR', 'C', 'E', 'E1', 'F', 'G','GF', 'H', 'MI', 'P', 'PD', 'PO','R', 'SC', 'TI', 'W'], 
                             ['0B','0C','0E', '0E', '0F','0Z','0G','HP','NS','0P', '0D','0D','0R', '0S', '0I','0C'], 
                             inplace=True)

# mapping remaining NEW SOR trade code to respective textual descriptions (Trade Names)
sor.set_index('cde', inplace = True)
sor_map = sor.to_dict()['dsc']
job['sor'] = job['sortrd-cde-1'].map(sor_map)

job.head()


# ###### Analysing Trades by number of times each is required for repair/maintenance job
# Plumbing jobs are the most frequent overall, closely followed by Electrical jobs. A large number of Gas Fitting jobs are also reported/executed as well as Carpentry jobs.
# A few classes of Trade tpes are extremely rare making them anomalous in nature. Fo example, 'Health', 'Kenny Thomson', 'Bonus', 'Time Extension', etc.. Other rare Trade classes (with a count of less than 1% in the entire dataset) are dealt with in the following cell

# In[5]:


job1 = job[['job_report_date', 'sor']]

job1['sor'].value_counts()


# ###### Reducing classes for better predictive perfromance of models

# In[6]:


# very few datapoints for these classes (less than 20)
# remove extremely rare classes because models cannot learn much on sparse data

job_reduced = job1.drop(job1[job1['sor']=='TIME EXTENSION'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='HEALTH'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='Kenny Thomson'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='BONUS'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='Void Clearance'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLANNED WORKS'].index, axis = 0)

# dropping dummy tickets as it is not vital to predict this class
# dropping HandyLink as it is an obsolete Trade class, not used by GCH anymore. 
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='DUMMY TICKETS'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='HandyLink'].index, axis = 0)


# dual Trade Type (unable to classify)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLUMBER/CARPENT'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLUMBER/ELECTRI'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLASTERER/TILER'].index, axis = 0)

# grouping similar Trades together
job_reduced['sor'] = job_reduced['sor'].replace('ESTATE INS', 'INSPECTION')
job_reduced['sor'] = job_reduced['sor'].replace('PAINT/DECORATOR', 'DECS ISSUE') # decs issues was majority class (11000+). replace lower number (7000+) with higher number


# grouping Trades (with count of less than 1%) together that need a Specialist to execute the job
# assigning all to existing class 'Specialist'
job_reduced['sor'] = job_reduced['sor'].replace('WOODWORM/DAMP', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('ACCESS/SECURITY', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('TV AERIALS', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('DISABLED ADAPT', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('MEDICAL', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('LIFT/STAIRLIFT', 'SPECIALIST')


# grouping Trades that qualify as 'Wet Trades' together. [Individual count is less than 2% of dataset for each replaced class] 
# assigning all to existing class 'Wet Works'
job_reduced['sor'] = job_reduced['sor'].replace('STONE MASON', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('TILER', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('BRICK WORK', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('GLAZING', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('PLASTERER', 'WET TRADES')

# creating new class 'Exterior Works' and grouping all classes with similar nature together under this class
# [individual count is less than 1% of dataset for each replaced class]
job_reduced['sor'] = job_reduced['sor'].replace('ASPHALTER', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('SCAFFOLDING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('SKIP HIRE', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('ROOFING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('DRAINAGE', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('FENCING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('GROUND WORKS', 'EXTERIOR WORKS')

# creating new class 'Interior Non Utility' and grouping all classes with similar nature together under this class
# [Individual count is less than 0.5% of dataset for each replaced class]
job_reduced['sor'] = job_reduced['sor'].replace('BATH REFURBISH', 'INTERIOR NON UTILITY')
job_reduced['sor'] = job_reduced['sor'].replace('FLOORER', 'INTERIOR NON UTILITY')


# trade Classes are much more balanced than before in terms of count 
print('Trade Class imbalance significantly corrected')

# checking reduced Trade distribution 
job_reduced['sor'].value_counts()


# In[7]:


jr = pd.DataFrame(job_reduced['sor'].value_counts())
jr = jr.reset_index()
jr.columns = ['trade','count']
jr.plot(kind = 'bar', x = 'trade', y = 'count', figsize = (20,10), legend = None, fontsize = 18)
plt.title('Distribution of reduced Trade classes', fontsize = 18)
plt.xticks(fontsize = 16)
plt.xlabel('Trade',fontsize = 15)
plt.ylabel('Overall Repair Count',fontsize = 16)
plt.show()


# #### Calculating number of 'report counts' for each date in the dataframe

# In[9]:


job_counts = job_reduced.groupby(['job_report_date', 'sor']).size().reset_index(name='repair_count')
job_counts = job_counts.iloc[1:]
job_counts


# ##### Adding missing date-trade combinations to job_count dataframe
# For dates where no count for a particular Trade is reported, repair_count = 0

# In[10]:


# converting job_report_date to datetime
job_counts['job_report_date'] = pd.to_datetime(job_counts['job_report_date'])
# date range (the date at which GCH data begins to date it ends)
start_date = pd.to_datetime('1996-02-27')
end_date = pd.to_datetime('2023-06-15')
date_range = pd.date_range(start=start_date, end=end_date)
# new dataframe with all date-trade combinations
trades = list(job_counts['sor'].unique())
date_trade_combinations = []
for date in date_range:
    for sor in trades:
        date_trade_combinations.append({'job_report_date': date, 'sor': sor})

all_combinations_df = pd.DataFrame(date_trade_combinations)
# merging original job_counts frame with all_combinations_df
all_combinations_df['job_report_date'] = pd.to_datetime(all_combinations_df['job_report_date'])
# merging the original job_counts DataFrame with all_combinations_df
all_job_comb = pd.merge(all_combinations_df, job_counts, on=['job_report_date', 'sor'], how='left')
# filling NaN values in repair_count with 0
all_job_comb['repair_count'].fillna(0, inplace=True)
all_job_comb


# In[11]:


all_jobs = all_job_comb[['job_report_date', 'sor', 'repair_count']]

all_jobs['Year'] = pd.to_datetime(all_jobs['job_report_date']).dt.year
all_jobs['Week'] = pd.to_datetime(all_jobs['job_report_date']).dt.week
all_jobs['Month'] = pd.to_datetime(all_jobs['job_report_date']).dt.month
all_jobs['Day'] = pd.to_datetime(all_jobs['job_report_date']).dt.day

all_jobs['WeekDay'] = pd.to_datetime(all_jobs['job_report_date']).dt.dayofweek
all_jobs['Holiday'] = all_jobs['job_report_date'].isin(uk_holidays)
all_jobs['BeginMonth']=all_jobs.Day.isin([1,2,3]).astype(int)
all_jobs['Weekend']=all_jobs.WeekDay.isin([5,6]).astype(int)

all_jobs.head()


# ##### Get weather data

# In[12]:


# getting wmo codes (weather condition)
wmo.drop(0, axis = 1, inplace = True)
wmo.columns = ['description', 'weather condition']
my_weather_map = wmo['weather condition']
gl['weather_condition'] = gl['weathercode (wmo code)'].map(my_weather_map)
# gloucester weather data
gl = gl.reindex(columns=['time', 'weathercode (wmo code)', 'weather_condition','temperature_2m_max (°C)',
                           'temperature_2m_min (°C)', 'temperature_2m_mean (°C)',
                           'apparent_temperature_max (°C)', 'apparent_temperature_min (°C)',
                           'apparent_temperature_mean (°C)', 'shortwave_radiation_sum (MJ/m²)',
                           'precipitation_sum (mm)', 'rain_sum (mm)', 'snowfall_sum (cm)',
                           'precipitation_hours (h)', 'windspeed_10m_max (km/h)',
                           'windgusts_10m_max (km/h)', 'winddirection_10m_dominant (°)', ])


# ##### WMO code = 2 means that the weather at present is the same as last recorded weather (today is same as yesterday since this is daily weather data)
# Replace 2 with previous weather condition

# In[13]:


# first replacing 2 with NaN, then filling NaN in the weather condition column with the value of the preceding row 
modified_weather_codes = gl[['weathercode (wmo code)', 'weather_condition']]
modified_weather_codes[modified_weather_codes['weathercode (wmo code)']==2] = np.NaN
# modified_weather_codes
modified_weather_codes1 = modified_weather_codes.fillna(method='ffill')
modified_weather_codes1.columns = ['weathercode (wmo code) modified','weather_condition modified']
#modified_weather_codes1
gl_modified = pd.concat([gl, modified_weather_codes1], axis = 1)
gl_modified = gl_modified.reindex(columns=['time', 'weathercode (wmo code)', 'weather_condition', 
                                           'weathercode (wmo code) modified', 'weather_condition modified',
                                           'temperature_2m_max (°C)',
                                           'temperature_2m_min (°C)', 'temperature_2m_mean (°C)',
                                           'apparent_temperature_max (°C)', 'apparent_temperature_min (°C)',
                                           'apparent_temperature_mean (°C)', 'shortwave_radiation_sum (MJ/m²)',
                                           'precipitation_sum (mm)', 'rain_sum (mm)', 'snowfall_sum (cm)',
                                           'precipitation_hours (h)', 'windspeed_10m_max (km/h)',
                                           'windgusts_10m_max (km/h)', 'winddirection_10m_dominant (°)', ])


gl_updated = gl_modified[['time', 'weathercode (wmo code) modified', 'weather_condition modified',
           'temperature_2m_max (°C)',
           'temperature_2m_min (°C)', 'temperature_2m_mean (°C)',
           'apparent_temperature_max (°C)', 'apparent_temperature_min (°C)',
           'apparent_temperature_mean (°C)', 'shortwave_radiation_sum (MJ/m²)',
           'precipitation_sum (mm)', 'rain_sum (mm)', 'snowfall_sum (cm)',
           'precipitation_hours (h)', 'windspeed_10m_max (km/h)',
           'windgusts_10m_max (km/h)', 'winddirection_10m_dominant (°)', ]]
gl_updated


# #### Merge weather dataframe to mainframe. 
# Join on common date column and job_report_date to get weather data for every date each job is reported, and also for dates where no job is reported

# In[14]:


# converting DATE values in WEATHER dataset to DATETIME type for easy merging with REPAIR dataset
gl_updated.time = gl_updated.time.apply(lambda x: x.date())
gl_updated['time'] = pd.to_datetime(gl_updated['time'])
job_unique_date_weather = all_jobs.merge(gl_updated, how='inner', left_on='job_report_date', right_on='time')

# dropping numerical weather code and only keeping corresponding textual code to later encode
job_unique_date_weather = job_unique_date_weather.drop('weathercode (wmo code) modified', axis = 1)

# making Boolean value into integer
job_unique_date_weather['Holiday'] = job_unique_date_weather['Holiday'].apply(int) 
job_unique_date_weather


# #### Isolating data for each Trade Class 

# In[15]:


unique_sor_classes = job_unique_date_weather['sor'].unique()
sor_dataframes = {}

# creating separate dataframes for each trade class 
for sor_class in unique_sor_classes:
    sor_dataframes[sor_class] = job_unique_date_weather[job_unique_date_weather['sor'] == sor_class]

    
carpentry = sor_dataframes['CARPENTRY']
electrical = sor_dataframes['ELECTRICAL']
gas_fitter = sor_dataframes['GAS FITTER.']
exterior_works = sor_dataframes['EXTERIOR WORKS']
general = sor_dataframes['GENERAL']
ghs = sor_dataframes['GHS']
metalworker = sor_dataframes['METALWORKER']
plumbing = sor_dataframes['PLUMBING']
decs_issue = sor_dataframes['DECS ISSUE']
interior_non_utility = sor_dataframes['INTERIOR NON UTILITY']
not_specified = sor_dataframes['NOT SPECIFIED']
specialist = sor_dataframes['SPECIALIST']
pre_termination = sor_dataframes['PRE-TERMINATION']
wet_trades = sor_dataframes['WET TRADES']
call_outs = sor_dataframes['CALL OUTS']
fitter = sor_dataframes['FITTER']
decent_homes = sor_dataframes['DECENT HOMES']
inspection = sor_dataframes['INSPECTION']
day_rates = sor_dataframes['DAY RATES']
special_quotes = sor_dataframes['SPECIAL/QUOTES']
esw = sor_dataframes['ESW']
gas_servicing = sor_dataframes['Gas Servicing']
asbestos = sor_dataframes['ASBESTOS']


# ##### Investigate missing values

# In[17]:


for feature in all_jobs.columns.values:
    print('#####', feature, '-----number missing', all_jobs[feature].isnull().sum())
    


# NO MISSING VALUES

# #### Model Implementation 

# ### Experiment - Using Classifiers for predicting demand for each Trade type

# In[18]:


# loading all trade datasets
datasets = {
    'carpentry': carpentry,
    'electrical': electrical,
    'gas_fitter': gas_fitter,
    'exterior_works': exterior_works,
    'general': general,
    'ghs': ghs,
    'metalworker': metalworker,
    'plumbing': plumbing,
    'decs_issue': decs_issue,
    'interior_non_utility': interior_non_utility,
    'not_specified': not_specified,
    'specialist': specialist,
    'pre_termination': pre_termination,
    'wet_trades': wet_trades,
    'call_outs': call_outs,
    'fitter': fitter,
    'decent_homes': decent_homes,
    'inspection': inspection,
    'day_rates': day_rates,
    'special_quotes': special_quotes,
    'esw': esw,
    'gas_servicing': gas_servicing,
    'asbestos': asbestos
}


# declaring classifiers to be used
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'LightGBM': LGBMClassifier(random_state=42)
}

# list for storing performance metrics
all_metrics = []

# iterating over each dataset
for dataset_name, dataset in datasets.items():
    print(f'Training classifiers for dataset: {dataset_name}')
    
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    job_date_weather_target = dataset['repair_count']

    # feature scaling 
    scaler = StandardScaler()
    job_date_weather_predictors_scaled = scaler.fit_transform(job_date_weather_predictors)

    # splitting the data
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors_scaled, job_date_weather_target, test_size=0.2, random_state=42)

    # creating list for saving metrics for each classifier
    dataset_metrics = []

    # iterating over all classifiers, train, predict, and calculate metrics
    for name, classifier in classifiers.items():
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_cv)
        accuracy = accuracy_score(y_cv, y_pred)
        f1 = f1_score(y_cv, y_pred, average='weighted')
        f1_macro = f1_score(y_cv, y_pred, average='macro')
        rmse = np.sqrt(mean_squared_error(y_cv, y_pred))
        recall = recall_score(y_cv, y_pred, average='weighted', zero_division=1)

        dataset_metrics.append({'Classifier': name, 'Accuracy': accuracy, 'F1 Score': f1, 'F1 Macro':f1_macro, 'RMSE': rmse, 'Recall': recall})

    # saving metrics for each classifier
    all_metrics.extend([(dataset_name, metrics) for metrics in dataset_metrics])


# ##### plotting classifier performance

# In[20]:


# creating lists to store performance metrics of all classifiers for all priority typesdatasets = []
datasets = []
classifiers = []
accuracies = []
f1_scores = []
f1_macro = []
rmse_scores = []
recalls = []

# iterating over all_metrics and getting corresponding performance metrics of all classifiers for all priority types
for dataset, metrics in all_metrics:
    datasets.append(dataset)
    classifiers.append(metrics['Classifier'])
    accuracies.append(metrics['Accuracy'])
    f1_scores.append(metrics['F1 Score'])
    f1_macro.append(metrics['F1 Macro'])
    rmse_scores.append(metrics['RMSE'])
    recalls.append(metrics['Recall'])

# creating dataFrame for classifier metrics
metrics_df = pd.DataFrame({
    'Dataset': datasets,
    'Classifier': classifiers,
    'Accuracy': accuracies,
    'F1 Score': f1_scores,
    'F1 Macro': f1_macro,
    'RMSE': rmse_scores,
    'Recall': recalls
})

metrics_df


# In[21]:


# sorting the metrics by accuracy
sorted_metrics_df = metrics_df.sort_values(by='F1 Macro')

# get metrics of each classifier
each_classifier_scores = sorted_metrics_df.groupby('Classifier')

# plotting accuracy and F1 score for each type of classifier (trained on each trade dataset)
for classifier, classifier_scores in each_classifier_scores:
    plt.figure(figsize=(10, 6))
    plt.title(f'{classifier} - Accuracy and F1 Macro Score')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.xticks(rotation=65)

    plt.plot(classifier_scores['Dataset'], classifier_scores['Accuracy'], label='Accuracy', marker='o')
    plt.plot(classifier_scores['Dataset'], classifier_scores['F1 Macro'], label='F1 Macro', marker='o')

    plt.legend()
    plt.tight_layout()
    plt.show()


# ##### Identify best performing classifier

# In[24]:


# grouping the dataframe by 'Classifier' and calculating mean accuracy for each classifier
classifier_accuracy = metrics_df.groupby('Classifier')['Accuracy'].mean()
classifier_f1_macro = metrics_df.groupby('Classifier')['F1 Macro'].mean()

# classifier with the highest mean accuracy
best_classifier = classifier_accuracy.idxmax()
highest_accuracy = classifier_accuracy.max()
highest_f1_macro = classifier_f1_macro.max()


print(f"The best performing classifier is '{best_classifier}' with an average accuracy of {highest_accuracy:.6f} and highest F1 Macro Score of {highest_f1_macro:.6f}")


# In[25]:


metrics_df[metrics_df['Classifier']=='CatBoost']


# #### Clasiifier on oversampled data for minority classes

# In[26]:


# datasets
datasets = {
    'carpentry': carpentry,
    'electrical': electrical,
    'gas_fitter': gas_fitter,
    'exterior_works': exterior_works,
    #'general': general,
    'ghs': ghs,
    'metalworker': metalworker,
    'plumbing': plumbing,
    'decs_issue': decs_issue,
    'interior_non_utility': interior_non_utility,
    'not_specified': not_specified,
    'specialist': specialist,
    'pre_termination': pre_termination,
    'wet_trades': wet_trades,
    'call_outs': call_outs,
    'fitter': fitter,
    #'decent_homes': decent_homes,
    'inspection': inspection,
    'day_rates': day_rates,
    'special_quotes': special_quotes,
    'esw': esw,
    'gas_servicing': gas_servicing,
    'asbestos': asbestos
}


classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'LightGBM': LGBMClassifier(random_state=42)
}


all_metrics_smote = []

# iterating over each dataset
for dataset_name, dataset in datasets.items():
    
    print(f'Training classifiers for: {dataset_name}')
    

    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    job_date_weather_target = dataset['repair_count']
    
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)
    
    # dropping repair count values which have instances lower than minimum number of neigbours needed to create synthetic datapoints using SMOTE
    repair_count_counts = y_train.value_counts()
    values_to_drop = repair_count_counts[repair_count_counts < 10].index.tolist()
    x_train_filtered = x_train[~y_train.isin(values_to_drop)]
    y_train_filtered = y_train[~y_train.isin(values_to_drop)]
    
    # resampling training data using SMOTE
    oversample = SMOTE()
    resampled_X, resampled_y = oversample.fit_resample(x_train_filtered, y_train_filtered) 

    # list to store metrics for each classifier for current dataset (with resampled data)
    dataset_metrics = []

    # iterating over classifiers, train, predict, and calculating metrics
    for name, classifier in classifiers.items():
        classifier.fit(resampled_X, resampled_y)
        y_pred = classifier.predict(x_cv)
        accuracy = accuracy_score(y_cv, y_pred)
        f1 = f1_score(y_cv, y_pred, average='weighted')
        f1_macro = f1_score(y_cv, y_pred, average='macro')
        rmse = np.sqrt(mean_squared_error(y_cv, y_pred))
        recall = recall_score(y_cv, y_pred, average='weighted', zero_division=1)

        dataset_metrics.append({'Classifier': name, 'Accuracy': accuracy, 'F1 Score': f1, 'F1 Macro':f1_macro, 'RMSE': rmse, 'Recall': recall})

    # storing metrics for current dataset
    all_metrics_smote.extend([(dataset_name, metrics) for metrics in dataset_metrics])


# In[28]:


# storing performance metrics of each classifier for each oversampled priority type dataset
datasets = []
classifiers = []
accuracies = []
f1_scores = []
f1_macro = []
rmse_scores = []
recalls = []

for dataset, metrics in all_metrics_smote:
    datasets.append(dataset)
    classifiers.append(metrics['Classifier'])
    accuracies.append(metrics['Accuracy'])
    f1_scores.append(metrics['F1 Score'])
    f1_macro.append(metrics['F1 Macro'])
    rmse_scores.append(metrics['RMSE'])
    recalls.append(metrics['Recall'])

metrics_smote_df = pd.DataFrame({
    'Dataset': datasets,
    'Classifier': classifiers,
    'Accuracy': accuracies,
    'F1 Score': f1_scores,
    'F1 Macro': f1_macro,
    'RMSE': rmse_scores,
    'Recall': recalls
})

metrics_smote_df


# In[29]:


# identifying best classifier (after SMOTEing)
classifier_accuracy = metrics_smote_df.groupby('Classifier')['Accuracy'].mean()
classifier_f1_macro = metrics_smote_df.groupby('Classifier')['F1 Macro'].mean()

# Find the classifier with the highest mean accuracy
best_classifier = classifier_accuracy.idxmax()
highest_accuracy = classifier_accuracy.max()
highest_f1_macro = classifier_f1_macro.max()

# finding classifier with the highest mean accuracy
print(f"The best performing classifier after oversampling with SMOTE is '{best_classifier}' with an average accuracy of {highest_accuracy:.6f} and highest F1 Macro Score of {highest_f1_macro:.6f}")


# In[33]:


best_resampled_classifier_scores = metrics_smote_df[metrics_smote_df['Classifier'] == 'Random Forest']
best_resampled_classifier_scores = best_resampled_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
best_resampled_classifier_scores


# In[34]:


random_forest_classifier_scores = metrics_smote_df[metrics_smote_df['Classifier'] == 'Random Forest']
random_forest_classifier_scores = random_forest_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
plt.figure(figsize=(10, 6))
plt.title(f'Random Forest Classifier on Resampled data')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.plot(random_forest_classifier_scores['Dataset'], random_forest_classifier_scores['Accuracy'], label='Accuracy', marker='o')
plt.plot(random_forest_classifier_scores['Dataset'], random_forest_classifier_scores['F1 Macro'], label='F1 Macro', marker='o')

plt.legend()
plt.tight_layout()
plt.show()


# #### No significant improvement with oversampling

# #### True vs Predicted plots on best classifier 
# non resampled data

# In[35]:


datasets = {
    'carpentry': carpentry,
    'electrical': electrical,
    'gas_fitter': gas_fitter,
    'exterior_works': exterior_works,
    'general': general,
    'ghs': ghs,
    'metalworker': metalworker,
    'plumbing': plumbing,
    'decs_issue': decs_issue,
    'interior_non_utility': interior_non_utility,
    'not_specified': not_specified,
    'specialist': specialist,
    'pre_termination': pre_termination,
    'wet_trades': wet_trades,
    'call_outs': call_outs,
    'fitter': fitter,
    'decent_homes': decent_homes,
    'inspection': inspection,
    'day_rates': day_rates,
    'special_quotes': special_quotes,
    'esw': esw,
    'gas_servicing': gas_servicing,
    'asbestos': asbestos
}


predictions_dict = {}

# iterating over each dataset
for dataset_name, dataset in datasets.items():
    
    print(f'Retraining CatBoost Classifier for dataset: {dataset_name}')
    

    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    job_date_weather_target = dataset['repair_count']

    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)
    
    # retraining CatBoost classifier
    catboost = CatBoostClassifier(silent = True)
    catboost.fit(x_train, y_train)
    
    # making predictions on catboost classifier
    y_pred = catboost.predict(x_cv)
    
    # saving predicted values and true values to later plot
    predictions_dict[dataset_name] = {'Actual': y_cv.reset_index(drop = True), 'Predicted': y_pred}


# In[36]:


# check classification report for each classiifer (trained on each trade dataset)
# iterating over each dataset
for dataset_name, predictions in predictions_dict.items():
    actual_values = predictions['Actual']
    predicted_values = predictions['Predicted']
    
    # generating classification report
    cls_report = classification_report(actual_values, predicted_values,  zero_division='warn')
    
    print(f'Classification Report for dataset - {dataset_name}:\n')
    print(cls_report)


# ### Experiment - Using Regressors for predicting demand for each Trade type

# #### Training regressors on non-transformed data

# In[38]:


# datasets for regression (before target log transformed)
datasets = [carpentry, electrical, gas_fitter, exterior_works, general, ghs, 
           metalworker, plumbing, decs_issue, interior_non_utility, not_specified, 
           specialist, pre_termination, wet_trades, call_outs, fitter, decent_homes,
           inspection, day_rates, special_quotes, esw, gas_servicing, asbestos]



# List of regressors to be experimented with for predicting repair demand
regressors = [
    LinearRegression,
    KNeighborsRegressor,
    RandomForestRegressor,
    Lasso,
    ElasticNet,
    DecisionTreeRegressor,
    GradientBoostingRegressor,  
    LGBMRegressor,
    CatBoostRegressor
]

# list of metrics to calculate for evaluating regressors
metrics = {
    'MSE': mean_squared_error,
    'R2': r2_score,
}

rows = []

# creating dataFrame to store scores
scores_df = pd.DataFrame(columns=['Method', 'Dataset'] + list(metrics.keys()))

# iterating over regressors
for regressor in regressors:
    for dataset_name, dataset in zip(['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                                      'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                                      'specialist', 'pre_termination', 'wet_trades', 'call_outs','fitter', 
                                      'decent_homes', 'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 
                                      'asbestos'], datasets):
        
        print(f'Training {regressor.__name__} on {dataset_name}')

        # prediciting features and target
        job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified','temperature_2m_max (°C)','temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
        job_date_weather_target = dataset['repair_count']
        x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

        # one-hot encoding categorical columns
        s = (x_train.dtypes == 'object')
        object_cols = list(s[s].index)

        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
        OH_cols_cv = pd.DataFrame(OH_encoder.transform(x_cv[object_cols]))

        OH_cols_train.index = x_train.index
        OH_cols_cv.index = x_cv.index

        num_X_train = x_train.drop(object_cols, axis=1)
        num_X_cv = x_cv.drop(object_cols, axis=1)

        OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
        OH_X_cv = pd.concat([num_X_cv, OH_cols_cv], axis=1)

        # training regressors on encoded data and calculating metrics
        metrics_scores = []
        for metric_name, metric_func in metrics.items():
            model = regressor()
            model.fit(OH_X_train, y_train)
            y_pred = model.predict(OH_X_cv)
            metric_score = metric_func(y_cv, y_pred)
            metrics_scores.append(metric_score)
        # storing metrics of all regressors for current dataset
        row_data = {'Method': regressor.__name__, 'Dataset': dataset_name, **dict(zip(list(metrics.keys()), metrics_scores))}
        rows.append(row_data)
        
# dataframe for storing evalaution metric scores for each regressor
scores_df = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)     


# In[40]:


# getting best regressor model trained on non-transformed data
regressor_r2 = scores_df.groupby('Method')['R2'].mean()
# regressor with highest R2 score
best_regressor = regressor_r2.idxmax()
highest_r2 = regressor_r2.max()

print(f"The best performing Regressor trained on non transformed data is '{best_regressor}' with an average R2 score of {highest_r2:.6f}")


# In[43]:


best_regressor_scores = scores_df[scores_df['Method'] == 'LGBMRegressor']
best_regressor_scores = best_regressor_scores.sort_values(by = 'R2', ascending = False)
mse_mean = best_regressor_scores['MSE'].mean()
print(f"The best performing Regressor is '{best_regressor}' with an average MSE score of {mse_mean:.6f}")
best_regressor_scores


# In[45]:


# sorting lgbm scores
best_regressor_scores_sorted = best_regressor_scores.sort_values(by = 'R2')
# plotting R2 squared score for each type of priority
plt.figure(figsize=(10, 6))
plt.title(f'LGBM Regressor - R2')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.plot(best_regressor_scores_sorted['Dataset'], best_regressor_scores_sorted['R2'], label='R2', marker='o')
plt.legend()
plt.show()


# #### True vs Predicted Values (regression), and residual plots for all Trades

# In[48]:


datasets = [carpentry, electrical, gas_fitter, exterior_works, general, ghs, metalworker, plumbing, decs_issue, 
            interior_non_utility, not_specified, specialist, pre_termination, wet_trades, call_outs, fitter, 
            decent_homes, inspection, day_rates, special_quotes, esw, gas_servicing, asbestos]

dataset_names = ['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                 'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                 'specialist', 'pre_termination', 'wet_trades', 'call_outs', 'fitter', 'decent_homes', 
                 'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 'asbestos']

best_regressor = LGBMRegressor(silent = True)

# Loop through datasets
for dataset_name, dataset in zip(dataset_names, datasets):
    print(f'plots for {dataset_name}')
    
    # prepare predictors
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor',
                                                    'weather_condition modified', 'apparent_temperature_min (°C)',
                                                    'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)',
                                                    'winddirection_10m_dominant (°)'], axis=1)
    # target - repair count
    job_date_weather_target = dataset['repair_count']

    # train test split
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

    # fit best regressor
    best_regressor.fit(x_train, y_train)

    # predictions
    pred = best_regressor.predict(x_cv)
    compare = pd.DataFrame({'y_cv': y_cv, 'pred': pred})
    
    # figure with two subplots for true vs predicted plot, and residual plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    # plot 1: actual vs predicted
    ax1 = axes[0]
    ax1.scatter(compare['y_cv'], compare['pred'], color='blue', label='Predicted vs. True')
    ax1.plot(np.linspace(0, max(compare['y_cv']), 100), np.linspace(0, max(compare['y_cv']), 100), color='red', label='Ideal')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Actual vs. Predicted - {dataset_name.upper()}')
    ax1.legend()
    
    # plot 2: predicted vs residual
    ax2 = axes[1]
    residuals = compare['pred'] - compare['y_cv']
    ax2.scatter(compare['pred'], residuals, color='blue')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Predicted vs. Residuals - {dataset_name.upper()}')
    
    # adjust layout and display plots
    plt.tight_layout()
    display(fig)
    plt.close()


# #### Creating copies of datasets for each Trade type for regression, and applying log-transformation on repair counts (target)
# Transforming target 'repair_count' for better regression over extreme values/ outlier values of repair count
# 

# In[49]:


# preparing data for regression. Log tranforming target
datasets = [carpentry, electrical, gas_fitter, exterior_works, general, ghs, metalworker, plumbing, decs_issue, 
            interior_non_utility, not_specified, specialist, pre_termination, wet_trades, call_outs, fitter, 
            decent_homes, inspection, day_rates, special_quotes, esw, gas_servicing, asbestos]

# dataset names - trades
dataset_names = ['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                 'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                 'specialist', 'pre_termination', 'wet_trades', 'call_outs', 'fitter', 'decent_homes', 
                 'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 'asbestos']

# creating copies of each dataset
for name, dataset in zip(dataset_names, datasets):
    globals()[f'{name}_reg'] = dataset.copy()

# applying log transformation to the repair_count column of each copied dataset
for name in dataset_names:
    dataset = globals()[f'{name}_reg']
    dataset['repair_count'] = np.log1p(dataset['repair_count'])


# In[50]:


# datasets for regression (after target log transformed)
datasets_for_regression = [carpentry_reg, electrical_reg, gas_fitter_reg, exterior_works_reg, general_reg, ghs_reg, 
                           metalworker_reg, plumbing_reg, decs_issue_reg, interior_non_utility_reg, not_specified_reg, 
                           specialist_reg, pre_termination_reg, wet_trades_reg, call_outs_reg, fitter_reg, decent_homes_reg,
                           inspection_reg, day_rates_reg, special_quotes_reg, esw_reg, gas_servicing_reg, asbestos_reg]



# List of regressors to be experimented with for predicting repair demand
regressors = [
    LinearRegression,
    KNeighborsRegressor,
    RandomForestRegressor,
    Lasso,
    ElasticNet,
    DecisionTreeRegressor,
    GradientBoostingRegressor,  
    LGBMRegressor,
    CatBoostRegressor
]

# list of metrics to calculate for evaluating regressors
metrics = {
    'MSE': mean_squared_error,
    'R2': r2_score,
}

rows = []

# creating dataFrame to store scores
scores_df = pd.DataFrame(columns=['Method', 'Dataset'] + list(metrics.keys()))

# iterating over regressors
for regressor in regressors:
    for dataset_name, dataset in zip(['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                                      'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                                      'specialist', 'pre_termination', 'wet_trades', 'call_outs','fitter', 
                                      'decent_homes', 'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 
                                      'asbestos'], datasets_for_regression):
        
        print(f'Training {regressor.__name__} on {dataset_name}')

        # prediciting features and target
        job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified','temperature_2m_max (°C)','temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
        job_date_weather_target = dataset['repair_count']
        x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

        # one-hot encoding categorical columns
        s = (x_train.dtypes == 'object')
        object_cols = list(s[s].index)

        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
        OH_cols_cv = pd.DataFrame(OH_encoder.transform(x_cv[object_cols]))

        OH_cols_train.index = x_train.index
        OH_cols_cv.index = x_cv.index

        num_X_train = x_train.drop(object_cols, axis=1)
        num_X_cv = x_cv.drop(object_cols, axis=1)

        OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
        OH_X_cv = pd.concat([num_X_cv, OH_cols_cv], axis=1)

        # training regressors on encoded data and calculating metrics
        metrics_scores = []
        for metric_name, metric_func in metrics.items():
            model = regressor()
            model.fit(OH_X_train, y_train)
            y_pred = model.predict(OH_X_cv)
            metric_score = metric_func(y_cv, y_pred)
            metrics_scores.append(metric_score)
        # storing metrics of all regressors for current dataset
        row_data = {'Method': regressor.__name__, 'Dataset': dataset_name, **dict(zip(list(metrics.keys()), metrics_scores))}
        rows.append(row_data)
        
# dataframe for storing evalaution metric scores for each regressor
scores_df = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)     


# In[ ]:


# scores_df


# In[52]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.table(cellText=scores_df.values, colLabels=scores_df.columns, cellLoc='center', loc='center')
plt.savefig('scores_table.png', bbox_inches='tight', pad_inches=0.5)
plt.show()


# In[53]:


# sorting scores in order of decreasing R-squared
sorted_scores_df = scores_df.sort_values(by='R2')
# getting scores for each Classifier
all_regressor_scores = sorted_scores_df.groupby('Method')

# plotting R2 score of all regressors for each priority type
for regressor, regressor_scores in all_regressor_scores:
    plt.figure(figsize=(10, 6))
    plt.title(f'{regressor} - R2')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.xticks(rotation=45)

    plt.plot(regressor_scores['Dataset'], regressor_scores['R2'], label='R2', marker='o')

    plt.legend()
    plt.tight_layout()
    plt.show()


# ##### Identify best regressor

# In[55]:


# getting best regressor
regressor_r2 = scores_df.groupby('Method')['R2'].mean()
# regressor with highest R2 score
best_regressor = regressor_r2.idxmax()
highest_r2 = regressor_r2.max()

print(f"The best performing Regressor is '{best_regressor}' with an average R2 score of {highest_r2:.6f}")


# In[56]:


best_regressor_scores = scores_df[scores_df['Method'] == 'CatBoostRegressor']
best_regressor_scores = best_regressor_scores.sort_values(by = 'R2', ascending = False)
mse_mean = best_regressor_scores['MSE'].mean()
print(f"The best performing Regressor is '{best_regressor}' with an average MSE score of {mse_mean:.6f}")
best_regressor_scores


# In[57]:


# sorting catboost scores
catboost_scores_sorted = best_regressor_scores.sort_values(by = 'R2')
# plotting R2 squared score for each type of priority
plt.figure(figsize=(10, 6))
plt.title(f'LGBM Regressor - R2')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.plot(catboost_scores_sorted['Dataset'], catboost_scores_sorted['R2'], label='R2', marker='o')
plt.legend()
plt.show()


# ##### Residual Analysis for log transformed trade datasets

# In[58]:


datasets_for_regression = [carpentry_reg, electrical_reg, gas_fitter_reg, exterior_works_reg, general_reg, ghs_reg, 
                           metalworker_reg, plumbing_reg, decs_issue_reg, interior_non_utility_reg, not_specified_reg, 
                           specialist_reg, pre_termination_reg, wet_trades_reg, call_outs_reg, fitter_reg, decent_homes_reg,
                           inspection_reg, day_rates_reg, special_quotes_reg, esw_reg, gas_servicing_reg, asbestos_reg]


best_regressor = CatBoostRegressor(silent = True)


# Loop through datasets
for dataset_name, dataset in zip(dataset_names, datasets_for_regression):
    print(f'plots for {dataset_name}')
    
    # prepare predictors
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor',
                                                    'weather_condition modified', 'apparent_temperature_min (°C)',
                                                    'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)',
                                                    'winddirection_10m_dominant (°)'], axis=1)
    # target - repair count
    job_date_weather_target = dataset['repair_count']

    # train test split
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

    # fit best regressor
    best_regressor.fit(x_train, y_train)

    # predictions
    pred = best_regressor.predict(x_cv)
    compare = pd.DataFrame({'y_cv': y_cv, 'pred': pred})
    
    # figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    # plot 1: actual vs predicted
    ax1 = axes[0]
    ax1.scatter(compare['y_cv'], compare['pred'], color='blue', label='Predicted vs. True')
    ax1.plot(np.linspace(0, max(compare['y_cv']), 100), np.linspace(0, max(compare['y_cv']), 100), color='red', label='Ideal')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Actual vs. Predicted - {dataset_name.upper()}')
    ax1.legend()
    
    # plot 2: actual vs predicted
    ax2 = axes[1]
    residuals = compare['pred'] - compare['y_cv']
    ax2.scatter(compare['pred'], residuals, color='blue')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Predicted vs. Residuals - {dataset_name.upper()}')
    
    # adjusting layout and displaying the plots
    plt.tight_layout()
    display(fig)
    plt.close()


# #### Conclusion: Log transforming target varaible improves model performance

# #### Tuning regressors
# trained on log-transformed data

# In[62]:



datasets = [carpentry_reg, electrical_reg, gas_fitter_reg, exterior_works_reg, general_reg, ghs_reg, 
           metalworker_reg, plumbing_reg, decs_issue_reg, interior_non_utility_reg, not_specified_reg, 
           specialist_reg, pre_termination_reg, wet_trades_reg, call_outs_reg, fitter_reg, decent_homes_reg,
           inspection_reg, day_rates_reg, special_quotes_reg, esw_reg, gas_servicing_reg, asbestos_reg]

predictions_dict_tuned_regressor = {}


# defining parameter grid for tuning
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

# creating instance of best regressor - CatBoost Regressor
catboost_regressor = CatBoostRegressor(random_state=42, silent = True)

# creating instance of GridSearchCV
grid_search = GridSearchCV(catboost_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')

# iterating over each dataset
for dataset_name, dataset in zip(['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                             'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                             'specialist', 'pre_termination', 'wet_trades', 'call_outs', 'fitter', 'decent_homes', 
                             'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 'asbestos'], datasets):
    
    print(f'Tuning CatBoost Regressor on {dataset_name}')
    
    # splitting dataset into features and target
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    job_date_weather_target = dataset['repair_count']
    
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

    # one-hot encoding categorical columns (same as before)
    s = (x_train.dtypes == 'object')
    object_cols = list(s[s].index)

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
    OH_cols_cv = pd.DataFrame(OH_encoder.transform(x_cv[object_cols]))

    OH_cols_train.index = x_train.index
    OH_cols_cv.index = x_cv.index

    num_X_train = x_train.drop(object_cols, axis=1)
    num_X_cv = x_cv.drop(object_cols, axis=1)

    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_cv = pd.concat([num_X_cv, OH_cols_cv], axis=1)
    
    # performing grid search for hyperparameter tuning
    grid_search.fit(OH_X_train, y_train)
    
    # getting best parameters after tuning
    best_params = grid_search.best_params_
    
    # using best parameters to retrain the LGBM Regressor
    best_cb_regressor = CatBoostRegressor(**best_params, random_state=42, silent = True)
    best_cb_regressor.fit(OH_X_train, y_train)
    
    # making predictions on test data with model with best paramters
    y_pred = best_cb_regressor.predict(OH_X_cv)
    y_pred_real = np.expm1(y_pred)
    rounded_pred = np.round(y_pred_real).astype(int)
    rounded_pred[rounded_pred < 0] = 0

    # inverse log transformation true values
    y_cv_real = np.expm1(y_cv)
    
    # Calculate R2 and MSE scores
    r2_score_val = r2_score(y_cv, y_pred)
    mse_score_val = mean_squared_error(y_cv, y_pred)
    
    # saving best parameters, R2, and MSE scores in a dictionary
    best_results = {
        'Best Parameters': best_params,
        'R2 Score': r2_score_val,
        'MSE Score': mse_score_val
    }
    
    predictions_dict_tuned_regressor[dataset_name] = {'Actual': y_cv_real.reset_index(drop = True), 'Predicted': rounded_pred, 'Best Results': best_results}

print('Tuning complete')


# In[64]:


scores_after_tuning = {}

# iterating over each dataset
for dataset_name, dataset in zip(['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                             'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                             'specialist', 'pre_termination', 'wet_trades', 'call_outs', 'fitter', 'decent_homes', 
                             'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 'asbestos'], datasets):
    
    # getting prediction results from predictions dictioanary for the current dataset
    dataset_dict = predictions_dict_tuned_regressor[dataset_name]
    
    
    # getting best results
    best_results = dataset_dict['Best Results']
    r2_score_val = best_results['R2 Score']
    mse_score_val = best_results['MSE Score']
    
    # dataframe for storing best scores after tuning
    scores_tuned = pd.DataFrame({
        'Dataset': [dataset_name],
        'R2 Score': [r2_score_val],
        'MSE Score': [mse_score_val]
    })
    scores_after_tuning[dataset_name] = scores_tuned

scores_tuned = pd.concat(scores_after_tuning, ignore_index=True)

# scores after hyperparameter tuning
display(scores_tuned)


# In[65]:


scores_tuned.sort_values(by = 'R2 Score', ascending  = False)


# In[66]:


scores_tuned['R2 Score'].mean()


# In[67]:


scores_tuned['MSE Score'].mean()


# #### Conclusion: No improvement in MSE after tuning. Very slight improvement in R2 after tuning
# 

# #### Plot regressor and classifer performance on all TRADE classes
# R2 for regressors and F1 Macro for classifiers

# In[68]:


scores_df = scores_df[~scores_df['Dataset'].isin(['decent_homes', 'decs_issue'])]
metrics_df = metrics_df[~metrics_df['Dataset'].isin(['decent_homes', 'decs_issue'])]

# scores for best models - CatBoostRegressor and CatBoostClassifier
catboost_regressor_scores = scores_df[scores_df['Method'] == 'CatBoostRegressor']
catboost_classifier_scores = metrics_df[metrics_df['Classifier'] == 'CatBoost']

# filter scores greater than 0
catboost_regressor_scores = catboost_regressor_scores[catboost_regressor_scores['R2'] > 0]
catboost_classifier_scores = catboost_classifier_scores[catboost_classifier_scores['Accuracy'] > 0]

dataset_order_reg = catboost_regressor_scores.sort_values(by='R2')
dataset_order_cls = catboost_classifier_scores.sort_values(by='F1 Macro', ascending=False)

# reorder dataset_order_cls based on the order of dataset_order_reg
dataset_order_cls_reordered = dataset_order_cls.set_index('Dataset').loc[dataset_order_reg['Dataset']].reset_index()

plt.figure(figsize=(10, 6))

plt.title('Scores - CatBoost Regressor & CatBoost Classifier')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=65)

plt.plot(dataset_order_reg['Dataset'], dataset_order_reg['R2'], label='R2 (CatBoost Regressor)', marker='o')
plt.plot(dataset_order_cls_reordered['Dataset'], dataset_order_cls_reordered['F1 Macro'], label='F1 Macro (CatBoost Classifier)', marker='o')


plt.legend()
plt.tight_layout()
plt.show()


# #### Choosing and saving Best Models

# In[70]:


# not using classifier as a model to predict repair demand for trade types
best_regressor = CatBoostRegressor()

#best_classifier = CatBoostClassifier()


# #### Re-train best regressor on all trade datasets, store for SHAP analysis
# Get predicted values for all datasets to compare true vs predicted values

# In[71]:


# log transformed datasets
datasets = [carpentry_reg, electrical_reg, gas_fitter_reg, exterior_works_reg, general_reg, ghs_reg, 
           metalworker_reg, plumbing_reg, decs_issue_reg, not_specified_reg, 
           specialist_reg, pre_termination_reg, wet_trades_reg, call_outs_reg, fitter_reg, 
           inspection_reg, special_quotes_reg, esw_reg, gas_servicing_reg, asbestos_reg]

# List to store trained models
stored_models = []

# best regressor initialization
best_regressor = CatBoostRegressor(silent = True)

# list to store predictions and real values for all regressors
regressor_comparison_dataframes = []

# iterating over all trade datasets
for dataset_name, dataset in zip(['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 
                             'ghs', 'metalworker', 'plumbing', 'decs_issue', 'not_specified', 
                             'specialist', 'pre_termination', 'wet_trades', 'call_outs', 'fitter', 
                             'inspection', 'special_quotes', 'esw', 'gas_servicing', 'asbestos'], datasets):
    
    print(f'Training best regression model on {dataset_name}')

    # features and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified','temperature_2m_max (°C)','temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)

    # one-hot encoding categorical columns
    s = (x_train.dtypes == 'object')
    object_cols = list(s[s].index)

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
    OH_cols_cv = pd.DataFrame(OH_encoder.transform(x_cv[object_cols]))

    OH_cols_train.index = x_train.index
    OH_cols_cv.index = x_cv.index

    num_X_train = x_train.drop(object_cols, axis=1)
    num_X_cv = x_cv.drop(object_cols, axis=1)

    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_cv = pd.concat([num_X_cv, OH_cols_cv], axis=1)

    current_model = best_regressor
    current_model.fit(OH_X_train, y_train)
    
    # storing trained model
    stored_models.append((dataset_name, current_model))  
    y_pred = current_model.predict(x_cv)
    
    # inverse log transformation of predictions 
    y_pred_real = np.expm1(y_pred)
    rounded_pred = np.round(y_pred_real).astype(int)
    rounded_pred[rounded_pred < 0] = 0

    # inverse log transformation of actual values
    y_cv_real = np.expm1(y_cv)
    
    # comparing predicted values with true values
    compare = {'y_cv': y_cv_real.reset_index(drop = True), 'pred': rounded_pred}
    df = pd.DataFrame(compare)
    
    # adding comparison dataframe (for current dataset) to the list (for all datasets)
    regressor_comparison_dataframes.append(df)

print('All models trained')

# true and predicted values for all regressors (trained on all trade datasets)
for idx, df in enumerate(regressor_comparison_dataframes):
    print(f"Predictions for regressor dataset - '{['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 'metalworker', 'plumbing', 'decs_issue', 'not_specified', 'specialist', 'pre_termination', 'wet_trades', 'call_outs', 'fitter', 'inspection', 'special_quotes', 'esw', 'gas_servicing', 'asbestos'][idx]}':")
    display(df)
    print('\n')


# In[72]:


stored_models


# In[76]:


# iterating over the stored models and datasets 
for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'SHAP for {dataset_name}')
    
    # features and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # training our best regressor - CatBoostRegressor model to get feature importances
    model.fit(x_train, y_train)
    feature_importances = model.feature_importances_
    feature_names = x_train.columns

    # creating SHAP explainer
    explainer = shap.Explainer(model)
    
    # creating explanation data using test data
    explain_data = x_cv  
    
    # geting SHAP values
    shap_values = explainer(explain_data)
    
    # giving title to plot
    plt.gca().add_artist(plt.text(0.5, 1.08, f"Feature Importance plot for model trained on dataset - '{dataset_name}'", ha='center', va='center', transform=plt.gca().transAxes))
    # creating bar plot for feature importances
    shap.plots.bar(shap_values)


# In[77]:


# waterfall plots

for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'waterfall plot for {dataset_name}')
    
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    explainer = shap.Explainer(model)
    data_for_explain = x_cv      
    shap_values = explainer(data_for_explain)
    
    # creating waterfall plot for first datapoint (in each trade dataset)
    index_to_plot = 0
    
    # giving title to the plot for each trade 
    plt.gca().add_artist(plt.text(0.5, 1.08, f"Waterfall plot for model trained on dataset - '{dataset_name}'", ha='center', va='center', transform=plt.gca().transAxes))
    shap.plots.waterfall(shap_values[index_to_plot], max_display=10)  

    


# In[78]:


for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'summary plot for {dataset_name}')
    # features and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    explainer = shap.Explainer(model)
    data_for_explain = x_cv  
    shap_values = explainer(data_for_explain)
    
    # summary plot title for each trade dataset
    plt.gca().add_artist(plt.text(0.5, 1.08, f"Summary plot for model trained on dataset - '{dataset_name}'", ha='center', va='center', transform=plt.gca().transAxes))
    shap.summary_plot(shap_values)


# In[79]:


# force plots
for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'Force plot for {dataset_name}')
    
    # features and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)    
    explainer = shap.TreeExplainer(model)
    data_for_explain = x_cv  
    # force plots based on SHAP values for first 50 points in each dataset
    shap_values = explainer.shap_values(x_train.iloc[0:50, :])  
    shap_plot = shap.force_plot(explainer.expected_value, shap_values, x_train.iloc[0:50, :])
    shap.save_html(f'force_plot_{dataset_name}_obj1.html', shap_plot )


# #### Saving Models

# In[84]:


# initialising best regressor 
best_regressor = CatBoostRegressor(silent = True)
#best_classifier= CatBoostClassifier(silent = True)

# datasets for regression are log transformed (target)
datasets = [carpentry_reg, electrical_reg, gas_fitter_reg, exterior_works_reg, general_reg, ghs_reg, metalworker_reg, 
            plumbing_reg, decs_issue_reg, not_specified_reg, pre_termination_reg, wet_trades_reg, call_outs_reg, 
            fitter_reg, decent_homes_reg, inspection_reg, special_quotes_reg, esw_reg, gas_servicing_reg, 
            interior_non_utility_reg, day_rates_reg, asbestos_reg, specialist_reg]

# not using classifiers as final models for any dataset (initial experiments inlvolved saving these for minority trade classes)
regressors = [best_regressor] * 23 + [None] * 0
classifiers = [None] * 23 + [best_classifier] * 0

# dataset names
dataset_names = ['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 'metalworker', 'plumbing',
                'decs_issue', 'not_specified', 'pre_termination','wet_trades', 'call_outs', 'fitter', 'decent_homes', 
                 'inspection', 'special_quotes', 'esw', 'gas_servicing','interior_non_utility','day_rates', 
                 'asbestos', 'specialist']


# looping over datasets and models
for dataset, classifier, regressor, dataset_name in zip(datasets, classifiers, regressors, dataset_names):
    print(f'Saving dataset: {dataset_name}')
    
    if classifier:
        model_type = 'classifier'
    elif regressor:
        model_type = 'regressor'
    else:
        continue
    
    # cloning classifier or regressor
    trained_model = clone(classifier) if classifier else clone(regressor)
    
    # preparing predictors and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'sor', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    
    # splitting data for retraining
    x_train, _, y_train, _ = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # fitting best regressor model on training data for each dataset
    trained_model.fit(x_train, y_train)
    
    # naming model according to each trade dataset it was trained on 
    model_name = f'{model_type}_{dataset_name}_model_sor_wt'
    
    # saving trained model using joblib
    model_filename = f'{model_name}.joblib'
    joblib.dump(trained_model, model_filename)
    
    print(f'{model_name} trained')
    
print('All models saved')


# ##### The trained models are saved as separate joblib files in the same directory where this Jupyter Notebook is located. These models are later loaded for use in a separate notebook for creating the Dash based web-application. (Please see 'Dash GCH App Final')

# In[ ]:




