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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, recall_score
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.base import clone
import shap
import joblib


# #### Read all necessary data files

# In[2]:


job = pd.read_excel(r'..\UPDATED DATA\04. Repairs\Job.xlsx')
pty_codes = pd.read_excel(r'..\UPDATED DATA\04. Repairs\Pty.xlsx')
sor = pd.read_excel(r'..\UPDATED DATA\04. Repairs\SORTrd.xlsx')
properties = pd.read_excel(r'..\UPDATED DATA\01. Property\Pr.xlsx')
property_class = pd.read_excel(r'..\UPDATED DATA\01. Property\PrCls.xlsx')
property_style = pd.read_excel(r'..\UPDATED DATA\01. Property\APrStyle.xlsx')
property_street_add = pd.read_excel(r'..\UPDATED DATA\01. Property\Str.xlsx')
property_size = pd.read_excel(r'..\UPDATED DATA\01. Property\PrTypeSize.xlsx')


# ### Preparing Data

# In[3]:


# getting uk holiday dates (for holiday varaible needed to execute this project objective - one of the potential predictors)
uk_holidays = holidays.UnitedKingdom()


# ##### Cleaning Job frame
# ##### Merging job with priority type and replacing old SOR Trade codes with new ones

# In[4]:


pty_type = pty_codes[['pty_cde','pty_type']]
job['job_report_date'] = pd.to_datetime(job['reported-dat']).dt.date
job = job.merge(pty_type, how='left', left_on='pty-cde', right_on='pty_cde')
job = job.drop('pty_type', axis = 1)

# mapping priority code of each report with priority classifcation of repair (routine, cyclic, emergency, void...)
pty_map = dict(pty_codes[['pty_cde','pty_classification']].values)
job['priority'] = job['pty-cde'].map(pty_map)
job.drop('pty-cde', axis = 1, inplace = True)
# replacing OLD SOR trade codes with NEW ones
job['sortrd-cde-1'].replace(['BR', 'C', 'E', 'E1', 'F', 'G','GF', 'H', 'MI', 'P', 'PD', 'PO','R', 'SC', 'TI', 'W'], 
                             ['0B','0C','0E', '0E', '0F','0Z','0G','HP','NS','0P', '0D','0D','0R', '0S', '0I','0C'], 
                             inplace=True)

# mapping sor CODES to text description (Trade names) 
sor.set_index('cde', inplace = True)
sor_map = sor.to_dict()['dsc']
job['sor'] = job['sortrd-cde-1'].map(sor_map)


# ##### Adding 'Time of Year' features (predictors)
# 

# In[5]:


# creating new dataframe with relevant features only (predictors)
job1 = job[['job_report_date', 'pr-seq-no', 'void-num', 'priority', 'right-to-repair', 'sor']]

job1['Year'] = pd.to_datetime(job1['job_report_date']).dt.year
job1['Week'] = pd.to_datetime(job1['job_report_date']).dt.week
job1['Day'] = pd.to_datetime(job1['job_report_date']).dt.day
job1['Month'] = pd.to_datetime(job1['job_report_date']).dt.month
job1['WeekDay'] = pd.to_datetime(job1['job_report_date']).dt.dayofweek
job1['Holiday'] = job1['job_report_date'].isin(uk_holidays)
job1['BeginMonth']=job1.Day.isin([1,2,3]).astype(int)
job1['Weekend']=job1.WeekDay.isin([5,6]).astype(int)

job1.head()


# ###### Analysing Trades by number of times each is required for repair/maintenance job
# Plumbing jobs are the most frequent overall, closely followed by Electrical jobs. A large number of Gas Fitting jobs are also reported/executed as well as Carpentry jobs.
# A few classes of Trade tpes are extremely rare making them anomalous in nature. Fo example, 'Health', 'Kenny Thomson', 'Bonus', 'Time Extension', etc.. Other rare Trade classes (with a count of less than 1% in the entire dataset) are dealt with in the following cell

# In[6]:


job1['sor'].value_counts()


# ###### Reducing classes for better predictive perfromance of models

# In[7]:


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


# Dual Trade Type (unable to classify)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLUMBER/CARPENT'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLUMBER/ELECTRI'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLASTERER/TILER'].index, axis = 0)

# Grouping similar Trades together
job_reduced['sor'] = job_reduced['sor'].replace('ESTATE INS', 'INSPECTION')
job_reduced['sor'] = job_reduced['sor'].replace('PAINT/DECORATOR', 'DECS ISSUE') # decs issues was majority class (11000+). replace lower number (7000+) with higher number


# Grouping Trades (with count of less than 1%) together that need a Specialist to execute the job
# Assigning all to existing class 'Specialist'
job_reduced['sor'] = job_reduced['sor'].replace('WOODWORM/DAMP', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('ACCESS/SECURITY', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('TV AERIALS', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('DISABLED ADAPT', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('MEDICAL', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('LIFT/STAIRLIFT', 'SPECIALIST')


# Grouping Trades that qualify as 'Wet Trades' together. [Individual count is less than 2% of dataset for each replaced class] 
# Assigning all to existing class 'Wet Works'
job_reduced['sor'] = job_reduced['sor'].replace('STONE MASON', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('TILER', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('BRICK WORK', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('GLAZING', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('PLASTERER', 'WET TRADES')

# Creating new class 'Exterior Works' and grouping all classes with similar nature together under this class
# [Individual count is less than 1% of dataset for each replaced class]
job_reduced['sor'] = job_reduced['sor'].replace('ASPHALTER', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('SCAFFOLDING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('SKIP HIRE', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('ROOFING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('DRAINAGE', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('FENCING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('GROUND WORKS', 'EXTERIOR WORKS')

# Creating new class 'Interior Non Utility' and grouping all classes with similar nature together under this class
# [Individual count is less than 0.5% of dataset for each replaced class]
job_reduced['sor'] = job_reduced['sor'].replace('BATH REFURBISH', 'INTERIOR NON UTILITY')
job_reduced['sor'] = job_reduced['sor'].replace('FLOORER', 'INTERIOR NON UTILITY')


# Trade Classes are much more balanced than before in terms of count 
print('Trade Class imbalance significantly corrected')

# Checking reduced Trade distribution 
job_reduced['sor'].value_counts()


# #### Calculating number of 'report counts' for each date in the dataframe

# In[8]:


job_counts = job_reduced.groupby(['job_report_date', 'sor']).size().reset_index(name='repair_count')
job_counts = job_counts.iloc[1:]
job_counts


# In[9]:


# converting job_report_date datatype to datetime
job_counts['job_report_date'] = pd.to_datetime(job_counts['job_report_date'])


# ##### Adding missing date-trade combinations to job_count dataframe
# For dates where no count for a particular Trade is reported, repair_count = 0

# In[10]:


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
all_combinations_df


# In[11]:


# merging original job_counts frame with all_combinations_df
all_combinations_df['job_report_date'] = pd.to_datetime(all_combinations_df['job_report_date'])

merged_df = pd.merge(all_combinations_df, job_counts, on=['job_report_date', 'sor'], how='left')

# fillign missing values in repair_count with 0
merged_df['repair_count'].fillna(0, inplace=True)
merged_df


# #### Property data

# In[12]:


property_class.set_index('prcls_cde', inplace = True)
property_class_map = property_class.to_dict()['prcls_dsc']
properties['property_class'] = properties['prcls-cde'].map(property_class_map)

property_style.set_index('aprstyle-cde', inplace = True)
property_style_map = property_style.to_dict()['dsc']
properties['property_style'] = properties['aprstyle-cde'].map(property_style_map)

property_street_add.set_index('cde', inplace = True)
property_street_map = property_street_add.to_dict()['loc-cde']
properties['location_code'] = properties['str-cde'].map(property_street_map)


# ##### Checking for missing values
# 

# In[13]:


properties.isna().sum()


# #### Imputing missing property values 

# Construction Year

# In[14]:


# grouping by 'location_code', 'postcode', 'str-cde', 'property_class', 'prtyp-cde' to calculate median construction year
imputation_values= properties.groupby(['location_code', 'postcode', 'str-cde', 'property_class', 'prtyp-cde'])['construction-yr'].median()

# grouping by 'str-cde', 'property_class' to calculate median construction year for remaining missing values
imputation_values_remaining = properties.groupby(['str-cde', 'property_class'])['construction-yr'].median()


# In[15]:


# excluding properties with construction year equal to 0 or NaN
mask = (properties['construction-yr'] == 0) | properties['construction-yr'].isna()

# imputation to fill in missing values using groupby values
properties.loc[mask, 'construction-yr'] = properties.loc[mask].apply(
    lambda prop: imputation_values.get(
        (prop['location_code'], prop['postcode'], prop['str-cde'], prop['property_class'], prop['prtyp-cde']), 
        imputation_values_remaining.get((prop['str-cde'], prop['property_class']), 0)
    ), 
    axis=1
)

# converting remaining 0 values to NaN
properties.loc[properties['construction-yr'] == 0, 'construction-yr'] = np.nan

properties.isna().sum()


# In[16]:


# rounding imputed values to integers
properties['construction-yr'] = properties['construction-yr'].round().astype('Int64')

# setting construction year range as values outside this range are incomprehensible
min_year = 1700
max_year = 2023
properties['construction-yr'] = properties['construction-yr'].apply(lambda x: x if pd.isnull(x) or (min_year <= x <= max_year) else np.nan)

# converting 0 values to NaN
properties.loc[properties['construction-yr'] == 0, 'construction-yr'] = np.nan
properties['construction-yr'] = pd.to_numeric(properties['construction-yr'], errors='coerce')


# Most missing values belong to the property class 'Street' (please see below)

# In[17]:


properties[properties['construction-yr'].isna()]['property_class'].value_counts()


# Filling missing values with median values of each property_class

# In[18]:


na_public_bldg_idx = properties[(properties['construction-yr'].isna())&(properties['property_class']=='Public Bldg')].index
properties.loc[na_public_bldg_idx,'construction-yr'] = properties[properties['property_class']=='Public Bldg']['construction-yr'].median()

na_house_idx = properties[(properties['construction-yr'].isna())&(properties['property_class']=='House')].index
properties.loc[na_house_idx,'construction-yr'] = properties[properties['property_class']=='House']['construction-yr'].median()

na_block_idx = properties[(properties['construction-yr'].isna())&(properties['property_class']=='Block')].index
properties.loc[na_block_idx,'construction-yr'] = properties[properties['property_class']=='Block']['construction-yr'].median()

na_garage_idx = properties[(properties['construction-yr'].isna())&(properties['property_class']=='Garage')].index
properties.loc[na_garage_idx,'construction-yr'] = properties[properties['property_class']=='Garage']['construction-yr'].median()

na_street_idx = properties[(properties['construction-yr'].isna())&(properties['property_class']=='Street')].index
properties.loc[na_street_idx,'construction-yr'] = properties[properties['property_class']=='Street']['construction-yr'].median()

na_virtual_idx = properties[(properties['construction-yr'].isna())&(properties['property_class']=='Virtual')].index
properties.loc[na_virtual_idx,'construction-yr'] = properties[properties['property_class']=='Virtual']['construction-yr'].median()

na_commercial_idx = properties[(properties['construction-yr'].isna())&(properties['property_class']=='Commercial')].index
properties.loc[na_commercial_idx,'construction-yr'] = properties[properties['property_class']=='Commercial']['construction-yr'].median()


# In[19]:


# selecting final columns from properties - keeping construction year (to calculate property age later based on job REPORT DATE)
prop_final = properties[['seq-no', 'co-own', 'postcode','prtyp-cde', 'property_class', 'property_style',
       'location_code','construction-yr']]


# ###### Missing values: Location Code

# In[20]:


# fill in missing values for location codes based on analysing street code and textual 'address', assigning relevant location_code meticulously
# identifying locations of missing location codes and filling them
prop_final.loc[[8552, 8553, 8564, 8565, 8566, 8567, 8568, 8569, 8570],'location_code'] = 'BKW'

prop_final.loc[[8554, 8555, 8556, 8557, 8558, 8559, 8579],'location_code'] = 'STR'

prop_final.isna().sum()


# #### Merge job frame with final propertty frame
# 

# In[21]:


job_prop = job_reduced.merge(prop_final, left_on='pr-seq-no', right_on='seq-no')


# ##### check missing values

# In[22]:


job_prop.isna().sum()


# In[23]:


# filling missing values in priority and SOR
job_prop['priority'].fillna('Other', inplace=True)
job_prop['sor'].fillna('NOT SPECIFIED', inplace=True)


# ##### Calculate Property Age at the time of report

# In[24]:


job_prop['job_report_date'] = pd.to_datetime(job_prop['job_report_date'])

# changing 'construction-yr' datatype to string, removing decimal points and converting to datetime
job_prop['construction-yr'] = pd.to_datetime(job_prop['construction-yr'].astype(str).str.replace(r'\.0$', '', regex=True))

# PROPERTY AGE:
# calculating 'property_age' based on 'construction-yr' and 'job_report_date'
# property age at the time of job reported
job_prop['property_age'] = (job_prop['job_report_date'].dt.year - job_prop['construction-yr'].dt.year)
# final DataFrame
job_prop[['job_report_date', 'construction-yr', 'property_age']]


# In[25]:


# filtering properties with property age less than 0
negative_property_age = job_prop[job_prop['property_age'] < 0]
# construction year was wrongly imputed for these. Fill with median value
median_property_ages = job_prop.groupby('property_class')['property_age'].median()
# replacing negative values in the 'property_age' column with the median values
negative_mask = job_prop['property_age'] < 0
job_prop.loc[negative_mask, 'property_age'] = job_prop.loc[negative_mask, 'property_class'].map(median_property_ages)
# remaining negative values are for Property Class STREET (median 'property age' = 4)
# repalcing negative values in the 'property_age' column with 4 for property_class 'Street'
street_mask = (job_prop['property_age'] < 0) & (job_prop['property_class'] == 'Street')
job_prop.loc[street_mask, 'property_age'] = 4
# filtering properties with property age greater than 200
property_age_greater_200 = job_prop[job_prop['property_age'] > 200]

# properties with property age greater than 200
print('Properties with property age greater than 200:')
property_age_greater_200[['pr-seq-no','location_code','property_class','property_age']]


# ###### property size

# In[26]:


property_size = property_size.drop(['Unnamed: 2','dsc'], axis = 1)
property_size = property_size.set_index('cde')
property_size.columns = ['size']
property_size


# In[27]:


# map property size to job mainframe
property_size_map = property_size.to_dict()
property_size_map = property_size_map['size']
job_prop['prtyp-cde'] = job_prop['prtyp-cde'].str.upper()
job_prop['property_size'] = job_prop['prtyp-cde'].map(property_size_map)


# #### Selecting features for modelling

# Adding quarter value

# In[28]:


df = job_prop[['job_report_date','Year','Month','Week','Day','sor','priority','property_class', 'prtyp-cde', 'property_age', 'location_code','property_size']]

df['quarter'] = df['job_report_date'].dt.to_period('Q')
df['quarter'] = df['quarter'].astype(str)
df['quarter_value'] = df['quarter'].str.extract(r'Q(\d+)')
df['quarter_value'] = df['quarter'].str.extract(r'Q(\d+)').apply(lambda x: 'Q' + x)


# ### Quarterly repair count prediction for each repair trade type
# #### based on Property attributes
# 

# In[29]:


repair_count = df.groupby(['Year','quarter_value','property_class','property_size', 'location_code','property_age','sor']).size()

repair_count= repair_count.reset_index(name = 'repair_count')
# Dropping anomalies (3 repairs from year 1900)
repair_count = repair_count[3:]
repair_count = repair_count.reset_index(drop = True)
repair_count


# In[30]:


# adding zero repair_count rows - for unique combination of property characterics in the dataframe that have no repairs recorded
trades = pd.DataFrame({'sor': ['WET TRADES', 'CARPENTRY', 'GAS FITTER.', 'PLUMBING',
                                   'NOT SPECIFIED', 'PRE-TERMINATION', 'ELECTRICAL',
                                   'INTERIOR NON UTILITY', 'CALL OUTS', 'GHS', 'DECS ISSUE',
                                   'DECENT HOMES', 'EXTERIOR WORKS', 'Gas Servicing',
                                   'SPECIAL/QUOTES', 'INSPECTION', 'SPECIALIST', 'ESW', 'ASBESTOS',
                                   'FITTER', 'DAY RATES', 'GENERAL', 'METALWORKER']})

# unique combinations of relevant features
unique_combinations = repair_count.drop(columns=['sor', 'repair_count']).drop_duplicates()

# merging unique combinations with priorities to ensure all priority classes are present for each combination
all_trades = pd.merge(unique_combinations.assign(key=1), trades.assign(key=1), on='key').drop(columns='key')

# merging back with original frame to get repair counts and fill missing repair counts with 0
final_counts = pd.merge(all_trades, repair_count, on=['Year', 'quarter_value', 'property_class', 'property_size', 'location_code', 'property_age', 'sor'], how='left').fillna(0)
final_counts


# ##### Separating data into 23 datasets for training multiple models to predict repair count for each class

# In[47]:


unique_sor_classes = final_counts['sor'].unique()
sor_dataframes = {}

# creating separate dataframes for each trade class 
for sor_class in unique_sor_classes:
    sor_dataframes[sor_class] = final_counts[final_counts['sor'] == sor_class]

    
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


# #### Experiment 1: Using Classifiers for predicting demand for each trade type
# 

# ##### defining encoder and scaler

# In[56]:


def one_hot_encoder(data, cat_cols, drop_nan = True, nan_cols = []):
    cat_cols = [col for col in cat_cols if col in data.columns]
    encoded = pd.get_dummies(data, columns = cat_cols)
    if not drop_nan == False:
        for nan_col in nan_cols:
            if nan_col in encoded.columns:
                encoded = encoded.drop(nan_col, axis = 1)
    return encoded

cat_cols = ['property_class', 'property_size','location_code','quarter_value']

# MinMaxScaler for data normalization
scaler = MinMaxScaler()


# In[57]:


# load your datasets and preprocess them
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


# Create a dictionary to store classifier names and instances
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'LightGBM': LGBMClassifier(random_state=42)
}

# creating list to store metrics
all_classifiers_metrics = []

# iterating over each dataset
for dataset_name, dataset in datasets.items():
    print(f'Training classifiers for dataset: {dataset_name}')
    
    # splitting dataset into training and testing data
    predictors = dataset.drop(['repair_count','sor'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)

    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)

    # fit-transforming encoded training data
    x_train_normalized = scaler.fit_transform(X_train_encoded)

    # transforming testing data using fitted scaler
    x_test_normalized = scaler.transform(X_test_encoded)
    
    # storing metrics of each classifier for current dataset
    dataset_metrics = []

    # iterating over classifiers, training on normalized data, predicting, getting performance metrics
    for name, classifier in classifiers.items():
        classifier.fit(x_train_normalized, y_train)
        y_pred = classifier.predict(x_test_normalized)
        accuracy = accuracy_score(y_cv, y_pred)
        f1 = f1_score(y_cv, y_pred, average='weighted')
        f1_macro = f1_score(y_cv, y_pred, average='macro')  
        recall = recall_score(y_cv, y_pred, average='macro', zero_division=0)

        dataset_metrics.append({'Classifier': name, 'Accuracy': accuracy, 'F1 Score': f1, 'F1 Macro': f1_macro, 'Recall': recall})

    # storing metrics for current dataset
    all_classifiers_metrics.extend([(dataset_name, metrics) for metrics in dataset_metrics])


# In[58]:


all_classifiers_metrics


# In[60]:


# creating lists to add dictionary scores to dataframe
datasets = []
classifiers = []
accuracies = []
f1_scores = []
f1_macro = []
recalls = []

# iterating over all_classifiers_metrics and extracting data for final dataframe
for dataset, metrics in all_classifiers_metrics:
    datasets.append(dataset)
    classifiers.append(metrics['Classifier'])
    accuracies.append(metrics['Accuracy'])
    f1_scores.append(metrics['F1 Score'])
    f1_macro.append(metrics['F1 Macro'])
    recalls.append(metrics['Recall'])

# creating classifier metric dataframe
metrics_classifiers_df = pd.DataFrame({
    'Dataset': datasets,
    'Classifier': classifiers,
    'Accuracy': accuracies,
    'F1 Score': f1_scores,
    'F1 Macro': f1_macro,
    'Recall': recalls
})

metrics_classifiers_df


# In[61]:


# sorting classifier scores (accuracy) in increasing order
sorted_classifier_metrics = metrics_classifiers_df.sort_values(by='Accuracy')
each_classifier_scores = sorted_classifier_metrics.groupby('Classifier')


# plotting accuracy and F1 score for dataset for each classifier
for classifier, classifier_scores in each_classifier_scores:
    plt.figure(figsize=(10, 6))
    plt.title(f'{classifier} - Accuracy and F1 Score')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.xticks(rotation=45)

    plt.plot(classifier_scores['Dataset'], classifier_scores['Accuracy'], label='Accuracy', marker='o')
    plt.plot(classifier_scores['Dataset'], classifier_scores['F1 Score'], label='F1 Score', marker='o')

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[62]:


# classifier with highest average accuracy and F1 Macro
classifier_accuracy = metrics_classifiers_df.groupby('Classifier')['Accuracy'].mean()
best_classifier = classifier_accuracy.idxmax()
highest_accuracy = classifier_accuracy.max()

print(f"The best performing classifier is '{best_classifier}' with an average accuracy of {highest_accuracy:.6f}")
print(f"The best performing classifier is '{best_classifier}' with an average F1 Score of {metrics_classifiers_df.groupby('Classifier')['F1 Macro'].mean().max()}")


# In[63]:


best_classifier_scores = metrics_classifiers_df[metrics_classifiers_df['Classifier'] == 'CatBoost']
best_classifier_scores = best_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
best_classifier_scores


# In[64]:


catboost_classifier_scores = metrics_classifiers_df[metrics_classifiers_df['Classifier'] == 'CatBoost']
catboost_classifier_scores = catboost_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
plt.figure(figsize=(10, 6))
plt.title(f'CatBoost Classifier - Accuracy')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.plot(catboost_classifier_scores['Dataset'], catboost_classifier_scores['Accuracy'], label='Accuracy', marker='o')
plt.plot(catboost_classifier_scores['Dataset'], catboost_classifier_scores['F1 Macro'], label='F1 Macro', marker='o')

plt.legend()
plt.tight_layout()
plt.show()


# No experimentation with oversampling since that does not improve results 

# #### Experiment 2: Using Regressors for predicting demand for each trade type

# #### Regressors with target variable log-transformed

# In[67]:


# data for regression. Log tranformed target
datasets = [carpentry, electrical, gas_fitter, exterior_works, general, ghs, metalworker, plumbing, decs_issue, 
            interior_non_utility, not_specified, specialist, pre_termination, wet_trades, call_outs, fitter, 
            decent_homes, inspection, day_rates, special_quotes, esw, gas_servicing, asbestos]

# dataset names 
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


# In[69]:


# datasets for regression (after target log transformed)
datasets_for_regression = [carpentry_reg, electrical_reg, gas_fitter_reg, exterior_works_reg, general_reg, ghs_reg, 
                           metalworker_reg, plumbing_reg, decs_issue_reg, interior_non_utility_reg, not_specified_reg, 
                           specialist_reg, pre_termination_reg, wet_trades_reg, call_outs_reg, fitter_reg, decent_homes_reg,
                           inspection_reg, day_rates_reg, special_quotes_reg, esw_reg, gas_servicing_reg, asbestos_reg]


rows = []
# all our selected regressors
regressors = [
    LinearRegression,
    KNeighborsRegressor,
    RandomForestRegressor,
    Lasso,
    ElasticNet,
    GradientBoostingRegressor,
    LGBMRegressor,
    CatBoostRegressor
]

# metrics to calculate
metrics = {
    'MSE': mean_squared_error,
    'R2': r2_score,
}

# dataFrame to store the scores
scores_df = pd.DataFrame(columns=['Method', 'Dataset'] + list(metrics.keys()))

# iterating over regressors
for regressor in regressors:
    for dataset_name, dataset in zip(['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                                      'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                                      'specialist', 'pre_termination', 'wet_trades', 'call_outs','fitter', 
                                      'decent_homes', 'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 
                                      'asbestos'], datasets_for_regression):
        
        print(f'Training {regressor.__name__} on {dataset_name}')

        # splitting the dataset into features and target
        predictors = dataset.drop(['repair_count','sor'], axis=1)
        target = dataset['repair_count']
        x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)

        # one-hot encoding categorical columns
        X_train_encoded = one_hot_encoder(x_train, cat_cols)
        X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)

        # fit-transforming training data using scaler 
        x_train_normalized = scaler.fit_transform(X_train_encoded)

        # transforming testing data using fitted scaler
        x_test_normalized = scaler.transform(X_test_encoded)

        # training the regressors and calculate metrics
        metrics_scores = []
        for metric_name, metric_func in metrics.items():
            model = regressor()
            model.fit(x_train_normalized, y_train)
            y_pred = model.predict(x_test_normalized)
            metric_score = metric_func(y_cv, y_pred)
            metrics_scores.append(metric_score)
            
        # storing regressor name and corresponding score 
        row_data = {'Method': regressor.__name__, 'Dataset': dataset_name, **dict(zip(list(metrics.keys()), metrics_scores))}
        rows.append(row_data)

# storing all regressor names and scores in one place (concatenate all dataframes corresponding to each regressor)
scores_df = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)


# In[70]:


scores_df


# In[71]:


# fig, ax = plt.subplots(figsize=(10, 6))
# ax.axis('off')
# ax.table(cellText=scores_df.values, colLabels=scores_df.columns, cellLoc='center', loc='center')
# plt.savefig('scores_table.png', bbox_inches='tight', pad_inches=0.5)
# plt.show()


# In[72]:


# sorting scores of all regressors in increaing order of R2
sorted_scores_df = scores_df.sort_values(by='R2')

# getting scores of all regressors
all_regressor_scores = sorted_scores_df.groupby('Method')

# plotting R2 for each regressor trained on each type of trade dataset 
for regressor, regressor_scores in all_regressor_scores:
    plt.figure(figsize=(10, 6))
    plt.title(f'{regressor} - R2')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.xticks(rotation=65)

    plt.plot(regressor_scores['Dataset'], regressor_scores['R2'], label='R2', marker='o')
    #plt.plot(regressor_scores['Dataset'], regressor_scores['MSE'], label='MSE', marker='o')

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[73]:


# finding regressor with best metrics - highes r2 and lowest mse 
regressor_r2 = scores_df.groupby('Method')['R2'].mean()
best_regressor = regressor_r2.idxmax()
highest_r2 = regressor_r2.max()
regressor_mse = scores_df.groupby('Method')['MSE'].mean()
lowest_mse = regressor_mse.min()
print(f"The best performing regressor is '{best_regressor}' with an average R2 score of {highest_r2:.6f}")
print(f"The best performing classifier is '{best_regressor}' with an average mse score of {lowest_mse:.6f}")


# In[74]:


# isolating CatBoost Regressor scores
catboost_scores = scores_df[scores_df['Method'] == 'CatBoostRegressor']
catboost_scores


# In[75]:


# sorting catboost scores in order of decreasing R2
catboost_sorted = catboost_scores.sort_values(by = 'R2')
# plotting R2 for catboost regressor
plt.figure(figsize=(8, 5))
plt.title(f'CatBoost Regressor - R2')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.plot(catboost_sorted['Dataset'], catboost_sorted['R2'], label='R2', marker='o')
#plt.plot(catboost_scores['Dataset'], catboost_scores['MSE'], label='MSE', marker='o')

plt.legend()
plt.tight_layout()
plt.show()


# #### Combine Plots (best regressor and best classifier)

# In[77]:


# plotting regressor R2 vs classifier F1 weighted score
catboost_regressor_scores = scores_df[scores_df['Method'] == 'CatBoostRegressor']
catboost_classifier_scores = metrics_classifiers_df[metrics_classifiers_df['Classifier'] == 'CatBoost']

dataset_order_reg = catboost_regressor_scores.sort_values(by='R2')
dataset_order_cls = catboost_classifier_scores.sort_values(by='F1 Macro', ascending=False)

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


# #### Training best models again and saving
# Getting predicted values for comparison with true values
# ##### Observe comparison in output of cell

# In[79]:


datasets_for_regression = [carpentry_reg, electrical_reg, gas_fitter_reg, exterior_works_reg, general_reg, ghs_reg, 
                           metalworker_reg, plumbing_reg, decs_issue_reg, interior_non_utility_reg, not_specified_reg, 
                           specialist_reg, pre_termination_reg, wet_trades_reg, call_outs_reg, fitter_reg, decent_homes_reg,
                           inspection_reg, day_rates_reg, special_quotes_reg, esw_reg, gas_servicing_reg, asbestos_reg]

# dataset names
dataset_names =  ['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                  'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                  'specialist', 'pre_termination', 'wet_trades', 'call_outs','fitter', 
                  'decent_homes', 'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 
                  'asbestos']

best_regressor = CatBoostRegressor(verbose = False)

# not choosing classifier for predition of values for any dataset
regressors = [best_regressor] * 23
classifiers = [None] * 23

#regressors = [best_regressor] * 6 + [None] * 1
#classifiers = [None] * 6 + [best_classifier] * 1

comparison_dataframes = []

# looping over datasets and models
for dataset, classifier, regressor, dataset_name in zip(datasets, classifiers, regressors, dataset_names):
    
    if classifier or regressor:
        # cloning current model 
        trained_model = clone(classifier) if classifier else clone(regressor)
        
        # preparing predictors and target
        predictors = dataset.drop(['repair_count','sor'], axis=1)
        target = dataset['repair_count']
        
        # train test split
        x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)

        # onehot encoding categorical columns
        X_train_encoded = one_hot_encoder(x_train, cat_cols)
        X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)

        # fit-transforming encoded training data
        x_train_normalized = scaler.fit_transform(X_train_encoded)

        # transforming testing data using fitted scaler
        x_test_normalized = scaler.transform(X_test_encoded)
        
        # fitting the current model on normalized data
        trained_model.fit(x_train_normalized, y_train)
        
        # making predictions using trained model
        if classifier:
            pred = trained_model.predict(x_test_normalized)
            preds = [item[0] for item in pred]
        elif regressor:
            pred = trained_model.predict(x_test_normalized)
            y_pred_real = np.expm1(pred)
            
            rounded_pred = np.round(y_pred_real).astype(int)
            rounded_pred[rounded_pred < 0] = 0
            
            # inverse log transformation for log transformed target data used to train regressors
            y_cv_real = np.expm1(y_cv)
            y_cv_real = np.round(y_cv_real,0)
        
        # comparing predicted values with true values
        compare = {'y_cv': y_cv_real.reset_index(drop = True), 'pred': rounded_pred}
        df = pd.DataFrame(compare)
        
        # storing predicted vs true values for current dataset
        comparison_dataframes.append(df)

# true vs predicted values for all datasets
for idx, df in enumerate(comparison_dataframes):
    print(f"Predictions for dataset - '{dataset_names[idx]}':")
    display(df)
    print('\n')


# #### Residual plots for regression models trained on log-transformed datasets

# In[80]:


# storing models for analysis later through SHAP analysis
stored_models = []

# looping through all trade datasets
for dataset_name, dataset in zip(dataset_names, datasets_for_regression):
    print(f'plots for {dataset_name}')
    
    # preparing predictors and target
    predictors = dataset.drop(['repair_count','sor'], axis=1)
    target = dataset['repair_count']
    
    # train test split
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)

    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)

    # fit-transforming encoded training data
    x_train_normalized = scaler.fit_transform(X_train_encoded)

    # transforming testing data using fitted scaler
    x_test_normalized = scaler.transform(X_test_encoded)
    
    # initialize current model
    current_model = best_regressor
    # storing trained model
    stored_models.append((dataset_name, current_model))
    
    # fitting best regressor (catboost)
    current_model.fit(x_train_normalized, y_train, silent = True)

    # making predictions using trained model
    pred = current_model.predict(x_test_normalized)
    compare = pd.DataFrame({'y_cv': y_cv, 'pred': pred})
    
    # creating our subplots for true vs predicted, and residual plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    # true vs predicted plots
    ax1 = axes[0]
    ax1.scatter(compare['y_cv'], compare['pred'], color='blue', label='Predicted vs. True')
    ax1.plot(np.linspace(0, max(compare['y_cv']), 100), np.linspace(0, max(compare['y_cv']), 100), color='red', label='Ideal')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Actual vs Predicted - {dataset_name.upper()}')
    ax1.legend()
    
    # predicted vs residual plots
    ax2 = axes[1]
    residuals = compare['pred'] - compare['y_cv']
    ax2.scatter(compare['pred'], residuals, color='blue')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Predicted vs Residuals - {dataset_name.upper()}')
    
    # displaying plots
    plt.tight_layout()
    display(fig)
    plt.close()


# #### Model SHAP Analysis

# In[81]:


# using stored models to do SHAP analysis
stored_models


# In[85]:


# iterating over stored models and datasets 
for (dataset_name, model), dataset in zip(stored_models, datasets_for_regression):
    
    # features and target
    predictors = dataset.drop(['repair_count', 'sor'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)
   
    # fit-transforming encoded training data
    x_train_normalized = scaler.fit_transform(X_train_encoded)

    # transforming testing data using fitted scaler
    x_test_normalized = scaler.transform(X_test_encoded)
    
    # training stored model for current dataset
    model.fit(x_train_normalized, y_train)
    
    # getting feature importances from the model
    feature_importances = model.feature_importances_
    feature_names = x_train.columns
    
    # creating the SHAP explainer
    explainer = shap.Explainer(model)
    
    # explanation data
    explain_data = X_test_encoded
    
    # SHAP values
    shap_values = explainer(explain_data)
    
    # titles for all plots
    plt.gca().add_artist(plt.text(0.5, 1.08, f"Feature Importance plots for model trained on dataset - '{dataset_name}'", ha='center', va='center', transform=plt.gca().transAxes))

    # plotting feature importance plots 
    shap.plots.bar(shap_values, show = True)


# In[86]:


# waterfall plots

for (dataset_name, model), dataset in zip(stored_models, datasets_for_regression):
    print(f'waterfall plot for {dataset_name}')
    
    predictors = dataset.drop([ 'repair_count', 'sor',], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)
   
    # explaining current model
    explainer = shap.Explainer(model)
    data_for_explain = X_test_encoded 
    shap_values = explainer(data_for_explain)
    
    # Choose an index for which you want to create a waterfall plot
    index_to_plot = 0
    
    # title for summary plot for models trained on each dataset
    plt.gca().add_artist(plt.text(0.5, 1.08, f"Waterfall plot for model trained on dataset - '{dataset_name}'", ha='center', va='center', transform=plt.gca().transAxes))
    # water fall plot
    shap.plots.waterfall(shap_values[index_to_plot], max_display=10, show = True) 


# In[87]:


# summary plots

for (dataset_name, model), dataset in zip(stored_models, datasets_for_regression):
    print(f'summary plot for {dataset_name}')
    
    # features and target
    predictors = dataset.drop(['repair_count', 'sor'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)
    
    explainer = shap.Explainer(model)
    data_for_explain = X_test_encoded  
    
    shap_values = explainer(data_for_explain)
    # title for summary plot for each dataset
    plt.gca().add_artist(plt.text(0.5, 1.08, f"Summary plot for model trained on dataset - '{dataset_name}'", ha='center', va='center', transform=plt.gca().transAxes))
    # summary plot
    shap.summary_plot(shap_values)


# In[88]:


# force plots

for (dataset_name, model), dataset in zip(stored_models, datasets_for_regression):
    print(f'force plot for {dataset_name}')
    
    # features and target
    predictors = dataset.drop(['repair_count', 'sor',], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)
    
    # declaring explainer 
    explainer = shap.TreeExplainer(model)
    data_for_explain = X_test_encoded  
    
    # SHAP values for the first 50 data points
    shap_values = explainer.shap_values(X_train_encoded.iloc[0:50, :])  
    shap_plot = shap.force_plot(explainer.expected_value, shap_values, X_train_encoded.iloc[0:50, :])
    shap.save_html(f'force_plot_{dataset_name}.html', shap_plot )


# #### Saving models on system

# In[91]:


datasets_for_regression = [carpentry_reg, electrical_reg, gas_fitter_reg, exterior_works_reg, general_reg, ghs_reg, 
                           metalworker_reg, plumbing_reg, decs_issue_reg, interior_non_utility_reg, not_specified_reg, 
                           specialist_reg, pre_termination_reg, wet_trades_reg, call_outs_reg, fitter_reg, decent_homes_reg,
                           inspection_reg, day_rates_reg, special_quotes_reg, esw_reg, gas_servicing_reg, asbestos_reg]

# dataset names
dataset_names =  ['carpentry', 'electrical', 'gas_fitter', 'exterior_works', 'general', 'ghs', 
                  'metalworker', 'plumbing', 'decs_issue', 'interior_non_utility', 'not_specified', 
                  'specialist', 'pre_termination', 'wet_trades', 'call_outs','fitter', 
                  'decent_homes', 'inspection', 'day_rates', 'special_quotes', 'esw', 'gas_servicing', 
                  'asbestos']


regressors = [best_regressor] * 23
classifiers = [None] * 23 


comparison_dataframes = []


# looping over datasets and models
for dataset, classifier, regressor, dataset_name in zip(datasets, classifiers, regressors, dataset_names):
    print(f'Processing dataset: {dataset_name}')
    
    if classifier:
        model_type = 'classifier'
    elif regressor:
        model_type = 'regressor'
    else:
        continue
    
    # cloning the cuurent model (classifier or regressor)
    trained_model = clone(classifier) if classifier else clone(regressor)
    
    # preparing predictors and target
    predictors = dataset.drop(['repair_count','sor'], axis=1)
    target = dataset['repair_count']
    
    # splitting data into training and test set
    x_train, _, y_train, _ = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)

    # fit-transforming encoded training data
    x_train_normalized = scaler.fit_transform(X_train_encoded)
    
    # ftting the model
    trained_model.fit(x_train_normalized, y_train)
    
    # asssigning model name
    model_name = f'{model_type}_{dataset_name}_sor_prop_model'
    
    # saving trained model using joblib
    model_filename = f'{model_name}.joblib'
    joblib.dump(trained_model, model_filename)

    
print('All models saved')


# ##### The trained models are saved as separate joblib files in the same directory where this Jupyter Notebook is located. These models are later loaded for use in a separate notebook for creating the Dash based web-application. (Please see 'Dash GCH App Final')
