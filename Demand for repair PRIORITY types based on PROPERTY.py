#!/usr/bin/env python
# coding: utf-8

# #### Loading neccessary libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import holidays
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import shap
from sklearn.base import clone


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


# In[6]:


# Reducing number of Trade Type Classes 
# Removing classes with very few datapoints/instances (less than 0.05%)

job_reduced = job1.drop(job1[job1['sor']=='TIME EXTENSION'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='HEALTH'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='Kenny Thomson'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='BONUS'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='Void Clearance'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLUMBER/CARPENT'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLUMBER/ELECTRI'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLANNED WORKS'].index, axis = 0)
job_reduced = job_reduced.drop(job_reduced[job_reduced['sor']=='PLASTERER/TILER'].index, axis = 0)

# Assigning Estate Inspection with an existent class of same nature of Trade
# Assigning Paint/Decorator with an existent class of same nature of Trade
job_reduced['sor'] = job_reduced['sor'].replace('ESTATE INS', 'INSPECTION')
job_reduced['sor'] = job_reduced['sor'].replace('PAINT/DECORATOR', 'DECS ISSUE') # decs issues was majority class (11000+). replace lower number (7000+) with higher number


# Replace all Trades that require a Specialist to do the job with an existent Class label 'SPECIALIST'
job_reduced['sor'] = job_reduced['sor'].replace('WOODWORM/DAMP', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('ACCESS/SECURITY', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('TV AERIALS', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('DISABLED ADAPT', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('MEDICAL', 'SPECIALIST')
job_reduced['sor'] = job_reduced['sor'].replace('LIFT/STAIRLIFT', 'SPECIALIST')

# All trades that qualify as 'Wet Trades' are assigned to the existent class 'WET WORKS'
job_reduced['sor'] = job_reduced['sor'].replace('STONE MASON', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('TILER', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('BRICK WORK', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('GLAZING', 'WET TRADES')
job_reduced['sor'] = job_reduced['sor'].replace('PLASTERER', 'WET TRADES')

# Replace HandyLink (obsolete) with an existent Class label 'SPECIALIST'
# Replace dummy tickets with an existent Class label 'SPECIALIST'
job_reduced['sor'] = job_reduced['sor'].replace('HandyLink', 'GENERAL')
job_reduced['sor'] = job_reduced['sor'].replace('DUMMY TICKETS', 'NOT SPECIFIED')

# Create new class label (Exterior Works) in SOR and assign the class label to all Trades that fall into the category
job_reduced['sor'] = job_reduced['sor'].replace('ASPHALTER', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('SCAFFOLDING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('SKIP HIRE', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('ROOFING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('DRAINAGE', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('FENCING', 'EXTERIOR WORKS')
job_reduced['sor'] = job_reduced['sor'].replace('GROUND WORKS', 'EXTERIOR WORKS')

# Create new class label (Interior Non Utility)in SOR and assign the class label to all Trades that fall into the category
job_reduced['sor'] = job_reduced['sor'].replace('BATH REFURBISH', 'INTERIOR NON UTILITY')
job_reduced['sor'] = job_reduced['sor'].replace('FLOORER', 'INTERIOR NON UTILITY')

print('Target imbalance significantly improved')
# display value counts of Trade Types
job_reduced['sor'].value_counts()


# In[7]:


# Plotting reduced SOR TRADE by number of repair counts
plt.figure(figsize=(12, 7))
sns.countplot(data=job_reduced, x='sor', order=job_reduced['sor'].value_counts().index)
plt.xticks(rotation=75)
plt.show()


# In[8]:


# getting property class, style, and location to merge with property data
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

# In[9]:


properties.isna().sum()


# ### Imputing missing values

# ###### Construction Year

# In[10]:


len(properties[properties['construction-yr'].isna()])


# In[11]:


len(properties[properties['construction-yr']==0])


# Replacing 'zero' and 'unknown' construction years with the most frequently occuring construction year for the particular imputation group they belong in

# In[13]:


# grouping by 'location_code', 'postcode', 'str-cde', 'property_class', 'prtyp-cde' to calculate median construction year
imputation_values = properties.groupby(['location_code', 'postcode', 'str-cde', 'property_class', 'prtyp-cde'])['construction-yr'].median()


# In[14]:


# grouping by 'str-cde', 'property_class' to calculate median construction year for remaining missing values
imputation_values_remaining = properties.groupby(['str-cde', 'property_class'])['construction-yr'].median()
imputation_values_remaining


# In[15]:


# excluding properties with construction year equal to 0 or NaN
mask = (properties['construction-yr'] == 0) | properties['construction-yr'].isna()

# imputing to fill in missing values using groupby values
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


# Only a few missing construction year values are filled using this imputation method; Further analyze missing values for construction year. 

# In[16]:


# rounding imputed values to integers
properties['construction-yr'] = properties['construction-yr'].round().astype('Int64')

# setting construction year range as values outside this range are incomprehensible
min_year = 1700
max_year = 2023
properties['construction-yr'] = properties['construction-yr'].apply(lambda x: x if pd.isnull(x) or (min_year <= x <= max_year) else np.nan)

# converting 0 values to NaN
properties.loc[properties['construction-yr'] == 0, 'construction-yr'] = np.nan


# In[17]:


# checking all construction years
properties['construction-yr'] = pd.to_numeric(properties['construction-yr'], errors='coerce')
properties['construction-yr'].unique()


# Most missing values belong to the property class 'Street' (please see below)

# In[18]:


properties[properties['construction-yr'].isna()]['property_class'].value_counts()


# Filling missing values with median values of each property_class

# In[19]:


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


# In[20]:


# selecting final columns from properties - keeping construction year (to calculate property age later based on job REPORT DATE)
prop_final = properties[['seq-no', 'co-own', 'postcode','prtyp-cde', 'property_class', 'property_style',
       'location_code','construction-yr']]


# ###### Missing values: Location Code

# In[21]:


prop_final['location_code'].isna().sum()


# In[22]:


# fill in missing values for location codes based on analysing street code and textual 'address', assigning relevant location_code meticulously

# identifying locations of missing location codes and filling them
prop_final.loc[[8552, 8553, 8564, 8565, 8566, 8567, 8568, 8569, 8570],'location_code'] = 'BKW'

prop_final.loc[[8554, 8555, 8556, 8557, 8558, 8559, 8579],'location_code'] = 'STR'

prop_final.isna().sum()


# #### Merge job frame with final propertty frame

# In[23]:


job_prop = job_reduced.merge(prop_final, left_on='pr-seq-no', right_on='seq-no')
job_prop


# ##### check missing values

# In[24]:


job_prop.isna().sum()


# ######  Fill missing values in priority and SOR

# In[25]:


# filling missing values in priority and SOR
job_prop['priority'].fillna('Other', inplace=True)
job_prop['sor'].fillna('NOT SPECIFIED', inplace=True)


# ##### Calculate Property Age at the time of report

# In[26]:


job_prop['job_report_date'] = pd.to_datetime(job_prop['job_report_date'])

# changing 'construction-yr' datatype to string, removing decimal points and converting to datetime
job_prop['construction-yr'] = pd.to_datetime(job_prop['construction-yr'].astype(str).str.replace(r'\.0$', '', regex=True))

# PROPERTY AGE:
# calculating 'property_age' based on 'construction-yr' and 'job_report_date'
# property age at the time of job reported
job_prop['property_age'] = (job_prop['job_report_date'].dt.year - job_prop['construction-yr'].dt.year)
# final dataframe
job_prop[['job_report_date', 'construction-yr', 'property_age']]


# In[27]:


# filetering properties with property age less than 0
negative_property_age = job_prop[job_prop['property_age'] < 0]

# properties with negative property age
print('Properties with negative property age:')
negative_property_age


# ###### This means construction year was wrongly imputed for these. Fill with median value

# In[28]:


# median property age for each 'property_class' in 'job_prop' dataframe
median_property_ages = job_prop.groupby('property_class')['property_age'].median()

# replacing negative values in the 'property_age' column with the median values
negative_mask = job_prop['property_age'] < 0
job_prop.loc[negative_mask, 'property_age'] = job_prop.loc[negative_mask, 'property_class'].map(median_property_ages)


# ###### The remaining negative values are for Property Class STREET (median 'property age' = 4)

# In[29]:


# relpacing remaining negative values in the 'property_age' column with 4 for property_class 'Street'
street_mask = (job_prop['property_age'] < 0) & (job_prop['property_class'] == 'Street')
job_prop.loc[street_mask, 'property_age'] = 4


# In[30]:


# filtering out properties with property age greater than 200
property_age_greater_200 = job_prop[job_prop['property_age'] > 200]

# properties with property age greater than 200
print('Properties with property age greater than 200:')
property_age_greater_200[['pr-seq-no','location_code','property_class','property_age']]


# There is one property that is more than 200 years old - A Flat in Stroud

# ##### Plotting relationship between total repair demand and property age

# In[31]:


plt.figure(figsize = (20,10))
sns.histplot(binwidth=0.5, x='property_age', hue='priority', data=job_prop, stat='count', multiple='stack')
plt.title('Repair counts by Property Age', fontsize = 16)


# ##### Plotting relationship between total repair demand and property age (after removing anomolous 200 year old property)

# In[32]:


plt.figure(figsize = (20,10))
sns.histplot(binwidth=0.5, x='property_age', hue='priority', data=job_prop[job_prop['property_age'] < 150], stat='count', multiple='stack')
plt.title('Repair counts by Property Age', fontsize = 16)


# In[33]:


plt.figure(figsize = (20,10))
ax = sns.countplot(x='property_age', data=job_prop)
# adjusting x-axis tick frequency
tick_freq = 5  
ax.set_xticks(ax.get_xticks()[::tick_freq])
plt.title('Repair counts by Property Age', fontsize = 18)
plt.show()


# A large number of properties that report repairs with age between 50-70

# In[34]:


plot_df = job_prop.drop(job_prop[job_prop['property_age']>150].index, axis = 0)
mean_prop_age_counts = pd.DataFrame(plot_df.groupby('pr-seq-no')['property_age'].mean())
#mean_prop_age_counts.reset_index(inplace = True)
mean_prop_age_counts
ax = sns.histplot(binwidth=0.5, x='property_age', data=mean_prop_age_counts, stat='count')
tick_freq = 1
ax.set_xticks(ax.get_xticks()[::tick_freq])
plt.title('Property Counts by mean Property Age at the time of repair', fontsize = 10)
plt.show()


# ##### A spike in number of repairs reported for newer properties
# ##### Most repairs reported by properties in the age range 50 - 70 which makes sense because most properties owned by GCH are 50 - 70 years old (please see below)

# In[41]:


unique_properties = job_prop.drop_duplicates(subset=['pr-seq-no'])
unique_properties = unique_properties.reset_index(drop = True)
# dropping outlier that is more than 200 years old
filtered_properties = unique_properties[unique_properties['property_age'] <= 200]

plt.figure(figsize=(10, 6))
filtered_properties['property_age'].plot(kind='hist', bins = 20)

plt.title('Number of properties by Property Age')
plt.xlabel('Property Age')
plt.ylabel('Number of Properties')

plt.show()


# ##### property size

# In[35]:


property_size = property_size.drop(['Unnamed: 2','dsc'], axis = 1)
property_size = property_size.set_index('cde')
property_size.columns = ['size']
property_size


# ###### Map above values to main job_property frame

# In[36]:


property_size_map = property_size.to_dict()
property_size_map = property_size_map['size']
job_prop['prtyp-cde'] = job_prop['prtyp-cde'].str.upper()
job_prop['property_size'] = job_prop['prtyp-cde'].map(property_size_map)


# ### Selecting most important features for modelling

# In[37]:


df = job_prop[['job_report_date','Year','Month','Week','Day','sor','priority','property_class', 'prtyp-cde', 'property_age', 'location_code','property_size']]
df


# ##### Adding a column for Yearly Quarter

# In[38]:


df['quarter'] = df['job_report_date'].dt.to_period('Q')
df['quarter'] = df['quarter'].astype(str)
df['quarter_value'] = df['quarter'].str.extract(r'Q(\d+)')
df['quarter_value'] = df['quarter'].str.extract(r'Q(\d+)').apply(lambda x: 'Q' + x)
df


# ### Quarterly repair count prediction for each priority trade type
# #### based on Property attributes
# 
# 
# 

# In[39]:


repair_count = df.groupby(['Year','quarter_value','property_class','property_size', 'location_code','property_age','priority']).size()
repair_count= repair_count.reset_index(name = 'repair_count')
# Dropping anomalies (3 repairs from year 1900)
repair_count = repair_count[3:]
repair_count = repair_count.reset_index(drop = True)
repair_count


# ##### Adding zero repair_count rows - for unique combination of property characterics in the dataframe that have no repairs recorded
# For a more comprehensive picture of repair counts and to increase the learning ability of the model

# In[40]:


priorities = pd.DataFrame({'priority': ['Emergency Repair', 'Void Works', 'Routine Repair', 'Inspection', 'Other', 'Planned Work', 'Cyclical Works']})

# unique combinations of relevant features
unique_combinations = repair_count.drop(columns=['priority', 'repair_count']).drop_duplicates()

# merging unique combinations with priorities to ensure all priority classes are present for each combination
all_priority = pd.merge(unique_combinations.assign(key=1), priorities.assign(key=1), on='key').drop(columns='key')

# merging back with original frame to get repair counts and fill missing repair counts with 0
final_count = pd.merge(all_priority, repair_count, on=['Year', 'quarter_value', 'property_class', 'property_size', 'location_code', 'property_age', 'priority'], how='left').fillna(0)
final_count


# ##### Separating data into 7 datasets for training multiple models to predict repair count for each class

# creating multiple datasets for training multiple regressors/classifiers

# In[41]:


routine = final_count[final_count['priority']=='Routine Repair']
emergency = final_count[final_count['priority']=='Emergency Repair']
void = final_count[final_count['priority']=='Void Works']
planned = final_count[final_count['priority']=='Planned Work']
cyclical = final_count[final_count['priority']=='Cyclical Works']
other = final_count[final_count['priority']=='Other']
inspection = final_count[final_count['priority']=='Inspection']


# ##### Creating copies of each dataset and transforming target (repair count) with Log Transformation 
# For regression task as target will be continous. Classifiers will be trained on original datasets

# In[42]:


cyclical_reg = cyclical.copy()
planned_reg = planned.copy()
routine_reg = routine.copy()
other_reg = other.copy()
emergency_reg = emergency.copy()
void_reg = void.copy()
inspection_reg = inspection.copy()


# Appling log transformation to repair_count for each dataset
cyclical_reg['repair_count'] = np.log1p(cyclical_reg['repair_count'])
planned_reg['repair_count'] = np.log1p(planned_reg['repair_count'])
routine_reg['repair_count'] = np.log1p(routine_reg['repair_count'])
other_reg['repair_count'] = np.log1p(other_reg['repair_count'])
emergency_reg['repair_count'] = np.log1p(emergency_reg['repair_count'])
void_reg['repair_count'] = np.log1p(void_reg['repair_count'])
inspection_reg['repair_count'] = np.log1p(inspection_reg['repair_count'])


# ### Experiment - Using Regressors for predicting demand for each priority type

# 
# #### Train Test Split, Encoding, normalizing each dataset before fitting various regressors

# In[46]:


# log-transformed datasets for regression task
datasets = [routine_reg, emergency_reg, planned_reg, cyclical_reg, void_reg, inspection_reg, other_reg]

# categorical predictors
cat_cols = ['property_class', 'property_size','location_code','quarter_value']

# define fucntion for one-hot encoding
def one_hot_encoder(data, cat_cols, drop_nan = True, nan_cols = []):
    cat_cols = [col for col in cat_cols if col in data.columns]
    encoded = pd.get_dummies(data, columns = cat_cols)
    if not drop_nan == False:
        for nan_col in nan_cols:
            if nan_col in encoded.columns:
                encoded = encoded.drop(nan_col, axis = 1)
    return encoded

# MinMaxScaler for data normalization
scaler = MinMaxScaler()

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
    for dataset_name, dataset in zip(['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other'], datasets):
        print(f'Training {regressor.__name__} on {dataset_name}')

        # splitting the dataset into features and target
        predictors = dataset.drop(['repair_count','priority'], axis=1)
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


# In[47]:


# display scores for all regressors 
scores_df


# In[48]:


# fig, ax = plt.subplots(figsize=(10, 6))
# ax.axis('off')
# ax.table(cellText=scores_df.values, colLabels=scores_df.columns, cellLoc='center', loc='center')
# plt.savefig('scores_table.png', bbox_inches='tight', pad_inches=0.5)
# plt.show()


# In[49]:


# visualizing scores for all regressors
sorted_scores_df = scores_df.sort_values(by='R2')
each_reg_scores = sorted_scores_df.groupby('Method')

# plotting R squared score for each dataset for each regressor
for regressor, reg_scores in each_reg_scores:
    plt.figure(figsize=(10, 6))
    plt.title(f'{regressor} - R2')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.xticks(rotation=45)

    plt.plot(reg_scores['Dataset'], reg_scores['R2'], label='R2', marker='o')

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[50]:


# best performing regressor based on average R2 score 
regressor_r2 = scores_df.groupby('Method')['R2'].mean()
best_regressor = regressor_r2.idxmax()
highest_r2 = regressor_r2.max()

print(f"The best performing regressor is '{best_regressor}' with an average R2 score of {highest_r2:.6f}")


# In[51]:


# average MSE of best performing regressor
regressor_mse = scores_df.groupby('Method')['MSE'].mean()
lowest_mse = regressor_mse.min()
print(f"The best performing regressor is '{best_regressor}' with an average Mean Squared Error of {lowest_mse:.6f}")


# In[52]:


# isolating CatBoost Regressor scores
catboost_scores = scores_df[scores_df['Method'] == 'CatBoostRegressor']
catboost_scores


# In[53]:


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


# ### Experiment - Using Classifiers for predicting demand for each priority type

# In[64]:


# loading all orginal datasets without log-transform
datasets = {
    'routine': routine, 
    'emergency': emergency,
    'void': void,
    'other': other,
    'cyclical': cyclical,
    'inspection': inspection,
    'planned': planned
}

# loading all selectied classifiers for experiment
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
    predictors = dataset.drop(['repair_count','priority'], axis=1)
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


# In[65]:


all_classifiers_metrics


# In[66]:


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


# In[68]:


# sorting classifier scores (accuracy) in increasing order
sorted_class_metrics_df = metrics_classifiers_df.sort_values(by='Accuracy')
each_classifier_scores = sorted_class_metrics_df.groupby('Classifier')

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


# In[69]:


# sorting metrics in ascending order of accuracy for all classifiers
sorted_metrics_df = metrics_classifiers_df.sort_values(by='Accuracy')

# gettings metrics of each classifier
grouped = sorted_metrics_df.groupby('Classifier')

# plotting accuracy and F1 score for each type of classifier (trained on each priority dataset)
for classifier, group_data in grouped:
    plt.figure(figsize=(10, 6))
    plt.title(f'{classifier} - Accuracy and F1 Macro Score')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.xticks(rotation=45)

    plt.plot(group_data['Dataset'], group_data['Accuracy'], label='Accuracy', marker='o')
    plt.plot(group_data['Dataset'], group_data['F1 Macro'], label='F1 Macro', marker='o')

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[71]:


# classifier with highest average accuracy and F1 Macro
classifier_accuracy = metrics_classifiers_df.groupby('Classifier')['Accuracy'].mean()
best_classifier = classifier_accuracy.idxmax()
highest_accuracy = classifier_accuracy.max()

print(f"The best performing classifier is '{best_classifier}' with an average accuracy of {highest_accuracy:.6f}")
print(f"The best performing classifier is '{best_classifier}' with an average F1 Score of {metrics_classifiers_df.groupby('Classifier')['F1 Macro'].mean().max()}")


# In[74]:


# isolating scores for best performing classifier
catboost_classifier_scores = metrics_classifiers_df[metrics_classifiers_df['Classifier'] == 'CatBoost']
catboost_classifier_scores = catboost_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
catboost_classifier_scores


# In[75]:


# plotting scores of best classifier
cat_classifier_scores = metrics_classifiers_df[metrics_classifiers_df['Classifier'] == 'CatBoost']
cat_classifier_scores = cat_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
plt.figure(figsize=(10, 6))
plt.title(f'CatBoost Classifier - Accuracy')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.plot(cat_classifier_scores['Dataset'], cat_classifier_scores['Accuracy'], label='Accuracy', marker='o')
#plt.plot(cat_classifier_scores['Dataset'], cat_classifier_scores['F1 Score'], label='F1 Score', marker='o')

plt.legend()
plt.tight_layout()
plt.show()


# #### Combine Plots (best regressor and best classifier)

# In[77]:


# plotting regressor R2 vs classifier F1 weighted score
catboost_regressor_scores = scores_df[scores_df['Method'] == 'CatBoostRegressor']
catboost_classifier_scores = metrics_classifiers_df[metrics_classifiers_df['Classifier'] == 'CatBoost']

dataset_order_reg = catboost_regressor_scores.sort_values(by='R2')
dataset_order_cls = catboost_classifier_scores.sort_values(by='F1 Score', ascending=False)

dataset_order_cls_reordered = dataset_order_cls.set_index('Dataset').loc[dataset_order_reg['Dataset']].reset_index()

plt.figure(figsize=(10, 6))

plt.title('Scores - CatBoost Regressor & CatBoost Classifier')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=65)

plt.plot(dataset_order_reg['Dataset'], dataset_order_reg['R2'], label='R2 (CatBoost Regressor)', marker='o')
plt.plot(dataset_order_cls_reordered['Dataset'], dataset_order_cls_reordered['F1 Score'], label='F1 Score W (CatBoost Classifier)', marker='o')


plt.legend()
plt.tight_layout()
plt.show()


# In[78]:


# plotting regressor R2 vs classifier accuracy
plt.figure(figsize=(10, 6))

plt.title('Scores - CatBoost Regressor & CatBoost Classifier')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=65)

plt.plot(dataset_order_reg['Dataset'], dataset_order_reg['R2'], label='R2 (CatBoost Regressor)', marker='o')
plt.plot(dataset_order_cls_reordered['Dataset'], dataset_order_cls_reordered['Accuracy'], label='Accuracy (CatBoost Classifier)', marker='o')


plt.legend()
plt.tight_layout()
plt.show()


# In[79]:


# plotting regressor R2 vs classifier F1 Macro
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


# ##### Saving best models

# In[80]:


# selected catboost regressor for predicting repair counts for all priority types
best_regressor = CatBoostRegressor()

# best_classifier = CatBoostClassifier()


# #### Tuning regressors

# In[85]:


# dictionary for storing tuned regressors results on all datasetes
best_regressor_tuned = {}

# declaring dataset names
dataset_names = ['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other']
datasets = [routine_reg, emergency_reg, planned_reg, cyclical_reg, void_reg, inspection_reg, other_reg]

# iterating over all datasets
for dataset_name, dataset in zip(dataset_names, datasets):
    print(f'Tuning CatBoost Regressor for dataset: {dataset_name}')
    
    # splitting the dataset into features and target
    predictors = dataset.drop(['repair_count', 'priority'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)

    # fit-transforming encoded training data
    x_train_normalized = scaler.fit_transform(X_train_encoded)

    # transforming testing data using fitted scaler
    x_test_normalized = scaler.transform(X_test_encoded)
 
    # defining parameter grid for hyperparamter tuning
    param_grid = {
        'learning_rate': [0.1, 0.2],
        'depth': [6, 8, 10],
        'reg_lambda': [0.01, 0.1, 1.0],    
        'early_stopping_rounds': [10, 20] 
        
    }

    # tunign for our chosen best regressor
    catboost_regressor = CatBoostRegressor(random_state=42, verbose=0)

    # using GridSearchCV to find best hyperparamterss
    grid_search = GridSearchCV(catboost_regressor, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(x_train_normalized, y_train)

    # getting best parameters and best corresponding regressor model
    best_params = grid_search.best_params_
    best_regressor = grid_search.best_estimator_

    # predicitng using best tuned regressor
    y_pred = best_regressor.predict(x_test_normalized)
    mse = mean_squared_error(y_cv, y_pred)
    r2 = r2_score(y_cv, y_pred)
    
    # print(f'Best parameters for {dataset_name}: {best_params}')
    # print(f'Mean Squared Error on validation data: {mse}')
    # print(f'R2 score on validation data: {r2}')
    
    # saving scores for tuned regressor for current dataset
    best_regressor_tuned[dataset_name] = {
        'best_params': best_params,
        'mse': mse,
        'r2': r2,
        'best_regressor': best_regressor
    }

# printing and saving scores for tuned regressor for all datasets
for dataset_name, tuned in best_regressor_tuned.items():
    print(f'Best parameters for dataset {dataset_name}:')
    print(f'Best parameters: {tuned['best_params']}')
    print(f'Mean Squared Error: {tuned['mse']}')
    print(f'R2 score: {tuned['r2']}')
    print('\n')


# In[87]:


# saving tuned regressors amd results in dataframe
tuned_list = []
for dataset_name, tuned in best_regressor_tuned.items():
    tuned_list.append({
        'Dataset': dataset_name,
        'Best Parameters': tuned['best_params'],
        'Mean Squared Error': tuned['mse'],
        'R2 Score': tuned['r2'],
        'Best Regressor': tuned['best_regressor']
    })

tuned_df = pd.DataFrame(tuned_list)
tuned_df


# ### True vs Predicted values
# #### Predicted vs Residual plots (non-transformed target data)

# In[89]:


# plotting true vs predicted values for all datasets
datasets = [routine, emergency, void, other , planned, cyclical, inspection]
dataset_names = ['routine', 'emergency', 'void', 'other', 'planned', 'cyclical', 'inspection']

# looping through all priority datasets
for dataset_name, dataset in zip(dataset_names, datasets):
    print(f'Creating plots for {dataset_name}')
    
    # preparing predictors and target
    predictors = dataset.drop(['repair_count','priority'], axis=1)
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

    # fitting best regressor (catboost)
    best_regressor.fit(x_train_normalized, y_train, silent = True)

    # making predictions using trained model
    pred = best_regressor.predict(x_test_normalized)
    compare = pd.DataFrame({'y_cv': y_cv, 'pred': pred})
    
    # creating subplots for true vs predicted, and residual plot
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


# #### Residual plot for tuned regressors (for log-tranformed target data)

# In[91]:


# log-transformed (target) datasets and dataset names
datasets = [routine_reg, emergency_reg, void_reg, other_reg , planned_reg, cyclical_reg, inspection_reg]
dataset_names = ['routine', 'emergency', 'void', 'other', 'planned', 'cyclical', 'inspection']

# looping through all priority datasets
for dataset_name, dataset in zip(dataset_names, datasets):
    print(f'plots for {dataset_name}')
    
    # preparing predictors and target
    predictors = dataset.drop(['repair_count','priority'], axis=1)
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

    # fitting best regressor (catboost)
    best_regressor.fit(x_train_normalized, y_train, silent = True)

    # making predictions using trained model
    pred = best_regressor.predict(x_test_normalized)
    compare = pd.DataFrame({'y_cv': y_cv, 'pred': pred})
    
    # creating subplots for true vs predicted, and residual plot
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


# #### Training best models again and saving
# Getting predicted values for comparison with true values
# ##### Observe comparison in output of cell

# In[99]:


datasets = [routine_reg, emergency_reg, planned_reg, cyclical_reg, void_reg, inspection_reg, other_reg]

# List to store trained models
stored_models = []

# creating instance of best regressor
best_regressor = CatBoostRegressor(silent = True)

# list to compare actual and predicted values
regressor_comparison_dataframes = []

# iterating over datasets
for dataset_name, dataset in zip(['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other'], datasets):

    # features and target
    predictors = dataset.drop(['repair_count', 'priority'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)

    # fit-transforming encoded training data
    x_train_normalized = scaler.fit_transform(X_train_encoded)

    # transforming testing data using fitted scaler
    x_test_normalized = scaler.transform(X_test_encoded)
    
    # fitting the current model on normalized data
    current_model = best_regressor
    current_model.fit(x_train_normalized, y_train)  
    
    # storing trained model
    stored_models.append((dataset_name, current_model))  
    
    # getting predictions of current model on test data
    y_pred = current_model.predict(x_test_normalized)
    
    # inverse log transformation of predictions
    y_pred_real = np.expm1(y_pred)
    rounded_pred = np.round(y_pred_real).astype(int)
    rounded_pred[rounded_pred < 0] = 0

    # inverse log transformation of actual values
    y_cv_real = np.expm1(y_cv)
    
    # comparing predicted values with actual values
    compare = {'y_cv': y_cv_real.reset_index(drop = True), 'pred': rounded_pred}
    df = pd.DataFrame(compare)
    
    # adding comparison dataframe for current priority dataset to our final list
    regressor_comparison_dataframes.append(df)

# displaying all comparison dataframes (for all priority types)
for idx, df in enumerate(regressor_comparison_dataframes):
    print(f'Predictions for regressor dataset - '{['routine', 'emergency', 'other', 'planned', 'inspection', 'void','cyclical'][idx]}':')
    display(df)
    print('\n')


# ### Shap Analysis of final models

# In[100]:


# using stored models to do SHAP analysis
stored_models


# In[101]:


# iterating over stored models and datasets 
for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'SHAP for {dataset_name}')
    
    # features and target
    predictors = dataset.drop(['repair_count', 'priority'], axis=1)
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
    # titles for all plots - by current dataset
    plt.gca().add_artist(plt.text(0.5, 1.08, f'Feature Importance plots for model trained on dataset - '{dataset_name}'', ha='center', va='center', transform=plt.gca().transAxes))

   # plotting feature importance plots 
    shap.plots.bar(shap_values, show = True)


# In[107]:


# summary plots

for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'summary plot for {dataset_name}')
    
    # splitting the dataset into features and target
    predictors = dataset.drop(['repair_count', 'priority'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)
    
    explainer = shap.Explainer(model)
    data_for_explain = X_test_encoded  
    
    shap_values = explainer(data_for_explain)
    # title for summary plot for current dataset
    plt.gca().add_artist(plt.text(0.5, 1.08, f'Summary plot for model trained on dataset - '{dataset_name}'', ha='center', va='center', transform=plt.gca().transAxes))
    # summary plot
    shap.summary_plot(shap_values)


# In[111]:


# waterfall plots

for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'waterfall plot for {dataset_name}')
    
    predictors = dataset.drop([ 'repair_count', 'priority',], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)
    X_test_encoded = one_hot_encoder(x_cv, cat_cols).reindex(columns=X_train_encoded.columns, fill_value=0)
   
    # explaining current model
    explainer = shap.Explainer(model)
    data_for_explain = X_test_encoded 
    shap_values = explainer(data_for_explain)
    
    # waterfall plot for the first datapoint
    index_to_plot = 0
    
    # title for waterfall plot for current dataset
    plt.gca().add_artist(plt.text(0.5, 1.08, f'Waterfall plot for model trained on dataset - '{dataset_name}'', ha='center', va='center', transform=plt.gca().transAxes))
    # plotting waterfall plot
    shap.plots.waterfall(shap_values[index_to_plot], max_display=10, show = True)  # Adjust 'max_display' as needed
  


# In[110]:


# force plots

for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'force plot for {dataset_name}')
    
    # features and target
    predictors = dataset.drop(['repair_count', 'priority',], axis=1)
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


# ### Saving models on system

# In[96]:


datasets = [routine_reg, emergency_reg, void_reg, other_reg , planned_reg, cyclical_reg, inspection]
#best_regressor = CatBoostRegressor(silent = True)

# not choosing classifiers for predicting repair demand for repair priority types based on property data
# previously experimented with deploying classifiers for minority classes
regressors = [best_regressor] * 7
classifiers = [None] * 7 

# regressors = [best_regressor] * 6 + [None] * 1
# classifiers = [None] * 6 + [best_classifier] * 1

comparison_dataframes = []

# dataset names
dataset_names = ['routine', 'emergency', 'void', 'other', 'planned', 'cyclical', 'inspection']

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
    predictors = dataset.drop(['repair_count','priority'], axis=1)
    target = dataset['repair_count']
    
    # split data into training and test set
    x_train, _, y_train, _ = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # onehot encoding categorical columns
    X_train_encoded = one_hot_encoder(x_train, cat_cols)

    # fit-transforming encoded training data
    x_train_normalized = scaler.fit_transform(X_train_encoded)
    
    # ftting the model
    trained_model.fit(x_train_normalized, y_train)
    
    # asssign model name
    model_name = f'{model_type}_{dataset_name}_priority_prop_model'
    
    # saving trained model using joblib
    model_filename = f'{model_name}.joblib'
    joblib.dump(trained_model, model_filename)
    
    
print('All models saved')


# ##### The trained models are saved as separate joblib files in the same directory where this Jupyter Notebook is located. These models are later loaded for use in a separate notebook for creating the Dash based web-application. (Please see 'Dash GCH App Final')
