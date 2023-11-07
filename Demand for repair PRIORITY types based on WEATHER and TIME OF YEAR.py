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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, recall_score, r2_score, classification_report
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
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
import joblib
from sklearn.base import clone


# In[2]:


# getting uk holiday dates (for holiday varaible needed to execute this project objective - one of the potential predictors)
uk_holidays = holidays.UnitedKingdom()
# print all the holidays in UnitedKingdom in year 2023 for demonstration purpose
for ptr in holidays.UnitedKingdom(years = 2023).items():
    print(ptr)


# #### Loading datasets needed to execute this project objective

# In[3]:


job = pd.read_excel(r'..\UPDATED DATA\04. Repairs\Job.xlsx')
pty_codes = pd.read_excel(r'..\UPDATED DATA\04. Repairs\Pty.xlsx')
sor = pd.read_excel(r'..\UPDATED DATA\04. Repairs\SORTrd.xlsx')
# weather data in Gloucester from 27 Feb 1996 to 15 June 2023
gl = pd.read_excel(r'..\UPDATED DATA\Weather Data\Gloucester.xlsx')
# weather codes from WMO - for states of sky - CLOUDY, CLEAR, RAIN, LIGHT DRIZZLE, SNOW etc
wmo = pd.read_csv(r'..\UPDATED DATA\wmo_codes.csv', header = None)


# ##### Merging job with priority type and replacing old SOR Trade codes with new ones

# In[4]:


pty_type = pty_codes[['pty_cde','pty_type']]
job['job_report_date'] = pd.to_datetime(job['reported-dat']).dt.date
job = job.merge(pty_type, how='left', left_on='pty-cde', right_on='pty_cde')
# mapping priority code of each report with priority classifcation of repair (routine, cyclic, emergency, void)
pty_map = dict(pty_codes[['pty_cde','pty_classification']].values)
job['priority'] = job['pty-cde'].map(pty_map)
# droppin pty_cde column 
job.drop('pty_cde', axis = 1, inplace = True)
# replacing OLD SOR trade codes with NEW ones
job['sortrd-cde-1'].replace(['BR', 'C', 'E', 'E1', 'F', 'G','GF', 'H', 'MI', 'P', 'PD', 'PO','R', 'SC', 'TI', 'W'], 
                             ['0B','0C','0E', '0E', '0F','0Z','0G','HP','NS','0P', '0D','0D','0R', '0S', '0I','0C'], 
                            inplace=True)
# mapping SOR Trade codes to their descriptions
sor.set_index('cde', inplace = True)
sor_map = sor.to_dict()['dsc']
job['sor'] = job['sortrd-cde-1'].map(sor_map)
job.head()


# #### Calculating number of 'report counts' for each date in the dataframe

# In[5]:


job1 = job[['job_report_date', 'priority']]
job_counts = job1.groupby(['job_report_date', 'priority']).size().reset_index(name='repair_count')
job_counts = job_counts.iloc[1:]
job_counts


# ##### getting repair counts for each type of repair priority for each date

# In[6]:


# converting job_report_date to datetime
job_counts['job_report_date'] = pd.to_datetime(job_counts['job_report_date'])
# defining the date range
start_date = pd.to_datetime('1996-02-27')
end_date = pd.to_datetime('2023-06-15')
date_range = pd.date_range(start=start_date, end=end_date)

# creating a DataFrame with all date-priority combinations
priorities = ['Emergency Repair', 'Inspection', 'Other', 'Routine Repair', 'Void Works', 'Planned Work', 'Cyclical Works']
date_priority_combinations = []
for date in date_range:
    for priority in priorities:
        date_priority_combinations.append({'job_report_date': date, 'priority': priority})

all_combinations_df = pd.DataFrame(date_priority_combinations)
# merging the original job_counts DataFrame with all_combinations_df
all_job_comb = pd.merge(all_combinations_df, job_counts, on=['job_report_date', 'priority'], how='left')

# filling NaN values in repair_count with 0
all_job_comb['repair_count'].fillna(0, inplace=True)
all_job_comb


# In[7]:


all_jobs = all_job_comb[['job_report_date', 'priority', 'repair_count']]

all_jobs['Year'] = pd.to_datetime(all_jobs['job_report_date']).dt.year
all_jobs['Week'] = pd.to_datetime(all_jobs['job_report_date']).dt.week
all_jobs['Month'] = pd.to_datetime(all_jobs['job_report_date']).dt.month
all_jobs['Day'] = pd.to_datetime(all_jobs['job_report_date']).dt.day

all_jobs['WeekDay'] = pd.to_datetime(all_jobs['job_report_date']).dt.dayofweek
all_jobs['Holiday'] = all_jobs['job_report_date'].isin(uk_holidays)
all_jobs['BeginMonth']=all_jobs.Day.isin([1,2,3]).astype(int)
all_jobs['Weekend']=all_jobs.WeekDay.isin([5,6]).astype(int)

all_jobs.head()


# ### Merging with weather data

# ##### Mapping weather codes to Gloucester weather data

# In[8]:


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


# ###### WMO code = 2 means that the weather at present is the same as last recorded weather (today is same as yesterday since this is daily weather data)
# Replace 2 with previous weather condition

# In[9]:


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


# #### Merge weather dataframe to mainframe. Join on common date column, so we have weather info as well as repair count for each date

# In[10]:


# converting DATE values in WEATHER dataset to DATETIME type for easy merging with REPAIR dataset
gl_updated.time = gl_updated.time.apply(lambda x: x.date())
gl_updated['time'] = pd.to_datetime(gl_updated['time'])
job_unique_date_weather = all_jobs.merge(gl_updated, how='inner', left_on='job_report_date', right_on='time')

# dropping numerical weather code and only keeping corresponding textual code to later encode
job_unique_date_weather = job_unique_date_weather.drop('weathercode (wmo code) modified', axis = 1)

# making Boolean value into integer
job_unique_date_weather['Holiday'] = job_unique_date_weather['Holiday'].apply(int) 
job_unique_date_weather


# #### Isolating data for each repair priority type

# In[11]:


routine = job_unique_date_weather[job_unique_date_weather['priority']=='Routine Repair']
emergency = job_unique_date_weather[job_unique_date_weather['priority']=='Emergency Repair']
void = job_unique_date_weather[job_unique_date_weather['priority']=='Void Works']
planned = job_unique_date_weather[job_unique_date_weather['priority']=='Planned Work']
cyclical = job_unique_date_weather[job_unique_date_weather['priority']=='Cyclical Works']
other = job_unique_date_weather[job_unique_date_weather['priority']=='Other']
inspection = job_unique_date_weather[job_unique_date_weather['priority']=='Inspection']


# #### Checking for missing values

# In[12]:


for feature in all_jobs.columns.values:
    print('#####', feature, '-----number missing', all_jobs[feature].isnull().sum())
    
    


# NO MISSING VALUES

# ### Experiment - Using Classifiers for predicting demand for each priority type

# In[13]:


# loading all priority datasets
datasets = {
    'routine': routine,  
    'emergency': emergency,
    'void': void,
    'other': other,
    'cyclical': cyclical,
    'inspection': inspection,
    'planned': planned
}

# declaring classifiers to be used
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'LightGBM': LGBMClassifier(random_state=42)
}

# list for stroing performance metrics
all_metrics = []

# iterating over each dataset
for dataset_name, dataset in datasets.items():
    print(f'Training classifiers for dataset: {dataset_name}')
    
    # predictors and target
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    job_date_weather_target = dataset['repair_count']

    # splitting data into training and testing sets
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

    # storing performance scores of each classifier for current dataset
    dataset_metrics = []

    # iterating over classifiers, train, predict, and calculate metrics
    for name, classifier in classifiers.items():
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_cv)
        accuracy = accuracy_score(y_cv, y_pred)
        f1 = f1_score(y_cv, y_pred, average='weighted')
        f1_macro = f1_score(y_cv, y_pred, average='macro')
        rmse = np.sqrt(mean_squared_error(y_cv, y_pred))
        recall = recall_score(y_cv, y_pred, average='weighted', zero_division=1)

        dataset_metrics.append({'Classifier': name, 'Accuracy': accuracy, 'F1 Score': f1, 'F1 Macro':f1_macro, 'RMSE': rmse, 'Recall': recall})

    # storing metrics for this dataset
    all_metrics.extend([(dataset_name, metrics) for metrics in dataset_metrics])


# In[14]:


all_metrics


# In[15]:


# creating lists to store performance metrics of all classifiers for all priority types
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

# final dataframe
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


# In[16]:


# sorting the metrics by accuracy
sorted_metrics_df = metrics_df.sort_values(by='Accuracy')

# get metrics of each classifier
all_classifier_scores = sorted_metrics_df.groupby('Classifier')

# plotting accuracy and F1 score for each type of classifier (trained on each priority dataset)
for classifier, classifier_scores in all_classifier_scores:
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


# #### Identify best classifier

# In[17]:


# grouping the dataframe by 'Classifier' and calculating mean accuracy for each classifier
classifier_accuracy = metrics_df.groupby('Classifier')['Accuracy'].mean()
classifier_f1_macro = metrics_df.groupby('Classifier')['F1 Macro'].mean()

# classifier with the highest mean accuracy
best_classifier = classifier_accuracy.idxmax()
highest_accuracy = classifier_accuracy.max()
highest_f1_macro = classifier_f1_macro.max()

print(f"The best performing classifier is '{best_classifier}' with an average accuracy of {highest_accuracy:.6f} with an average accuracy of {highest_accuracy:.6f} and highest F1 Macro Score of {highest_f1_macro:.6f}")


# In[18]:


best_classifier_scores = metrics_df[metrics_df['Classifier'] == 'CatBoost']
best_classifier_scores = best_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
best_classifier_scores


# In[19]:


catboost_classifier_scores = metrics_df[metrics_df['Classifier'] == 'CatBoost']
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


# #### Oversampling

# In[21]:


# loading all datasets and classifiers for training 
# subsequently evalauting each classifier's performance for each priority dataset
datasets = {
    'routine': routine, 
    'emergency': emergency,
    'void': void,
    'other': other,
    'cyclical': cyclical,
    'inspection': inspection,
    'planned': planned
}

classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'LightGBM': LGBMClassifier(random_state=42)
}

all_metrics_smt = []

# iterating over each dataset
for dataset_name, dataset in datasets.items():
    print(f'Training classifiers for dataset: {dataset_name}')
    
    
    
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
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
    all_metrics_smt.extend([(dataset_name, metrics) for metrics in dataset_metrics])


# In[22]:


# storing performance metrics of each classifier for each oversampled priority type dataset
datasets = []
classifiers = []
accuracies = []
f1_scores = []
f1_macro = []
rmse_scores = []
recalls = []

for dataset, metrics in all_metrics_smt:
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


# In[25]:


# identifying best classifier (after SMOTEing)
classifier_accuracy = metrics_smote_df.groupby('Classifier')['Accuracy'].mean()
classifier_f1_macro = metrics_smote_df.groupby('Classifier')['F1 Macro'].mean()

# finding classifier with the highest mean accuracy
best_classifier = classifier_accuracy.idxmax()
highest_accuracy = classifier_accuracy.max()
highest_f1_macro = classifier_f1_macro.max()

print(f'The best performing classifier after oversampling with SMOTE is \'{best_classifier}\' with an average accuracy of {highest_accuracy:.6f} and highest F1 Macro Score of {highest_f1_macro:.6f}')


# In[26]:


best_resampled_classifier_scores = metrics_smote_df[metrics_smote_df['Classifier'] == 'Random Forest']
best_resampled_classifier_scores = best_resampled_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
best_resampled_classifier_scores


# In[27]:


cat_smt_classifier_scores = metrics_smote_df[metrics_smote_df['Classifier'] == 'Random Forest']
cat_smt_classifier_scores = cat_smt_classifier_scores.sort_values(by = 'Accuracy', ascending = False)
plt.figure(figsize=(10, 6))
plt.title(f'CatBoost Classifier on Resampled data')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.plot(cat_smt_classifier_scores['Dataset'], cat_smt_classifier_scores['Accuracy'], label='Accuracy', marker='o')
plt.plot(cat_smt_classifier_scores['Dataset'], cat_smt_classifier_scores['F1 Macro'], label='F1 Macro', marker='o')

plt.legend()
plt.tight_layout()
plt.show()


# #### Conclusion: no significant improvement after oversampling. F1 Macro score remains equal to 0.44

# ##### Retraining classifiers and storing predictions

# In[28]:


datasets = {
    'routine': routine,  
    'emergency': emergency,
    'void': void,
    'other': other,
    'cyclical': cyclical,
    'inspection': inspection,
    'planned': planned
}


# dictionary to store predicted values and actual repair count values for each dataset
predictions_dict = {}

# iterating over each dataset
for dataset_name, dataset in datasets.items():
    print(f'Retraining CatBoost Classifier for dataset: {dataset_name}')

    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    job_date_weather_target = dataset['repair_count']
    
    # splitting data into train and test sets
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

    # retraining the best classifier
    catboost = CatBoostClassifier(random_state=42, silent = True)
    catboost.fit(x_train, y_train)
    
    # predictions on test data
    y_pred = catboost.predict(x_cv)
    
    # savng predicted values and actual values in the dictionary
    predictions_dict[dataset_name] = {'Actual': y_cv, 'Predicted': y_pred}


# In[29]:


# check predicting performance of classifiers
# creating scatter plot for randomly selecting datapoints (actual vs predicted repair count)
num_samples = 500

# Iterate over each dataset
for dataset_name, predictions in predictions_dict.items():
    actual_values = predictions['Actual']
    predicted_values = predictions['Predicted']
    
    # Reset indices of the actual_values DataFrame
    actual_values = actual_values.reset_index(drop=True)
    
    # Randomly sample data points for plotting
    max_samples = min(len(actual_values), num_samples)  # Limit samples to available data length
    sample_indices = random.sample(range(len(actual_values)), max_samples)
    sampled_actual = [actual_values[i] for i in sample_indices]
    sampled_predicted = [predicted_values[i] for i in sample_indices]
    
    # Create a scatter plot of sampled actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(sampled_actual, sampled_predicted, color='blue', alpha=0.5)
    plt.title(f'Sampled Actual vs Predicted for {dataset_name} Dataset')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.xlim(min(sampled_actual), max(sampled_actual))
    plt.ylim(min(sampled_predicted), max(sampled_predicted))
    plt.plot([min(sampled_actual), max(sampled_actual)], [min(sampled_actual), max(sampled_actual)], color='red', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.show()


# In[30]:


# get classification report for CatBoost classifier trained on each priority dataset
# iterating over each dataset
for dataset_name, predictions in predictions_dict.items():
    actual_values = predictions['Actual']
    predicted_values = predictions['Predicted']
    
    # classification report
    cls_report = classification_report(actual_values, predicted_values)
    
    print(f'Classification Report for {dataset_name} Dataset:\n')
    print(cls_report)


# ### Experiment - Using Regressors for predicting demand for each priority type
# ##### Training models without normalization/standardization of numeric features. Only encoding categorical features

# #### Creating copies of datasets for each priority type for regression, and applying log-transformation on repair counts (target)

# Transforming target 'repair_count' for better regression over extreme values/ outlier values of repair count
# 

# In[31]:


pd.set_option('display.max_rows', None)


# In[32]:


cyclical_reg = cyclical.copy()
planned_reg = planned.copy()
routine_reg = routine.copy()
other_reg = other.copy()
emergency_reg = emergency.copy()
void_reg = void.copy()
inspection_reg = inspection.copy()


# applying log transformation to repair_count for each dataset
cyclical_reg['repair_count'] = np.log1p(cyclical_reg['repair_count'])
planned_reg['repair_count'] = np.log1p(planned_reg['repair_count'])
routine_reg['repair_count'] = np.log1p(routine_reg['repair_count'])
other_reg['repair_count'] = np.log1p(other_reg['repair_count'])
emergency_reg['repair_count'] = np.log1p(emergency_reg['repair_count'])
void_reg['repair_count'] = np.log1p(void_reg['repair_count'])
inspection_reg['repair_count'] = np.log1p(inspection_reg['repair_count'])


# In[33]:


# datasets for regression corresponding to each priority type: routine, emergency, planned, cyclical, void, inspection, other
# log Transformed datasets
datasets = [routine_reg, emergency_reg, planned_reg, cyclical_reg, void_reg, inspection_reg, other_reg]

scores = []
# list of regressors to experiment with
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

# list of metrics to calculate for evaluating regressors
metrics = {
    'MSE': mean_squared_error,
    'R2': r2_score,
}

# creating dataFrame to store scores
scores_df = pd.DataFrame(columns=['Method', 'Dataset'] + list(metrics.keys()))

# iterating over regressors
for regressor in regressors:
    for dataset_name, dataset in zip(['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other'], datasets):
        print(f'Training {regressor.__name__} on {dataset_name}')

        # prediciting features and target
        job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified','temperature_2m_min (°C)','temperature_2m_max (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
        job_date_weather_target = dataset['repair_count']
        x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

        # One-hot encoding categorical columns
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
        metrics_reg = {'Method': regressor.__name__, 'Dataset': dataset_name, **dict(zip(list(metrics.keys()), metrics_scores))}
        scores.append(metrics_reg)

# dataframe for storing evalaution metric scores for each regressor
scores_df = pd.concat([pd.DataFrame([score]) for score in scores], ignore_index=True)


# In[34]:


scores_df


# In[35]:


pd.reset_option('max_rows')


# In[36]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.table(cellText=scores_df.values, colLabels=scores_df.columns, cellLoc='center', loc='center')
plt.savefig('scores_table.png', bbox_inches='tight', pad_inches=0.5)
plt.show()


# In[37]:


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


# In[38]:


# getting best regressor
regressor_r2 = scores_df.groupby('Method')['R2'].mean()
# regressor with highest R2 score
best_regressor = regressor_r2.idxmax()
highest_r2 = regressor_r2.max()

print(f'The best performing Regressor is \'{best_regressor}\' with an average R2 score of {highest_r2:.6f}')


# In[44]:


lgbm_scores = scores_df[scores_df['Method'] == 'LGBMRegressor']
mse_lgbm = lgbm_scores['MSE'].mean()
print(f'The best performing Regressor is \'{best_regressor}\' with an average MSE score of {mse_lgbm:.6f}')
lgbm_scores


# In[45]:


# sorting lgbm scores
lgbm_scores_sorted = lgbm_scores.sort_values(by = 'R2')
lgbm_scores_sorted
# plotting R2 squared score for each type of priority
plt.figure(figsize=(10, 6))
plt.title(f'LGBM Regressor - R2')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.plot(lgbm_scores_sorted['Dataset'], lgbm_scores_sorted['R2'], label='R2', marker='o')
plt.legend()
plt.show()


# #### Experiment - Regression without log transforming target variable

# In[46]:


# datasets for regression corresponding to each priority type: routine, emergency, planned, cyclical, void, inspection, other
# Non-log Transformed datasets
datasets = [routine, emergency, planned, cyclical, void, inspection, other]

scores = []
# list of regressors to experiment with
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

# list of metrics to calculate for evaluating regressors
metrics = {
    'MSE': mean_squared_error,
    'R2': r2_score,
}

# creating dataFrame to store scores
scores_df_no_transformation = pd.DataFrame(columns=['Method', 'Dataset'] + list(metrics.keys()))

# iterating over regressors
for regressor in regressors:
    for dataset_name, dataset in zip(['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other'], datasets):
        print(f'Training {regressor.__name__} on {dataset_name}')

        # prediciting features and target
        job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified','temperature_2m_min (°C)','temperature_2m_max (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
        job_date_weather_target = dataset['repair_count']
        x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

        # One-hot encoding categorical columns
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
        metrics_reg = {'Method': regressor.__name__, 'Dataset': dataset_name, **dict(zip(list(metrics.keys()), metrics_scores))}
        scores.append(metrics_reg)

# dataframe for storing evalaution metric scores for each regressor
scores_df_no_transformation = pd.concat([pd.DataFrame([score]) for score in scores], ignore_index=True)


# In[47]:


# getting best regressor trained on non-transformed data
regressor_r2 = scores_df_no_transformation.groupby('Method')['R2'].mean()
# regressor with highest R2 score
best_regressor = regressor_r2.idxmax()
highest_r2 = regressor_r2.max()

print(f'The best performing Regressor on non-transformed datasets is \'{best_regressor}\' with an average R2 score of {highest_r2:.6f}')


# In[49]:


nt_scores = scores_df_no_transformation[scores_df_no_transformation['Method'] == 'CatBoostRegressor']
mse_lgbm = nt_scores['MSE'].mean()
print(f'The best performing Regressor is \'{best_regressor}\' with an average MSE score of {mse_lgbm:.6f}\n\n')
nt_scores


# #### Conclusion - Log-transformation improves model performance

# ##### True vs predicted values on CatBoost regressor for all non-transformed priority types

# In[52]:


datasets = [routine, emergency, planned, cyclical, void, inspection, other]
dataset_names = ['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other']
best_regressor = CatBoostRegressor(silent = True)


# Loop through datasets
for dataset_name, dataset in zip(dataset_names, datasets):
    print(f'plots for {dataset_name}')
    
    # prepare predictors
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority',
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
    compare = pd.DataFrame({'y_cv': y_cv, 'pred': np.round(pred, 0)})
    
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


# ##### True vs predicted values on LGBM regressor for all log-transformed priority types

# In[56]:


datasets = [routine_reg, emergency_reg, planned_reg, cyclical_reg, void_reg, inspection_reg, other_reg]
dataset_names = ['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other']
best_regressor = CatBoostRegressor(silent = True)


# Loop through datasets
for dataset_name, dataset in zip(dataset_names, datasets):
    print(f'plots for {dataset_name}')
    
    # prepare predictors
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority',
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


# #### Tuning regressors
# with log-transformation of target variable

# Storing tuned regressors and predictions

# In[58]:


# dictionary to store predictions by lgbm regressor on each priority dataset (log transformed traget)
predictions_dict_regressor = {}

datasets = [routine_reg, emergency_reg, planned_reg, cyclical_reg, void_reg, inspection_reg, other_reg]

# defining the parameter grid for tuning
param_grid = {
    'num_leaves': [10, 20, 30],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

# creating an instance of LGBM Regressor
lgbm_regressor = LGBMRegressor(random_state=42, silent = True)

# creating GridSearchCV instance
grid_search = GridSearchCV(lgbm_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')

# iterating over each dataset
for dataset_name, dataset in zip(['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other'], datasets):
    print(f'Tuning LGBM Regressor on {dataset_name}')
    
    # features and target
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    job_date_weather_target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

    # one-hot encoding categorical columns (same as before)
    s = (x_train.dtypes == 'object')
    object_cols = list(s[s].index)

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
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
    
    # getiing best parameters
    best_params = grid_search.best_params_
    
    # using best parameters to retrain LGBM Regressor
    best_lgbm_regressor = LGBMRegressor(**best_params, random_state=42)
    best_lgbm_regressor.fit(OH_X_train, y_train)
    
    # making predictions on test data
    y_pred = best_lgbm_regressor.predict(OH_X_cv)
    y_pred_real = np.expm1(y_pred)
    rounded_pred = np.round(y_pred_real).astype(int)
    rounded_pred[rounded_pred < 0] = 0

    # inverse log transformation of true values
    y_cv_real = np.expm1(y_cv)
    
    # calculating R2 and MSE scores
    r2_score_val = r2_score(y_cv, y_pred)
    mse_score_val = mean_squared_error(y_cv, y_pred)
    
    # saving best parameters, R2, and MSE scores in a dictionary
    best_results = {
        'Best Parameters': best_params,
        'R2 Score': r2_score_val,
        'MSE Score': mse_score_val
    }
    
    predictions_dict_regressor[dataset_name] = {'Actual': y_cv_real, 'Predicted': rounded_pred, 'Best Results': best_results}
    
    

print('Tuning complete')


# In[64]:


scores_after_tuning = {}

# iterating over each dataset
for dataset_name, dataset in zip(['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other'], datasets):
    # getting prediction results from predictions dictioanary for the current dataset
    results = predictions_dict_regressor[dataset_name]
    #print(results)
    # getting best results
    best_results = results.get('Best Results',{})
    r2_score_val = best_results.get('R2 Score', None)
    mse_score_val = best_results.get('MSE Score', None)
    
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


scores_tuned['R2 Score'].mean()


# In[66]:


scores_tuned['MSE Score'].mean()


# #### Conclusion: No significant improvement after tuning

# #### Combine plots - CatBoost Regressor & CatBoost Classifier

# In[68]:


# getting best regressor and best classifier scores for direct comparison
lgbm_regressor_scores = scores_df[scores_df['Method'] == 'LGBMRegressor']
catboost_classifier_scores = metrics_df[metrics_df['Classifier'] == 'CatBoost']
smt_classifier_scores = metrics_smote_df[metrics_smote_df['Classifier'] == 'Random Forest']
# sort scores by R2 for regressors and F1 Macro for classifiers
dataset_order_reg = lgbm_regressor_scores.sort_values(by='R2')
dataset_order_reg_tuned = scores_tuned.sort_values(by = 'R2 Score')
dataset_order_cls = catboost_classifier_scores.sort_values(by='F1 Macro', ascending=False)
dataset_order_cls_smt = smt_classifier_scores.sort_values(by='F1 Macro', ascending=False)
# reordering dataset_order_cls based on the order of dataset_order_reg
dataset_order_cls_reordered = dataset_order_cls.set_index('Dataset').loc[dataset_order_reg['Dataset']].reset_index()
dataset_order_cls_smt_reordered = dataset_order_cls_smt.set_index('Dataset').loc[dataset_order_reg['Dataset']].reset_index()
dataset_order_reg_tuned_reordered = dataset_order_reg_tuned.set_index('Dataset').loc[dataset_order_reg['Dataset']].reset_index()

# plotting combined plot
plt.figure(figsize=(10, 6))

plt.title('Scores - LGBM Regressor (Untuned, Tuned) & CatBoost Classifier & Random Forest Classifier (on oversampled data)')
plt.xlabel('Dataset')
plt.ylabel('Score')
plt.xticks(rotation=65)

plt.plot(dataset_order_reg['Dataset'], dataset_order_reg['R2'], label='R2 (LGBM Regressor)', marker='o')
plt.plot(dataset_order_cls_reordered['Dataset'], dataset_order_cls_reordered['F1 Macro'], label='F1 Macro (CatBoost Classifier)', marker='o')
plt.plot(dataset_order_cls_smt_reordered['Dataset'], dataset_order_cls_smt_reordered['F1 Macro'], label='F1 Macro (Random Forest Classifier (smt))', marker='o')
plt.plot(dataset_order_reg_tuned_reordered['Dataset'], dataset_order_reg_tuned_reordered['R2 Score'], label = 'R2 Tuned (LGBM Regressor)', marker = 'o')


plt.legend()
plt.tight_layout()
plt.show()


# #### Choosing best models

# Choosing LGBM regresor for predicting demand for all priority types 

# In[69]:


best_regressor = LGBMRegressor()

#best_classifier = CatBoostClassifier()


# #### Retraining and storing models for SHAP analysis
# Display predictions

# In[71]:


datasets = [routine_reg, emergency_reg, planned_reg, cyclical_reg, void_reg, inspection_reg, other_reg]

# List to store trained models
stored_models = []

# creating instance of best regressor
best_regressor = LGBMRegressor()

# list to compare actual and predicted values
regressor_comparison_dataframes = []

# iterating over datasets
for dataset_name, dataset in zip(['routine', 'emergency', 'planned', 'cyclical', 'void', 'inspection', 'other'], datasets):
    print(f'Training best regression model on {dataset_name}')

    # features and target
    job_date_weather_predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified','temperature_2m_max (°C)','temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    job_date_weather_target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(job_date_weather_predictors, job_date_weather_target, test_size=0.2, random_state=42)

    # One-hot encoding categorical columns
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
    
    # comparing predicted values with actual values
    compare = {'y_cv': y_cv_real.reset_index(drop = True), 'pred': rounded_pred}
    df = pd.DataFrame(compare)
    
    # adding comparison dataframe for current priority dataset to our final list
    regressor_comparison_dataframes.append(df)

print('models trained')

# displaying all comparison dataframes (for all priority types)
for idx, df in enumerate(regressor_comparison_dataframes):
    print(f"Predictions for regressor dataset - '{['routine', 'emergency', 'other', 'planned', 'inspection', 'void','cyclical'][idx]}':")
    display(df)
    print('\n')


# In[72]:


stored_models


# In[74]:


# feature importance plots
# iterating over the stored models and datasets 
for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'SHAP for {dataset_name}')
    
    # features and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # training LGBM regressor and getting feature importances
    model.fit(x_train, y_train)
    feature_importances = model.feature_importances_
    feature_names = x_train.columns

    # creating SHAP explainer
    explainer = shap.Explainer(model)
    
    # creating explanation data using the test data
    explain_data = x_cv 
    
    # getting SHAP values
    shap_values = explainer(explain_data)
    # creating bar plot for feature importances
    shap.plots.bar(shap_values, show = True)


# In[75]:


# waterfall plots

for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'waterfall plot for {dataset_name}')
    
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    explainer = shap.Explainer(model)
    data_for_explain = x_cv   
    shap_values = explainer(data_for_explain)
    
    # creating waterfall plot for first datapoint (in each priority dataset)
    index = 0
    # title
    plt.gca().add_artist(plt.text(0.5, 1.08, f"Waterfall plot for model trained on dataset - '{dataset_name}'", ha='center', va='center', transform=plt.gca().transAxes))

    shap.plots.waterfall(shap_values[index], max_display=10, show = True)  


# In[76]:


# summary plots

for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'summary plot for {dataset_name}')
    
    #features and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    explainer = shap.Explainer(model)
    data_for_explain = x_cv  
    
    shap_values = explainer(data_for_explain)
    # title
    plt.gca().add_artist(plt.text(0.5, 1.08, f"Summary plot for model trained on dataset - '{dataset_name}'", ha='center', va='center', transform=plt.gca().transAxes))
    # summary plot
    shap.summary_plot(shap_values)


# In[77]:


# force plots
for (dataset_name, model), dataset in zip(stored_models, datasets):
    print(f'force plot for {dataset_name}')
    
    # features and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    x_train, x_cv, y_train, y_cv = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    data_for_explain = x_cv  
    
    shap_values = explainer.shap_values(x_train.iloc[0:50, :]) 
    shap_plot = shap.force_plot(explainer.expected_value, shap_values, x_train.iloc[0:50, :])
    shap.save_html(f'force_plot_{dataset_name}.html', shap_plot )


# ### Saving models on system for deployment 

# In[80]:


# best regressor
best_regressor = LGBMRegressor()

# datasets 
regressor_datasets = [routine_reg, emergency_reg, other_reg, planned_reg, inspection_reg, void_reg, cyclical_reg]

# iterating over all datasets
for dataset, dataset_name in zip(regressor_datasets, ['routine', 'emergency', 'other',  'planned', 'inspection', 'void', 'cyclical']):
    print(f'Processing dataset: {dataset_name}')
    
    # predictors and target
    predictors = dataset.drop(['time', 'job_report_date', 'repair_count', 'priority', 'weather_condition modified', 'apparent_temperature_min (°C)', 'apparent_temperature_mean (°C)', 'apparent_temperature_max (°C)', 'winddirection_10m_dominant (°)'], axis=1)
    target = dataset['repair_count']
    
    # getting training data
    x_train, _, y_train, _ = train_test_split(predictors, target, test_size=0.2, random_state=42)
    
    # retraining best regressor
    trained_regressor = best_regressor
    trained_regressor.fit(x_train, y_train)
    
    # model name according to cuurent priority dataset
    model_name = f'regressor_priority_{dataset_name}_model_AUG'
    
    # saving trained model using joblib
    model_filename = f'{model_name}.joblib'
    joblib.dump(trained_regressor, model_filename)
    
    print(f'{model_name} trained and saved')

print('Models saved')


# ##### The trained models are saved as separate joblib files in the same directory where this Jupyter Notebook is located. These models are later loaded for use in a separate notebook for creating the Dash based web-application. (Please see 'Dash GCH App Final')

# In[ ]:




