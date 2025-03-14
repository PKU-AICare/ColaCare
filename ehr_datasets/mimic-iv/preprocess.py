# %% [markdown]
# ## Import packages

# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from ehr_datasets.utils.tools import forward_fill_pipeline, normalize_dataframe, normalize_df_with_statistics

data_dir = "./ehr_datasets/mimic-iv/"
Path(os.path.join(data_dir, 'processed')).mkdir(parents=True, exist_ok=True)

SEED = 42

# %% [markdown]
# ## Read data from files

# %% [markdown]
# ### Record feature names

# %%
basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS', 'Readmission']
demographic_features = ['Sex', 'Age'] # Sex and ICUType are binary features, others are continuous features
labtest_features = ['Capillary refill rate->0.0', 'Capillary refill rate->1.0',
        'Glascow coma scale eye opening->To Pain',
        'Glascow coma scale eye opening->3 To speech',
        'Glascow coma scale eye opening->1 No Response',
        'Glascow coma scale eye opening->4 Spontaneously',
        'Glascow coma scale eye opening->None',
        'Glascow coma scale eye opening->To Speech',
        'Glascow coma scale eye opening->Spontaneously',
        'Glascow coma scale eye opening->2 To pain',
        'Glascow coma scale motor response->1 No Response',
        'Glascow coma scale motor response->3 Abnorm flexion',
        'Glascow coma scale motor response->Abnormal extension',
        'Glascow coma scale motor response->No response',
        'Glascow coma scale motor response->4 Flex-withdraws',
        'Glascow coma scale motor response->Localizes Pain',
        'Glascow coma scale motor response->Flex-withdraws',
        'Glascow coma scale motor response->Obeys Commands',
        'Glascow coma scale motor response->Abnormal Flexion',
        'Glascow coma scale motor response->6 Obeys Commands',
        'Glascow coma scale motor response->5 Localizes Pain',
        'Glascow coma scale motor response->2 Abnorm extensn',
        'Glascow coma scale total->11', 'Glascow coma scale total->10',
        'Glascow coma scale total->13', 'Glascow coma scale total->12',
        'Glascow coma scale total->15', 'Glascow coma scale total->14',
        'Glascow coma scale total->3', 'Glascow coma scale total->5',
        'Glascow coma scale total->4', 'Glascow coma scale total->7',
        'Glascow coma scale total->6', 'Glascow coma scale total->9',
        'Glascow coma scale total->8',
        'Glascow coma scale verbal response->1 No Response',
        'Glascow coma scale verbal response->No Response',
        'Glascow coma scale verbal response->Confused',
        'Glascow coma scale verbal response->Inappropriate Words',
        'Glascow coma scale verbal response->Oriented',
        'Glascow coma scale verbal response->No Response-ETT',
        'Glascow coma scale verbal response->5 Oriented',
        'Glascow coma scale verbal response->Incomprehensible sounds',
        'Glascow coma scale verbal response->1.0 ET/Trach',
        'Glascow coma scale verbal response->4 Confused',
        'Glascow coma scale verbal response->2 Incomp sounds',
        'Glascow coma scale verbal response->3 Inapprop words',
        'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose',
        'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation',
        'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight',
        'pH']
require_impute_features = labtest_features
normalize_features = ['Age'] + ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose',
        'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation',
        'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight',
        'pH'] + ['LOS']

# %%
df = pd.read_csv(os.path.join(data_dir, "processed", f"format_mimic4_ehr.csv"))
df.head(5)

# %%
# if a patient has multiple records, we only use the first 48 items
# we also discard the patients with less than 48 items

# Ensure dataframe is sorted by PatientID and RecordTime
df = df.sort_values(['PatientID', 'RecordTime'])

# Filter out patients with less than 48 records
df = df.groupby('PatientID').filter(lambda x: len(x) >= 48)

# Select the first 48 records for each patient
df = df.groupby('PatientID').head(48)

# %% [markdown]
# ## Stratified split dataset into `Training`, `Validation` and `Test` sets
# 
# - Stratified dataset according to `Outcome` column
# - 90% Training, 5% Validation, 5% Test
#   - Name: train, val, test
# 

# %%
# Group the dataframe by patient ID
grouped = df.groupby('PatientID')
len(grouped)

# %%
# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# Get the train_val/test patient IDs
train_val_patients, test_patients = train_test_split(patients, test_size=5/100, random_state=SEED, stratify=patients_outcome)

# Get the train/val patient IDs
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=5/95, random_state=SEED, stratify=train_val_patients_outcome)


# %%
len(train_patients), len(val_patients), len(test_patients)

# %%
# Create train, val, test, [traincal, calib] dataframes for the current fold
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]

# %%
save_dir = os.path.join(data_dir, 'processed', 'fold_1') # forward fill
os.makedirs(save_dir, exist_ok=True)

# %%
# # Save the train, val, and test dataframes for the current fold to csv files
# train_df.to_csv(os.path.join(save_dir, "train_raw.csv"), index=False)
# val_df.to_csv(os.path.join(save_dir, "val_raw.csv"), index=False)
# test_df.to_csv(os.path.join(save_dir, "test_raw.csv"), index=False)
test_raw_x, test_raw_y, test_raw_pid = forward_fill_pipeline(test_df, test_df[normalize_features].median(), demographic_features, labtest_features, target_features, require_impute_features)
pd.to_pickle(test_raw_x, os.path.join(save_dir, "test_raw_x.pkl"))
val_raw_x, val_raw_y, val_raw_pid = forward_fill_pipeline(val_df, val_df[normalize_features].median(), demographic_features, labtest_features, target_features, require_impute_features)
pd.to_pickle(val_raw_x, os.path.join(save_dir, "val_raw_x.pkl"))

# %%
# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# # Save the zscored dataframes to csv files
# train_df.to_csv(os.path.join(save_dir, "train_after_zscore.csv"), index=False)
# val_df.to_csv(os.path.join(save_dir, "val_after_zscore.csv"), index=False)
# test_df.to_csv(os.path.join(save_dir, "test_after_zscore.csv"), index=False)

# Forward Imputation after grouped by PatientID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

# Save the imputed dataset to pickle file
pd.to_pickle(train_x, os.path.join(save_dir, "train_x.pkl"))
pd.to_pickle(train_y, os.path.join(save_dir, "train_y.pkl"))
pd.to_pickle(train_pid, os.path.join(save_dir, "train_pid.pkl"))
pd.to_pickle(val_x, os.path.join(save_dir, "val_x.pkl"))
pd.to_pickle(val_y, os.path.join(save_dir, "val_y.pkl"))
pd.to_pickle(val_pid, os.path.join(save_dir, "val_pid.pkl"))
pd.to_pickle(test_x, os.path.join(save_dir, "test_x.pkl"))
pd.to_pickle(test_y, os.path.join(save_dir, "test_y.pkl"))
pd.to_pickle(test_pid, os.path.join(save_dir, "test_pid.pkl"))
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl")) # LOS statistics (calculated from the train set)
pd.to_pickle(['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation','Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH'], os.path.join(save_dir, 'labtest_features.pkl'))

# %%
len(test_pid), len(val_pid), len(train_pid)

# %%
pd.to_pickle(df.groupby('Outcome').get_group(0).describe().to_dict('dict'), os.path.join(save_dir, 'survival.pkl'))
pd.to_pickle(df.groupby('Outcome').get_group(1).describe().to_dict('dict'), os.path.join(save_dir, 'dead.pkl'))

# %%
pd.to_pickle(df[['PatientID', 'Sex', 'Age']].groupby('PatientID').first().to_dict('index'), os.path.join(data_dir, 'processed', 'fold_1', 'basic.pkl'))

# %%
# for fold in range(folds):
fold = 1
data = []
save_dir = os.path.join(data_dir, 'processed', f'fold_{fold}')
for mode in ['train', 'val', 'test']:
    x = pd.read_pickle(os.path.join(save_dir, f"{mode}_x.pkl"))
    y = pd.read_pickle(os.path.join(save_dir, f"{mode}_y.pkl"))
    pid = pd.read_pickle(os.path.join(save_dir, f"{mode}_pid.pkl"))
    data.extend([(pid[i], x[i], y[i]) for i in range(len(x))])
    
sort_data = sorted(data, key=lambda x: x[0])
all_pid = [x[0] for x in sort_data]
all_x = [x[1] for x in sort_data]
all_y = [x[2] for x in sort_data]

print(len(all_pid), len(all_x), len(all_y))

pd.to_pickle(all_pid, os.path.join(save_dir, "all_pid.pkl"))
pd.to_pickle(all_x, os.path.join(save_dir, "all_x.pkl"))
pd.to_pickle(all_y, os.path.join(save_dir, "all_y.pkl"))


