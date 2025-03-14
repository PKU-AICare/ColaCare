# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from ehr_datasets.utils.tools import forward_fill_pipeline, normalize_dataframe

data_dir = "./ehr_datasets/esrd/"
Path(os.path.join(data_dir, 'processed')).mkdir(parents=True, exist_ok=True)

SEED = 42

# %%
basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome']
labtest_features = ['Cl', 'CO2CP', 'WBC', 'Hb', 'Urea', 'Ca', 'K', 'Na', 'Scr', 'P', 'Albumin', 'hs-CRP', 'Glucose', 'Appetite', 'Weight', 'SBP', 'DBP']
demographic_features = []
require_impute_features = labtest_features
normalize_features = labtest_features

# %%
df = pd.read_csv(os.path.join(data_dir, "processed", "esrd_dataset.csv"))
df.head(5)

# %%
grouped = df.groupby('PatientID')
len(grouped)

# %%
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

train_val_patients, test_patients = train_test_split(patients, test_size=20/100, random_state=SEED, stratify=patients_outcome)

train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=20/80, random_state=SEED, stratify=train_val_patients_outcome)

# %%
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]
len(train_patients), len(val_patients), len(test_patients)

# %%
save_dir = os.path.join(data_dir, 'processed', 'fold_1')
os.makedirs(save_dir, exist_ok=True)

# %%
def save_record_time(df: pd.DataFrame):
    record_times = []
    grouped_df = df.groupby('PatientID')
    for group in grouped_df.groups:
        record_times.append(grouped_df.get_group(group)['RecordTime'].values.tolist())
    return record_times
pd.to_pickle(save_record_time(train_df), os.path.join(save_dir, 'train_record_time.pkl'))
pd.to_pickle(save_record_time(val_df), os.path.join(save_dir, 'val_record_time.pkl'))
pd.to_pickle(save_record_time(test_df), os.path.join(save_dir, 'test_record_time.pkl'))

# %%
test_raw_x, test_raw_y, test_raw_pid = forward_fill_pipeline(test_df, test_df[normalize_features].median(), demographic_features, labtest_features, target_features, require_impute_features)
pd.to_pickle(test_raw_x, os.path.join(save_dir, "test_raw_x.pkl"))
val_raw_x, val_raw_y, val_raw_pid = forward_fill_pipeline(val_df, val_df[normalize_features].median(), demographic_features, labtest_features, target_features, require_impute_features)
pd.to_pickle(val_raw_x, os.path.join(save_dir, "val_raw_x.pkl"))

# %%
# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, _, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

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
pd.to_pickle(labtest_features, os.path.join(save_dir, "labtest_features.pkl"))

# %%
pd.to_pickle(df.groupby('Outcome').get_group(0).describe().to_dict('dict'), os.path.join(save_dir, 'survival.pkl'))
pd.to_pickle(df.groupby('Outcome').get_group(1).groupby('PatientID').last().describe().to_dict('dict'), os.path.join(save_dir, 'dead.pkl'))