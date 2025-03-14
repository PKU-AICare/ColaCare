# %% [markdown]
# ## Import packages

# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from ehr_datasets.utils.tools import forward_fill_pipeline, normalize_dataframe

data_dir = "./ehr_datasets/cdsl/"
Path(os.path.join(data_dir, 'processed')).mkdir(parents=True, exist_ok=True)

SEED = 42

# %% [markdown]
# ## Read data from files

# %% [markdown]
# ### Record feature names

# %%
basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS']
demographic_features = ['Sex', 'Age']
labtest_features = ['MAX_BLOOD_PRESSURE', 'MIN_BLOOD_PRESSURE', 'TEMPERATURE', 'HEART_RATE', 'OXYGEN_SATURATION', 'ADW -- Coeficiente de anisocitosis', 'ADW -- SISTEMATICO DE SANGRE', 'ALB -- ALBUMINA', 'AMI -- AMILASA', 'AP -- ACTIVIDAD DE PROTROMBINA', 'APTT -- TIEMPO DE CEFALINA (APTT)', 'AU -- ACIDO URICO', 'BAS -- Bas¢filos', 'BAS -- SISTEMATICO DE SANGRE', 'BAS% -- Bas¢filos %', 'BAS% -- SISTEMATICO DE SANGRE', 'BD -- BILIRRUBINA DIRECTA', 'BE(b) -- BE(b)', 'BE(b)V -- BE (b)', 'BEecf -- BEecf', 'BEecfV -- BEecf', 'BT -- BILIRRUBINA TOTAL', 'BT -- BILIRRUBINA TOTAL                                                               ', 'CA -- CALCIO                                                                          ', 'CA++ -- Ca++ Gasometria', 'CHCM -- Conc. Hemoglobina Corpuscular Media', 'CHCM -- SISTEMATICO DE SANGRE', 'CK -- CK (CREATINQUINASA)', 'CL -- CLORO', 'CREA -- CREATININA', 'DD -- DIMERO D', 'EOS -- Eosin¢filos', 'EOS -- SISTEMATICO DE SANGRE', 'EOS% -- Eosin¢filos %', 'EOS% -- SISTEMATICO DE SANGRE', 'FA -- FOSFATASA ALCALINA', 'FER -- FERRITINA', 'FIB -- FIBRINàGENO', 'FOS -- FOSFORO', 'G-CORONAV (RT-PCR) -- Tipo de muestra: Exudado Far¡ngeo/Nasofar¡ngeo', 'GGT -- GGT (GAMMA GLUTAMIL TRANSPEPTIDASA)', 'GLU -- GLUCOSA', 'GOT -- GOT (AST)', 'GPT -- GPT (ALT)', 'HCM -- Hemoglobina Corpuscular Media', 'HCM -- SISTEMATICO DE SANGRE', 'HCO3 -- HCO3-', 'HCO3V -- HCO3-', 'HCTO -- Hematocrito', 'HCTO -- SISTEMATICO DE SANGRE', 'HEM -- Hemat¡es', 'HEM -- SISTEMATICO DE SANGRE', 'HGB -- Hemoglobina', 'HGB -- SISTEMATICO DE SANGRE', 'INR -- INR', 'K -- POTASIO', 'LAC -- LACTATO', 'LDH -- LDH', 'LEUC -- Leucocitos', 'LEUC -- SISTEMATICO DE SANGRE', 'LIN -- Linfocitos', 'LIN -- SISTEMATICO DE SANGRE', 'LIN% -- Linfocitos %', 'LIN% -- SISTEMATICO DE SANGRE', 'MG -- MAGNESIO', 'MONO -- Monocitos', 'MONO -- SISTEMATICO DE SANGRE', 'MONO% -- Monocitos %', 'MONO% -- SISTEMATICO DE SANGRE', 'NA -- SODIO', 'NEU -- Neutr¢filos', 'NEU -- SISTEMATICO DE SANGRE', 'NEU% -- Neutr¢filos %', 'NEU% -- SISTEMATICO DE SANGRE', 'PCO2 -- pCO2', 'PCO2V -- pCO2', 'PCR -- PROTEINA C REACTIVA', 'PH -- pH', 'PHV -- pH', 'PLAQ -- Recuento de plaquetas', 'PLAQ -- SISTEMATICO DE SANGRE', 'PO2 -- pO2', 'PO2V -- pO2', 'PROCAL -- PROCALCITONINA', 'PT -- PROTEINAS TOTALES', 'SO2C -- sO2c (Saturaci¢n de ox¡geno)', 'SO2CV -- sO2c (Saturaci¢n de ox¡geno)', 'TCO2 -- tCO2(B)c', 'TCO2V -- tCO2 (B)', 'TP -- TIEMPO DE PROTROMBINA', 'TROPO -- TROPONINA', 'U -- UREA', 'VCM -- SISTEMATICO DE SANGRE', 'VCM -- Volumen Corpuscular Medio', 'VPM -- SISTEMATICO DE SANGRE', 'VPM -- Volumen plaquetar medio', 'VSG -- VSG']
require_impute_features = labtest_features
normalize_features = ['Age'] + labtest_features + ['LOS']

# %%
df = pd.read_csv(os.path.join(data_dir, "processed", f"cdsl_dataset_formatted.csv"))
# df.head()

# %%
# Group the dataframe by patient ID
grouped = df.groupby('PatientID')

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# Get the train_val/test patient IDs
train_val_patients, test_patients = train_test_split(patients, test_size=25/100, random_state=SEED, stratify=patients_outcome)

# Get the train/val patient IDs
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=25/75, random_state=SEED, stratify=train_val_patients_outcome)

# Create train, val, test, [traincal, calib] dataframes for the current fold
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]

save_dir = os.path.join(data_dir, 'processed', 'fold_1') # forward fill
os.makedirs(save_dir, exist_ok=True)
# # Save the train, val, and test dataframes for the current fold to csv files
# train_df.to_csv(os.path.join(save_dir, "train_raw.csv"), index=False)
# val_df.to_csv(os.path.join(save_dir, "val_raw.csv"), index=False)
# test_df.to_csv(os.path.join(save_dir, "test_raw.csv"), index=False)
test_raw_x, test_raw_y, test_raw_pid = forward_fill_pipeline(test_df, test_df[normalize_features].median(), demographic_features, labtest_features, target_features, require_impute_features)
pd.to_pickle(test_raw_x, os.path.join(save_dir, "test_raw_x.pkl"))
val_raw_x, val_raw_y, val_raw_pid = forward_fill_pipeline(val_df, val_df[normalize_features].median(), demographic_features, labtest_features, target_features, require_impute_features)
pd.to_pickle(val_raw_x, os.path.join(save_dir, "val_raw_x.pkl"))
# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# Drop rows if all features are recorded NaN
train_df = train_df.dropna(axis=0, how='all', subset=normalize_features)
val_df = val_df.dropna(axis=0, how='all', subset=normalize_features)
test_df = test_df.dropna(axis=0, how='all', subset=normalize_features)


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
pd.to_pickle(labtest_features, os.path.join(save_dir, 'labtest_features.pkl'))

# %%
len(test_pid), len(val_pid), len(train_pid)

# %%
folds = 3
for fold in range(folds):
    data = []
    save_dir = os.path.join(data_dir, 'ensemble', f'fold_{fold}')
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

# %% [markdown]
# ## Data-driven Information

# %%
pd.to_pickle(df.groupby('Outcome').get_group(0).describe().to_dict('dict'), os.path.join(save_dir, 'survival.pkl'))
pd.to_pickle(df.groupby('Outcome').get_group(1).describe().to_dict('dict'), os.path.join(save_dir, 'dead.pkl'))

# %% [markdown]
# ## Basic Information

# %%
pd.to_pickle(df[['PatientID', 'Sex', 'Age']].groupby('PatientID').first().to_dict('index'), os.path.join(data_dir, 'processed', 'fold_1', 'basic.pkl'))


