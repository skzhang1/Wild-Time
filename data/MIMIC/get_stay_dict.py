import os
import pickle
import random

import numpy as np
import pandas as pd

"""
Reference: https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer
"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sample_year(anchor_year_group):
    year_min = int(anchor_year_group[:4])
    year_max = int(anchor_year_group[-4:])
    assert year_max - year_min == 2
    return random.randint(year_min, year_max)


def assign_readmission_label(row):
    curr_subject_id = row.subject_id
    curr_admittime = row.admittime

    next_row_subject_id = row.next_row_subject_id
    next_row_admittime = row.next_row_admittime

    if curr_subject_id != next_row_subject_id:
        label = 0
    elif (next_row_admittime - curr_admittime).days > 15:
        label = 0
    else:
        label = 1

    return label


def diag_icd9_to_3digit(icd9):
    if icd9.startswith('E'):
        if len(icd9) >= 4:
            return icd9[:4]
        else:
            print(icd9)
            return icd9
    else:
        if len(icd9) >= 3:
            return icd9[:3]
        else:
            print(icd9)
            return icd9


def diag_icd10_to_3digit(icd10):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def diag_icd_to_3digit(icd):
    if icd[:4] == 'ICD9':
        return 'ICD9_' + diag_icd9_to_3digit(icd[5:])
    elif icd[:5] == 'ICD10':
        return 'ICD10_' + diag_icd10_to_3digit(icd[6:])
    else:
        raise


def list_join(lst):
    return ' <sep> '.join(lst)


def proc_icd9_to_3digit(icd9):
    if len(icd9) >= 3:
        return icd9[:3]
    else:
        print(icd9)
        return icd9


def proc_icd10_to_3digit(icd10):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def proc_icd_to_3digit(icd):
    if icd[:4] == 'ICD9':
        return 'ICD9_' + proc_icd9_to_3digit(icd[5:])
    elif icd[:5] == 'ICD10':
        return 'ICD10_' + proc_icd10_to_3digit(icd[6:])
    else:
        raise


def process_mimic_data():
    set_seed(seed=42)
    data_path = os.getcwd()

    for file in ['patients.csv, diagnoses_icd.csv', 'procedures_icd.csv']:
        if not os.path.isfile(os.path.join(data_path, f'raw/mimic4/{file}')):
            raise ValueError(f'Please download {file} to {data_path}/raw/mimic4')

    # Patients
    patients = pd.read_csv(os.path.join(data_path, 'raw/mimic4/patients.csv'))
    patients['real_anchor_year_sample'] = patients.anchor_year_group.apply(lambda x: sample_year(x))
    patients = patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'real_anchor_year_sample']]
    patients = patients.dropna().reset_index(drop=True)
    admissions = pd.read_csv(os.path.join(data_path, 'raw/mimic4/admissions.csv'))
    admissions['admittime'] = pd.to_datetime(admissions['admittime']).dt.date
    admissions = admissions[['subject_id', 'hadm_id', 'ethnicity', 'admittime', 'hospital_expire_flag']]
    admissions = admissions.dropna()
    admissions['mortality'] = admissions.hospital_expire_flag
    admissions = admissions.sort_values(by=['subject_id', 'hadm_id', 'admittime'])
    admissions['next_row_subject_id'] = admissions.subject_id.shift(-1)
    admissions['next_row_admittime'] = admissions.admittime.shift(-1)
    admissions['readmission'] = admissions.apply(lambda x: assign_readmission_label(x), axis=1)
    admissions = admissions[['subject_id', 'hadm_id', 'ethnicity', 'admittime', 'mortality', 'readmission']]
    admissions = admissions.dropna().reset_index(drop=True)

    # Diagnoses ICD
    diagnoses_icd = pd.read_csv(os.path.join(data_path, 'raw/mimic4/diagnoses_icd.csv'))
    diagnoses_icd = diagnoses_icd.dropna()
    diagnoses_icd = diagnoses_icd.drop_duplicates()
    diagnoses_icd = diagnoses_icd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'])
    diagnoses_icd['icd_code'] = diagnoses_icd.apply(lambda x: f'ICD{x.icd_version}_{x.icd_code}', axis=1)
    diagnoses_icd['icd_3digit'] = diagnoses_icd.icd_code.apply(lambda x: diag_icd_to_3digit(x))
    diagnoses_icd = diagnoses_icd.groupby(['subject_id', 'hadm_id']).agg({'icd_3digit': list_join}).reset_index()
    diagnoses_icd = diagnoses_icd.rename(columns={'icd_3digit': 'diagnoses'})

    # Procedures ICD
    procedures_icd = pd.read_csv(os.path.join(data_path, 'raw/mimic4/procedures_icd.csv'))
    procedures_icd = procedures_icd.dropna()
    procedures_icd = procedures_icd.drop_duplicates()
    procedures_icd = procedures_icd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'])
    procedures_icd['icd_code'] = procedures_icd.apply(lambda x: f'ICD{x.icd_version}_{x.icd_code}', axis=1)
    procedures_icd['icd_3digit'] = procedures_icd.icd_code.apply(lambda x: proc_icd_to_3digit(x))
    procedures_icd = procedures_icd.groupby(['subject_id', 'hadm_id']).agg({'icd_3digit': list_join}).reset_index()
    procedures_icd = procedures_icd.rename(columns={'icd_3digit': 'procedure'})

    # Merge
    df = admissions.merge(patients, on='subject_id', how='inner')
    df['real_admit_year'] = df.apply(lambda x: x.admittime.year - x.anchor_year + x.real_anchor_year_sample, axis=1)
    df['age'] = df.apply(lambda x: x.admittime.year - x.anchor_year + x.anchor_age, axis=1)
    df = df[['subject_id', 'hadm_id',
             'admittime', 'real_admit_year',
             'age', 'gender', 'ethnicity',
             'mortality', 'readmission']]
    df = df.merge(diagnoses_icd, on=['subject_id', 'hadm_id'], how='inner')
    df = df.merge(procedures_icd, on=['subject_id', 'hadm_id'], how='inner')
    df.to_csv('./data_preprocessed.csv')

    # Cohort Selection
    df = df[df.age.apply(lambda x: (x >= 18) & (x <= 89))]

    # Save
    directory = os.path.join(data_path, 'processed/mimic4/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    df.to_csv(os.path.join(data_path, 'processed/mimic4/data.csv'), index=False)


class MIMICStay:

    def __init__(self,
                 icu_id,
                 icu_timestamp,
                 mortality,
                 readmission,
                 age,
                 gender,
                 ethnicity):
        self.icu_id = icu_id    # str
        self.icu_timestamp = icu_timestamp  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity  # str

        self.diagnosis = []     # list of tuples (timestamp in min (int), diagnosis (str))
        self.treatment = []     # list of tuples (timestamp in min (int), treatment (str))

    def __repr__(self):
        return f'MIMIC ID-{self.icu_id}, mortality-{self.mortality}, readmission-{self.readmission}'


def get_stay_dict():
    process_mimic_data()
    mimic_dict = {}
    input_path = './data/MIMIC/processed/mimic4/data.csv'
    fboj = open(input_path)
    name_list = fboj.readline().strip().split(',')
    for eachline in fboj:
        t=eachline.strip().split(',')
        tempdata={eachname: t[idx] for idx, eachname in enumerate(name_list)}
        mimic_value = MIMICStay(icu_id=tempdata['hadm_id'],
                                 icu_timestamp=tempdata['real_admit_year'],
                                 mortality=tempdata['mortality'],
                                 readmission=tempdata['readmission'],
                                 age=tempdata['age'],
                                 gender=tempdata['gender'],
                                 ethnicity=tempdata['ethnicity'])
        mimic_value.diagnosis = tempdata['diagnoses'].split(' <sep> ')
        mimic_value.treatment = tempdata['procedure'].split(' <sep> ')
        mimic_dict[tempdata['hadm_id']]=mimic_value

    pickle.dump(mimic_dict, open('./Data/mimic_stay_dict.pkl', 'wb'))