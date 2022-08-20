import argparse
import os
import pickle
from typing import List

import gdown

from data.MIMIC.get_stay_dict import get_stay_dict
from data.MIMIC.preprocess import preprocess_MIMIC


def download_gdrive(url, save_path, is_folder):
    """ Download the preprocessed data from Google Drive. """
    if not is_folder:
        gdown.download(url=url, output=save_path, quiet=False)
    else:
        gdown.download_folder(url=url, output=save_path, quiet=False)


def download_arxiv(data_dir):
    download_gdrive(
        url='https://drive.google.com/file/d/17JdUhLWMSm75Tw6z7JBegjz_eWNW4Bnz/view?usp=sharing',
        save_path=os.path.join(data_dir, 'arxiv.pkl'),
        is_folder=False
    )


def download_drug(data_dir):
    download_gdrive(
        url='https://drive.google.com/drive/folders/1796kUMCTs8r0dnQjBiLt7YTm6ZtZAd3f?usp=sharing',
        save_path=os.path.join(data_dir, 'Drug-BA'),
        is_folder=True
    )
    download_gdrive(
        url='https://drive.google.com/file/d/179g3KTOG2mBTZzF6lxG6iodNp8xmdxpg/view?usp=sharing',
        save_path=os.path.join(data_dir, 'drug_preprocessed.pkl'),
        is_folder=False
    )


def download_fmow(data_dir):
    download_gdrive(
        url='https://drive.google.com/file/d/17Jplcs6QSCZb5pmE8xDKKwQZnrjBClsf/view?usp=sharing',
        save_path=os.path.join(data_dir, 'fmow.pkl'),
        is_folder=False
    )


def download_huffpost(data_dir):
    download_gdrive(
        url='https://drive.google.com/file/d/17Jplcs6QSCZb5pmE8xDKKwQZnrjBClsf/view?usp=sharing',
        save_path=os.path.join(data_dir, 'huffpost.pkl'),
        is_folder=False
    )


def download_mimic(args, data_dir):
    if not os.path.exists('./Data/mimic_stay_dict.pkl'):
        get_stay_dict(data_dir)
    data = pickle.load(open('./Data/mimic_stay_dict.pkl', 'rb'))
    preprocess_MIMIC(data, 'readmission', args)
    preprocess_MIMIC(data, 'mortality', args)


def download_precipitation(data_dir):
    download_gdrive(
        url='https://drive.google.com/file/d/19BnZT3VxYEtNIEj0fN76UcGG2GP2JP65/view?usp=sharing',
        save_path=os.path.join(data_dir, 'weather.pkl'),
        is_folder=False
    )


def download_yearbook(data_dir):
    download_gdrive(
        url='https://drive.google.com/file/d/17I50QFmMAowDRswOD0sr-uq9tQZdjM4v/view?usp=sharing',
        save_path=os.path.join(data_dir, 'yearbook.pkl'),
        is_folder=False
    )


def download_datasets(
        data_dir: str,
        datasets: List[str]
):
    if 'arxiv' in datasets:
        download_arxiv(data_dir)
    if 'drug' in datasets:
        download_drug(data_dir)
    if 'fmow' in datasets:
        download_fmow(data_dir)
    if 'huffpost' in datasets:
        download_huffpost(data_dir)
    if 'mimic' in datasets:
        download_mimic(args, data_dir)
    if 'precipitation' in datasets:
        download_precipitation(data_dir)
    if 'yearbook' in datasets:
        download_yearbook(data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--datasets', nargs='*', type=str,
                        default=['arxiv', 'drug', 'fmow', 'huffpost', 'mimic', 'precipitation', 'yearbook'])
    args = parser.parse_args()

    download_datasets(args.data_dir, args.datasets)