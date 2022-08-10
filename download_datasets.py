import argparse
import os
from typing import List

import gdown


def download_gdrive(url, save_path, is_folder):
    """ Download the preprocessed data from Google Drive. """
    if not is_folder:
        gdown.download(url=url, ooutput=save_path, quiet=False)
    else:
        gdown.download_folder(url=url, output=save_path, quiet=False)


def download_datasets(
        save_dir: str,
        datasets: List[str]
):
    if 'arxiv' in datasets:
        download_gdrive(
            url='https://drive.google.com/file/d/17JdUhLWMSm75Tw6z7JBegjz_eWNW4Bnz/view?usp=sharing',
            save_path=os.path.join(save_dir, 'arxiv.pkl'),
            is_folder=False
        )
    if 'drug' in datasets:
        download_gdrive(
            url='https://drive.google.com/drive/folders/1796kUMCTs8r0dnQjBiLt7YTm6ZtZAd3f?usp=sharing',
            save_path=os.path.join(save_dir, 'Drug-BA'),
            is_folder=True
        )
        download_gdrive(
            url='https://drive.google.com/file/d/179g3KTOG2mBTZzF6lxG6iodNp8xmdxpg/view?usp=sharing',
            save_path=os.path.join(save_dir, 'drug_preprocessed.pkl'),
            is_folder=False
        )
    if 'fmow' in datasets:
        download_gdrive(
            url='https://drive.google.com/file/d/17Jplcs6QSCZb5pmE8xDKKwQZnrjBClsf/view?usp=sharing',
            save_path=os.path.join(save_dir, 'fmow.pkl'),
            is_folder=False
        )
    if 'huffpost' in datasets:
        download_gdrive(
            url='https://drive.google.com/file/d/17Jplcs6QSCZb5pmE8xDKKwQZnrjBClsf/view?usp=sharing',
            save_path=os.path.join(save_dir, 'huffpost.pkl'),
            is_folder=False
        )
    if 'precipitation' in datasets:
        download_gdrive(
            url='https://drive.google.com/file/d/19BnZT3VxYEtNIEj0fN76UcGG2GP2JP65/view?usp=sharing',
            save_path=os.path.join(save_dir, 'weather.pkl'),
            is_folder=False
        )
    if 'yearbook' in datasets:
        download_gdrive(
            url='https://drive.google.com/file/d/17I50QFmMAowDRswOD0sr-uq9tQZdjM4v/view?usp=sharing',
            save_path=os.path.join(save_dir, 'yearbook.pkl'),
            is_folder=False
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--save_dir', type=str, default='./Data')
    parser.add_argument('--datasets', nargs='*', type=str,
                        default=['arxiv', 'drug', 'fmow', 'huffpost', 'precipitation', 'yearbook'])
    args = parser.parse_args()

    download_datasets(args.save_dir, args.datasets)