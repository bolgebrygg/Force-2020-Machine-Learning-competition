"""
FORCE challenge
ISPL Team submission package

Luca Bondi (luca.bondi@polimi.it)
Vincenzo Lipari (vincenzo.lipari@polimi.it)
Paolo Bestagini (paolo.bestagini@polimi.it)
Francesco Picetti (francesco.picetti@polimi.it)
Edoardo Daniele Cannas (edoardo.daniele.cannas@polimi.it)
"""

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from utils import lithology


def download_models(dst: Path):
    url = 'https://www.dropbox.com/s/o5z2t7uvchpid9l/SubmissionEnsamble_v1.pkl?dl=1'
    print('Model can be downloaded separately at {}'.format(url))
    sha1 = '85e6b00cd845ece99d48a10b759aa58033943be4'
    buffer_size = 1024
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    progress = tqdm(response.iter_content(buffer_size), f"Download", total=file_size, unit="B",
                    unit_scale=True, unit_divisor=1024)
    hasher = hashlib.sha1()
    dst.parent.mkdir(exist_ok=True, parents=True)
    with open(dst, "wb") as f:
        for data in progress:
            f.write(data)
            hasher.update(data)
            progress.update(len(data))

    if hasher.hexdigest() != sha1:
        raise RuntimeError('Model hash doesn\'t match')


def main(*, input_csv: str or Path, output_csv: str or Path):
    """
    Predict lithologies
    :param input_csv: CSV input file containing well log data
    :param output_csv: CSV output file where predictions will be saved
    :return:
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f'Input CSV file not found at: {input_csv}')

    if output_csv.exists():
        raise FileExistsError(f'Output CSV file already exists at: {output_csv}')

    print(f'Loading data from {input_csv}')
    data_test = pd.read_csv(input_csv, sep=';')

    print('Loading model')
    model_path = Path('models/SubmissionEnsamble_v1.pkl')

    if not model_path.exists():
        print('Model not found locally. Downloading from Dropbox.')
        download_models(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found at: {model_path}')
    models = joblib.load(model_path)

    print('Predict')
    proba_list = []
    for model in tqdm(models):
        proba_list.append(model.predict_proba(data_test))

    proba_array = np.asarray(proba_list)
    avg_proba = proba_array.mean(axis=0)
    test_pred = np.argmax(avg_proba, axis=1)

    print(f'Saving predictions to: {output_csv}')
    test_codes = list(map(lithology.label2code.__getitem__, test_pred))
    test_df = pd.DataFrame({'lithology': test_codes}, dtype=np.int)
    test_df.to_csv(output_csv, index=False)
    print('Completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input CSV path (well measurements)', type=Path)
    parser.add_argument('output', help='Output CSV path (predicted lithology)', type=Path)
    args = parser.parse_args()
    main(input_csv=args.input, output_csv=args.output)
