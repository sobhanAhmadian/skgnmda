import pickle
import os
import json
import time

from typing import List, Union, Dict


def pickle_load(filename: str):
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        print(f"Logging Info - Loaded: {filename}")
    except EOFError:
        print(f"Logging Error - Cannot load: {filename}")
        obj = None

    return obj


def json_dump(file_path, obj: Union[List, Dict]):
    try:
        with open(file_path, "w+") as f:
            json.dump(obj, f)
    except Exception as e:
        print(e)


def json_load(file_path) -> Union[List, Dict]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(e)


def pickle_dump(filename: str, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"Logging Info - Saved: {filename}")


def format_filename(_dir: str, filename_template: str, **kwargs):
    """Obtain the filename of data_repository base on the provided template and parameters"""
    filename = os.path.join(_dir, filename_template.format(**kwargs))
    return filename
