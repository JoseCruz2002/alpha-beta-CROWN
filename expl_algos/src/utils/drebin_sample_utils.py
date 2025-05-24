from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import pandas as pd
import numpy as np

def load_sample(elsa_comp_path, classifier, SHA):
    "Only load from the goodware test dataset"
    sample_path = os.path.join(elsa_comp_path,
                                #"data/test_set_fp_check_features.zip")
                                "data/test_set_adv_features.zip")
    with ZipFile(sample_path, "r", ZIP_DEFLATED) as z:
        with z.open(f"{SHA}.json") as f:
            js = json.load(f)
            return [f"{k}::{v}" for k in js for v in js[k] if js[k]]


def load_classification(elsa_comp_path, classifier, SHA):
    submission_path = os.path.join(elsa_comp_path,
                                f"track_1/submissions/submission_{classifier}_track_1.json")
    with open(submission_path, "r") as f:
        submission = json.load(f)
    for test in submission:
        for (SHA_, [label, _]) in test.items():
            if SHA_ == SHA:
                print(f"SHA: {SHA}: label: {label}")
                return label

def load_features(features_path):
    """

    Parameters
    ----------
    features_path :
        Absolute path of the features compressed file.

    Returns
    -------
    generator of list of strings
        Iteratively returns the textual feature vector of each sample.
    """
    with ZipFile(features_path, "r", ZIP_DEFLATED) as z:
        for filename in z.namelist():
            with z.open(filename) as fp:
                js = json.load(fp)
                yield [f"{k}::{v}" for k in js for v in js[k] if js[k]]
    