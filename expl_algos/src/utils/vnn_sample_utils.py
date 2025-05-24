import os
import complete_verifier.read_vnnlib as read_vnnlib
import numpy as np

def parse_vnnlib(benchmarks_path, classifier, vnnlib_file):
    if '-' in classifier:
        clf_name, clf_type = classifier.split('-')
        vnnlib_file_path = os.path.join(benchmarks_path, f"{clf_name}/vnnlib/{clf_type}/{vnnlib_file}")
    else:
        vnnlib_file_path = os.path.join(benchmarks_path, f"{classifier}/vnnlib/{vnnlib_file}")
    final_rv = read_vnnlib.read_vnnlib(vnnlib_file_path)
    print(f"final_rv: {final_rv}")
    return final_rv

def load_sample(benchmarks_path, classifier, vnnlib_file):
    final_rv = parse_vnnlib(benchmarks_path, classifier, vnnlib_file)
    input = list((l+u)/2 for (l, u) in final_rv[0][0])
    print(f"input: {input}")
    return input

def load_classification(benchmarks_path, classifier, vnnlib_file):
    final_rv = parse_vnnlib(benchmarks_path, classifier, vnnlib_file)
    label = np.where(final_rv[0][1][0][0][0] == -1)[0][0]
    print(f"label: {label}")
    return label
