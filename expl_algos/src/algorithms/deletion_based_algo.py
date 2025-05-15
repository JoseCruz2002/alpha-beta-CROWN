from expl_algos.src.helper_definitions.explanation_problem import ExplanationProblem
from expl_algos.src.helper_definitions.classification_problem import ClassificationProblem
from expl_algos.src.advExOracle.base_oracle import BaseOracle
from expl_algos.src.advExOracle.abcrown_oracle import abCrown_Oracle
from expl_algos.src.utils.sample_utils import load_classification, load_sample, load_features

import json
import os
import argparse
import dill as pkl
import numpy as np

def findCXpDel(distance: int, explanation_problem: ExplanationProblem,
               norm: str, advEx_oracle: BaseOracle):
    F = explanation_problem.classification_problem.feature_set
    #print(f"F: {len(F)};")
    S: list[str] = F.copy()
    for feature in F:
        #print(f"\nbeginning S: {len(S)};")
        S.remove(feature)
        F_except_S = list(x for x in F if x not in S)
        print(f"F_except_S: {len(F_except_S)}")#; {F_except_S}")
        hasAE = advEx_oracle.findAdvEx(distance, F_except_S, explanation_problem, norm)
        #if True:
        if not hasAE:
            S.append(feature)
        #print(f"finishing S: {len(S)};")
        #break
    return S

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file_name", type=str,
                        help="The name of the file to use to configure the adv. ex. oracle.")
    parser.add_argument("-classifier", type=str,
                        help="The name of the classifier, used to load its vectorizer")
    parser.add_argument("-feat_set_file", type=str,
                        help="The name of the feature set file")
    parser.add_argument("-sampleSHA", type=str,
                        help="The SHA (id) of the instance to be evaluated")
    parser.add_argument("-distance", type=int,
                        help="The distance on which to look for adversarial examples")
    opt = parser.parse_args()
    print(f"The parameters passed to the program are:\n\
            config_file_name: {opt.config_file_name}\n\
            classifier: {opt.classifier}\n\
            feat_set_file: {opt.feat_set_file}\n\
            sampleSHA: {opt.sampleSHA}\n\
            distance: {opt.distance}")

    base_path = os.path.join(os.path.dirname(__file__))
    explanations_path = os.path.join(base_path, "../../explanations")
    elsa_comp_path = os.path.join(base_path, "../../../../elsa-cybersecurity")
    features_selected_path = os.path.join(elsa_comp_path,
                            f"track_1/selected_features/{opt.feat_set_file}")
    features_data_path = os.path.join(elsa_comp_path, "data/training_set_features.zip")

    with open(features_selected_path, "r") as f:
        features_selected = json.load(f)

    classification_problem = ClassificationProblem(
        feature_set=features_selected,
        classes_set=[0, 1],
        #classifier=opt.classifier
    )
    
    if "FFNN" in opt.classifier:
        classification_problem.fit_vectorizer(load_features(features_data_path))

    explanation_problem = ExplanationProblem(
        classification_problem=classification_problem,
        input=load_sample(elsa_comp_path, opt.sampleSHA),
        classification=load_classification(elsa_comp_path, opt.classifier,
                                           opt.sampleSHA)
    )
    advEx_oracle = abCrown_Oracle(costumization_file_name=opt.config_file_name)

    explanation = findCXpDel(opt.distance, explanation_problem, 1, advEx_oracle)
    print("############################################################################")
    print(f"explanation size: {len(explanation)}")
    print(f"Saving explanation to: {os.path.join(explanations_path, f'CXp_{opt.classifier}_{opt.distance}_{opt.sampleSHA}.json')}")
    #with open(os.path.join(explanations_path,
    #                       f"CXp_{opt.classifier}_eps{opt.distance}_{opt.sampleSHA}.json"), "w") as f:
    #    json.dump(explanation, f, indent=2)


if __name__ == "__main__":
    main()