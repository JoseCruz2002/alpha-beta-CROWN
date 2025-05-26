from expl_algos.src.advExOracle.base_oracle import BaseOracle
from  expl_algos.src.utils.vnnlib_file_builder import VnnlibFileBuilder
from  expl_algos.src.utils.config_file_builder import ConfigFileBuilder

import os
import re

import subprocess

class Debug_abCrown_Oracle(BaseOracle):

    def __init__(self, distance, norm, explanation_problem, base_costumization,
                 final_rv, sample_name, benchmark_year=""):
        BaseOracle.__init__(self)
        # Initiating some atributes
        # bounds - Has the initial bounds for the verification, before any fixing
        # iteration - simply how many features the algorithm has already went through
        self.bounds = final_rv[0][0] # benchmark models
        self.iteration = 0
        self.norm = norm
        self.distance = distance
        self.explanation_problem = explanation_problem
        # Initializing the vnnlib specifications file builder
        self.classifier_name = self.explanation_problem.classification_problem.classifier +\
                               f"_{sample_name}"
        self.vnnlib_file_builder = VnnlibFileBuilder(
            n_features=explanation_problem.get_num_features(),
            n_outputs=explanation_problem.get_num_outputs(),
            label=explanation_problem.classification,
            classifier_name=self.classifier_name
        )
        # Initializing the configuration file builder
        if benchmark_year == "":
            costumization_path = os.path.join(os.path.dirname(__file__),
                                    "../../../complete_verifier/exp_configs/elsa_comp")
        elif benchmark_year == "2024":
            costumization_path = os.path.join(os.path.dirname(__file__),
                                    "../../../complete_verifier/exp_configs/vnncomp24")
        elif benchmark_year == "2023":
            costumization_path = os.path.join(os.path.dirname(__file__),
                                    "../../../complete_verifier/exp_configs/vnncomp23")
        else:
            NotImplementedError
        self.config_file_builder = ConfigFileBuilder(
            base_config_path=os.path.join(costumization_path, base_costumization),
            norm=norm,
            distance=distance,
            classifier_name=self.classifier_name
        )
        # Path where I want the output of the alpha-beta-CROWN runs to go
        self.results_path = os.path.join(os.path.dirname(__file__), "../../abCROWN_results")

    def findAdvEx(self, distance, fixed_features, explanation_problem, norm):
        # Build the vnnlib file with the specifications of this iteration
        fixed_bouds = self.get_fixed_bouds(
            feature_set=self.explanation_problem.classification_problem.feature_set,
            fixed_features=fixed_features,
            )
        specs = self.vnnlib_file_builder.buildFile(fixed_bouds)
        specs_path = self.vnnlib_file_builder.saveFile(specs, self.iteration)
        # Build the configuration file with the desired specs of this iteration
        self.config_file_builder.buildFile(specs_path)
        config_path = self.config_file_builder.saveFile(self.iteration)
        # Run the alpha-beta-CROWN tool by the command line and save the results to a file
        dir_name = f"{self.results_path}/{self.classifier_name}_d{self.distance}_n{self.norm}"
        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, f"{self.classifier_name}_{self.iteration}.out")
        subprocess.run([
            "bash",
            "/home/josecruz/Documents/MEIC/Thesis/alpha-beta-CROWN/expl_algos/run_abCROWN.sh",
            config_path,
            output_path
        ])
        # Parse the output
        with open(os.path.join("../../", output_path), "r") as f:
            output = f.read()
        res = self.parse_output(output)
        self.iteration += 1
        if (res == "unsafe" or res == "sat" or res == "attack success"):
            return True
        else:
            return False
        
    def get_fixed_bouds(self, feature_set, fixed_features):
        index_map = {value: idx for idx, value in enumerate(feature_set)}
        fix_idxs = list(index_map[val] for val in fixed_features)
        data_min = []
        data_max = []
        for i in range(len(feature_set)):
            if i in fix_idxs:
                idx_value = self.explanation_problem.value[i]
                data_min += [float(idx_value)]
                data_max += [float(idx_value)]
            else:
                data_min += [self.bounds[i][0]]
                data_max += [self.bounds[i][1]]
        return list(zip(data_min, data_max))
    
    def parse_output(self, output):
        """
        output: Str - stdout of the alpha-beta-CROWN run
        """
        regex_verified_status = re.compile(r"verified_status [a-zA-Z0-9_-]+")
        regex_verified_success = re.compile(r"verified_success (True|False)")
        regex_result = re.compile(r"Result: [a-zA-Z0-9_-]+")
        regex_global_lb = re.compile(
            r"-- global_lb: tensor\(\[\[[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\]\]\)")
        for line in output.split('\n'):
            verified_status = regex_verified_status.findall(line)
            verified_success = regex_verified_success.findall(line)
            result = regex_result.findall(line)
            global_lb = regex_global_lb.findall(line)
            if len(global_lb) > 0:
                print(f"iteration {self.iteration} - global_lb: {global_lb[0]}")
            if len(verified_status) > 0 and len(verified_success) > 0:
                print(f"iteration {self.iteration} - verified_status: {verified_status[0]}")
                print(f"iteration {self.iteration} - verified_success: {verified_success[0]}")
                if verified_success[0].split(' ')[1] == "False":
                    return "unknown"
                else:
                    return verified_status[0].split(' ')[1]
            elif len(result) > 0:
                print(f"iteration {self.iteration} - result: {result[0].split(' ')[1]}")
                return result[0].split(' ')[1]
        print("DID NOT FIND RESULT OR COMBINATION OF VERIFIED SUCCESS AND STATUS!!")
        return "unknown"


