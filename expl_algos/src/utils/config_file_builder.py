import os
import yaml

class ConfigFileBuilder:

    def __init__(self, base_config_path, norm, distance, classifier_name):
        self.config_files_path = os.path.join(os.path.dirname(__file__),
                                                   "../../config_files")
        with open(base_config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.norm = norm
        self.distance = distance
        self.classifier_name = classifier_name

    def saveFile(self, algo_iteration):
        dir_name = f"{self.config_files_path}/{self.classifier_name}_d{self.distance}_n{str(self.norm).replace('.', ' ')}"
        os.makedirs(dir_name, exist_ok=True)
        path = os.path.join(dir_name, f"{self.classifier_name}_{algo_iteration}.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config, f)
        return path

    def buildFile(self, vnnlib_spec_file_path):
        """
        Put the distance (epsilon) and the norm on the config file.
        Give the path to the vnnlib file built here by the vnnlib_file_builder.py script.
        In the base configuration file the onnx_path starts with complete_verifier.
            This must not happen here as the alpha-beta-CROWN will be run from the
            complete_verifier directory.
        In the base configuration file the normal_run is set to false, this must be changed to true
        """
        if "specification" not in self.config:
            self.config["specification"] = {}
        self.config["specification"]["norm"] = self.norm
        self.config["specification"]["epsilon"] = self.distance
        self.config["specification"]["vnnlib_path"] = vnnlib_spec_file_path
        if "model" in self.config and "onnx_path" in self.config["model"]\
                and "complete_verifier" in self.config["model"]["onnx_path"]:
            self.config["model"]["onnx_path"] = '/'.join(self.config["model"]["onnx_path"].split('/')[1:])
        if "general" in self.config and "normal_run" in self.config["general"]:
            self.config["general"]["normal_run"] = True



#if __name__ == "__main__":
#    builder = ConfigFileBuilder(
#        base_config_path="complete_verifier/exp_configs/vnncomp24/safenlp_oracle.yaml",
#        norm=np.inf,
#        distance=10,
#        classifier_name="classifier"
#    )
#    builder.buildFile("expl_algos/vnnlib_spec_files/test_classifier_2.vnnlib")
#    builder.saveFile(2)
