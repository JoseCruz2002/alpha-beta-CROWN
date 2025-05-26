import os

class VnnlibFileBuilder:

    def __init__(self, n_features, n_outputs, label, classifier_name):
        """
        n_features: Int
        n_outputs: Int
        bounds: list(tuple) -> [(-0.9, 0.2), (2.3, 2.9), ...]
        label: Int
        classifier_name: Str
        """
        self.vnnlib_spec_files_path = os.path.join(os.path.dirname(__file__),
                                                   "../../vnnlib_spec_files")
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.label = label
        self.classifier_name = classifier_name

    def saveFile(self, vnnlib_specs, algo_iteration):
        dir_name = f"{self.vnnlib_spec_files_path}/{self.classifier_name}"
        os.makedirs(dir_name, exist_ok=True)
        path = os.path.join(dir_name, f"{self.classifier_name}_{algo_iteration}.vnnlib")
        with open(path, "w") as f:
            f.write(vnnlib_specs)
        return path

    def buildFile(self, bounds):
        # declare the features
        declarations = ""
        for i in range(self.n_features):
            declarations += self.declareFeaturesForm(i) + '\n'
        declarations += '\n'
        # declare the outputs
        for i in range(self.n_outputs):
            declarations += self.declareOutputsForm(i) + '\n'
        # constrain the input, i.e., create the ball
        input_constrains = ""
        for i in range(self.n_features):
            input_constrains += self.assertFeatureBoundsForm(i, bounds[i]) + '\n\n'
        # constrain the output, i.e., represent the label
        output_constrains = ""
        for i in range(self.n_outputs):
            if i != self.label:
                output_constrains += self.assertLabelForm(self.label, i) + '\n'
        # join everything together and create vnnlib spec file
        total = declarations + '\n' + input_constrains + '\n' + output_constrains
        return total

    def declareFeaturesForm(self, feat_num):
        return f"(declare-const X_{feat_num} Real)"
    
    def declareOutputsForm(self, output_num):
        return f"(declare-const Y_{output_num} Real)"
    
    def assertFeatureBoundsForm(self, feat_num, feat_bounds):
        return f"(assert (>= X_{feat_num} {feat_bounds[0]}))\n"+\
               f"(assert (<= X_{feat_num} {feat_bounds[1]}))"

    def assertLabelForm(self, true_label, output_num):
        return f"(assert (<= Y_{output_num} Y_{true_label}))"



#if __name__ == "__main__":
#    builder = VnnlibFileBuilder(
#        n_features=3,
#        n_outputs=2,
#        label=1,
#        classifier_name="test_classifier"
#    )
#    vnnlib_specs = builder.buildFile(bounds=[(0.3, 0.5), (2.9, 10.9), (-0.8, 2.3)])
#    builder.saveFile(vnnlib_specs, 2)
