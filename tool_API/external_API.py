import os

from alpha_beta_CROWN.complete_verifier.abcrown import ABCROWN

class abCrown_API():

    def __init__(self, costumization_file_name, benchmark_year):
        if benchmark_year == "":
            costumization_path = os.path.join(os.path.dirname(__file__),
                                              "../complete_verifier/exp_configs/elsa_comp")
        elif benchmark_year == "2024":
            costumization_path = os.path.join(os.path.dirname(__file__),
                                              "../complete_verifier/exp_configs/vnncomp24")
        elif benchmark_year == "2023":
            costumization_path = os.path.join(os.path.dirname(__file__),
                                              "../complete_verifier/exp_configs/vnncomp23")
        else:
            NotImplementedError
        self.abCrown = ABCROWN(args=['--config',
                                     os.path.join(costumization_path, costumization_file_name)])
        self.abCrown.main()

    def run_abCrown(self, input, vnnlib):
        """
        Arguments:
            input: The input I want to test for the existance of adversarial examples.
            vnnlib: Contains the bounds, i.e. the fixing of the features.
        Returns:
            verified_status: The result of the verifier.
        """
        return self.abCrown.run_one_instance_externally(input, vnnlib)