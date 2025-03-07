import os

from complete_verifier.abcrown import ABCROWN

class abCrown_API():

    def __init__(self, costumization_file_name):
        costumization_path = os.path.join(os.path.dirname(__file__),
                        "../complete_verifier/exp_configs/elsa_comp")
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