from expl_algos.src.helper_definitions.classification_problem import ClassificationProblem

class ExplanationProblem():

    def __init__(self, classification_problem: ClassificationProblem,
                 input: list[str], classification: int):
        self._classification_problem = classification_problem
        self._instance = [input, classification]
        self.transform_input()
        #print(f"instance: {self._instance}")

    @property
    def classification_problem(self):
        return self._classification_problem
    
    @property
    def instance(self):
        return self._instance
    
    @property
    def value(self):
        return self._instance[0]
    
    @property
    def classification(self):
        return self._instance[1]
    
    def transform_input(self):
        self.instance[0] = self.classification_problem.transform_input(self.value)
        return self.instance[0]