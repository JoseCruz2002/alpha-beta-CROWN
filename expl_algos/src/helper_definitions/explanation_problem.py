from expl_algos.src.helper_definitions.classification_problem import ClassificationProblem

class ExplanationProblem():

    def __init__(self, classification_problem: ClassificationProblem,
                 input, classification: int):
        self._classification_problem = classification_problem
        self._instance = [input, classification]
        if "FFNN" in classification_problem.classifier or "drebin" in classification_problem.classifier or\
           "secsvm" in classification_problem.classifier or "MLP_Sklearn" in classification_problem.classifier:
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
    
    def get_num_features(self):
        return len(self.classification_problem.feature_set)
    
    def get_num_outputs(self):
        return len(self.classification_problem.classes_set)
    
    def transform_input(self):
        self.instance[0] = self.classification_problem.transform_input(self.value)
        return self.instance[0]