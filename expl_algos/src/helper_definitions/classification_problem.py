from sklearn.feature_extraction.text import CountVectorizer

class ClassificationProblem():

    def __init__(self, feature_set: list[str], classes_set: list[int],
                 classifier=None):
        self._feature_set = feature_set
        self._classes_set = classes_set
        self._classifier = classifier
        self._vectorizer = CountVectorizer(
            input="content", lowercase=False,
            tokenizer=lambda x: x, binary=True, token_pattern=None,
            vocabulary=feature_set)

    @property
    def feature_set(self):
        return self._feature_set
    
    @property
    def classes_set(self):
        return self._classes_set
    
    @property
    def classifier(self):
        return self._classifier
    
    @property
    def vectorizer(self):
        return self._vectorizer
    
    def fit_vectorizer(self, features):
        self.vectorizer.fit(features)
    
    def transform_input(self, input):
        return self.vectorizer.transform([input])
