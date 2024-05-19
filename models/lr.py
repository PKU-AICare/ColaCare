from sklearn.linear_model import LogisticRegression


class LR():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        max_depth: int, depth of trees
        """
        task = params['task']
        self.task = task
        seed = params['seed']
        self.model = LogisticRegression(random_state=seed)

    def fit(self, x, y):
        if self.task == "outcome":
            self.model.fit(x, y)
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
    
    def predict(self, x):
        if self.task == "outcome":
            return self.model.predict_proba(x)[:, 1]
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
