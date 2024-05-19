from xgboost import XGBClassifier, XGBRegressor
import shap

class XGBoost():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        learning_rate: float, learning rate
        max_depth: int, depth of trees
        """
        task = params['task']
        self.task = task
        seed = params['seed']
        n_estimators = params['n_estimators']
        learning_rate = params['learning_rate']
        max_depth = params['max_depth']
        if task == "outcome":
            self.model = XGBClassifier(random_state=seed, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, verbose=0, eval_metric="aucpr", objective="binary:logistic")
        elif task == "los":
            self.model = XGBRegressor(random_state=seed, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, verbose=0, eval_metrics="mae", objective="reg:squarederror")
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")

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
    
    def get_feature_importance(self, x, importance_type="built-in"):
        """Get feature importance of the model

        Args:
            importance_type (str, optional): Type of importance. Defaults to 'built-in'. 'built-in' or 'shap'

        Returns:
            _type_: list or np.array
        """
        if importance_type == "built-in":
            return self.model.feature_importances_
        elif importance_type == "shap":
            explainer = shap.Explainer(self.model)
            shap_values = explainer.shap_values(x)
            return shap_values