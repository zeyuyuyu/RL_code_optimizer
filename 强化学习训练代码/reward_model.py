"""
 pair-wise reward model using logistic-regression on cheap features.
"""
import ast, re, numpy as np
from sklearn.linear_model import LogisticRegression

def _feat(code: str) -> np.ndarray:
    toks = re.split(r'\s+', code)
    return np.array([len(code), len(toks),
                     len(list(ast.walk(ast.parse(code)))),
                     code.count('for'), code.count('if')],
                    dtype=np.float32)

class PairwiseRewardModel:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)
        self.X, self.y = [], []
        self.ready = False

    def add(self, prev: str, new: str, heuristic_improved: bool):
        self.X.append(_feat(new) - _feat(prev))
        self.y.append(int(heuristic_improved))

    def fit(self):
        if len(self.y) < 30 or len(set(self.y)) < 2:
            return
        self.clf.fit(self.X, self.y)
        self.ready = True

    def score(self, prev: str, new: str) -> float:
        if not self.ready: return 0.0
        diff = (_feat(new) - _feat(prev)).reshape(1, -1)
        return float(self.clf.predict_proba(diff)[0, 1] - 0.5)  # [-0.5, +0.5]
