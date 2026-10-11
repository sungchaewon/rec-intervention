"""Two-stage harm-aware router: Gate (intervene at all?) + Selector (which action?).

Gate loss is weighted BCE. Users some action helps are rare (a few percent),
so with `balance_classes` positives are up-weighted by n_negative / n_positive;
on top of that, users the base already serves (base metric > 0) and whom no
action helps get `harm_weight`: intervening there can only harm.
The gate threshold is chosen on out-of-fold predictions within the fit data,
so the evaluation period is never used for model selection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recint.routing.data import RoutingData
from recint.routing.policy import NO_OP, realized

CLASSIFIERS = ("logistic", "gbdt")
LOGISTIC_MAX_ITER = 1000


def make_classifier(kind: str, seed: int) -> ClassifierMixin:
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=LOGISTIC_MAX_ITER))
    if kind == "gbdt":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown classifier {kind!r}; expected one of {CLASSIFIERS}")


def _fit(classifier: ClassifierMixin, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> ClassifierMixin:
    if weight is None:
        return classifier.fit(x, y)
    if hasattr(classifier, "steps"):  # pipeline: route the weight to the final estimator
        return classifier.fit(x, y, **{f"{classifier.steps[-1][0]}__sample_weight": weight})
    return classifier.fit(x, y, sample_weight=weight)


def _proba(classifier: ClassifierMixin, x: np.ndarray, n_classes: int, offset: int = 0) -> np.ndarray:
    """Class probabilities as an (n, n_classes) matrix even if some classes were unseen.
    Class label c maps to column c + offset."""
    proba = np.zeros((len(x), n_classes))
    proba[:, classifier.classes_.astype(int) + offset] = classifier.predict_proba(x)
    return proba


class _ConstantClassifier:
    """Stand-in when the training labels contain a single class."""

    def __init__(self, label: int) -> None:
        self.classes_ = np.array([label])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.ones((len(x), 1))


def _fit_or_constant(kind: str, seed: int, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None):
    if len(np.unique(y)) < 2:
        return _ConstantClassifier(int(y[0]) if len(y) else 0)
    return _fit(make_classifier(kind, seed), x, y, weight)


class GateSelectorRouter:
    def __init__(
        self,
        harm_weight: float,
        thresholds: Sequence[float],
        classifier: str,
        n_folds: int,
        seed: int,
        balance_classes: bool = True,
    ) -> None:
        if harm_weight < 1:
            raise ValueError("harm_weight must be >= 1")
        self.harm_weight = harm_weight
        self.balance_classes = balance_classes
        self.thresholds = list(thresholds)
        self.classifier = classifier
        self.n_folds = n_folds
        self.seed = seed
        self.threshold: float | None = None

    def _gate_targets(self, data: RoutingData) -> tuple[np.ndarray, np.ndarray]:
        should_intervene = (data.gains.max(axis=1) > 0).astype(int)
        protect = (should_intervene == 0) & (data.base > 0)
        weight = np.where(protect, self.harm_weight, 1.0)
        n_positive = should_intervene.sum()
        if self.balance_classes and 0 < n_positive < len(should_intervene):
            weight = np.where(should_intervene == 1, (len(should_intervene) - n_positive) / n_positive, weight)
        return should_intervene, weight

    def _fit_models(self, data: RoutingData, rows: np.ndarray):
        z, weight = self._gate_targets(data)
        x = data.features[rows]
        gate = _fit_or_constant(self.classifier, self.seed, x, z[rows], weight[rows])
        positive = rows[z[rows] == 1]
        selector = _fit_or_constant(
            self.classifier, self.seed, data.features[positive], data.selector_labels()[positive]
        )
        return gate, selector

    def _choices(self, gate, selector, features: np.ndarray, n_actions: int, threshold: float) -> np.ndarray:
        p_intervene = _proba(gate, features, 2)[:, 1]
        action = _proba(selector, features, n_actions).argmax(axis=1)
        return np.where(p_intervene > threshold, action, NO_OP)

    def fit(self, data: RoutingData) -> GateSelectorRouter:
        n_users, n_actions = data.gains.shape
        z, _ = self._gate_targets(data)
        p_out_of_fold = np.zeros(n_users)
        action_out_of_fold = np.zeros(n_users, dtype=np.int64)
        folds = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        for fit_rows, held_rows in folds.split(data.features, z):
            gate, selector = self._fit_models(data, fit_rows)
            held = data.features[held_rows]
            p_out_of_fold[held_rows] = _proba(gate, held, 2)[:, 1]
            action_out_of_fold[held_rows] = _proba(selector, held, n_actions).argmax(axis=1)

        def policy_gain(threshold: float) -> float:
            choices = np.where(p_out_of_fold > threshold, action_out_of_fold, NO_OP)
            return realized(data.gains, choices).mean()

        # Highest gain; among ties the most conservative (largest) threshold.
        self.threshold = max(self.thresholds, key=lambda t: (policy_gain(t), t))
        self._gate, self._selector = self._fit_models(data, np.arange(n_users))
        self._n_actions = n_actions
        return self

    def choose(self, data: RoutingData) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Call fit() before choose()")
        return self._choices(self._gate, self._selector, data.features, self._n_actions, self.threshold)


def make_regressor(kind: str, seed: int):
    """Regression counterpart of `make_classifier`: logistic -> ridge, gbdt -> gbdt."""
    if kind == "logistic":
        return make_pipeline(StandardScaler(), Ridge())
    if kind == "gbdt":
        return HistGradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Unknown regressor {kind!r}; expected one of {CLASSIFIERS}")


class GainRegressionRouter:
    """Uplift-style router: regress each action's gain, apply the best predicted action
    if its prediction exceeds a margin chosen on out-of-fold predictions.

    Margin candidates are -inf (always apply the best predicted action, i.e. close to the
    best uniform action), +inf (never intervene), and non-negative quantiles of the
    predictions: applying an action whose predicted gain is negative is never chosen
    except through the always-apply candidate.
    """

    def __init__(self, regressor: str, n_folds: int, seed: int, n_margins: int = 21) -> None:
        self.regressor = regressor
        self.n_folds = n_folds
        self.seed = seed
        self.n_margins = n_margins
        self.margin: float | None = None

    def _fit_models(self, features: np.ndarray, gains: np.ndarray) -> list:
        return [make_regressor(self.regressor, self.seed).fit(features, gains[:, a]) for a in range(gains.shape[1])]

    @staticmethod
    def _predict(models: list, features: np.ndarray) -> np.ndarray:
        return np.column_stack([model.predict(features) for model in models])

    def fit(self, data: RoutingData) -> GainRegressionRouter:
        predicted = np.zeros_like(data.gains)
        for fit_rows, held_rows in KFold(self.n_folds, shuffle=True, random_state=self.seed).split(data.features):
            models = self._fit_models(data.features[fit_rows], data.gains[fit_rows])
            predicted[held_rows] = self._predict(models, data.features[held_rows])
        best, action = predicted.max(axis=1), predicted.argmax(axis=1)
        quantiles = np.quantile(best, np.linspace(0.0, 1.0, self.n_margins))
        margins = np.r_[-np.inf, 0.0, quantiles[quantiles > 0], np.inf]

        def policy_gain(margin: float) -> float:
            return realized(data.gains, np.where(best > margin, action, NO_OP)).mean()

        # Highest out-of-fold gain; among ties the most conservative (largest) margin.
        self.margin = max(margins, key=lambda m: (policy_gain(m), m))
        self._models = self._fit_models(data.features, data.gains)
        return self

    def choose(self, data: RoutingData) -> np.ndarray:
        if self.margin is None:
            raise RuntimeError("Call fit() before choose()")
        predicted = self._predict(self._models, data.features)
        return np.where(predicted.max(axis=1) > self.margin, predicted.argmax(axis=1), NO_OP)


class SingleStageRouter:
    """MetaSelector-style ablation: one classifier over {NO_OP} + actions, no gate, no harm weight."""

    def __init__(self, classifier: str, seed: int) -> None:
        self.classifier = classifier
        self.seed = seed
        self._model = None

    @staticmethod
    def _labels(data: RoutingData) -> np.ndarray:
        best = data.selector_labels()
        helps = np.take_along_axis(data.gains, best[:, None], axis=1).ravel() > 0
        return np.where(helps, best, NO_OP)

    def fit(self, data: RoutingData) -> SingleStageRouter:
        self._model = _fit_or_constant(self.classifier, self.seed, data.features, self._labels(data))
        self._n_actions = data.gains.shape[1]
        return self

    def choose(self, data: RoutingData) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before choose()")
        # Column 0 is NO_OP (-1), column a + 1 is action a.
        proba = _proba(self._model, data.features, self._n_actions + 1, offset=1)
        return proba.argmax(axis=1) - 1
