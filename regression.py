"""Module for classification function."""
import numpy as np

from pyriemann.classification import MDM 
from pyriemann.tangentspace import FGDA, TangentSpace
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed




class KNearestNeighborRegression(MDM):

    """Classification by K-NearestNeighbor.

    Classification by nearest Neighbors. For each point of the test set, the
    pairwise distance to each element of the training set is estimated. The
    class is affected according to the majority class of the k nearest
    neighbors.

    Parameters
    ----------
    n_neighbors : int, (default: 5)
        Number of neighbors.
    metric : string | dict (default: 'riemann')
        The type of metric used for distance estimation.
        see `distance` for the list of supported metric.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the distance to the training set in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : list
        list of classes.

    See Also
    --------
    Kmeans
    MDM

    """

    def __init__(self, n_neighbors=5, metric='riemann', n_jobs=1):
        """Init."""
        # store params for cloning purpose
        self.n_neighbors = n_neighbors
        #MDM.__init__(self, metric=metric, n_jobs=n_jobs)
        self.metric = metric
        self.n_jobs = n_jobs

        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y):
        """Fit (store the training data).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : NearestNeighbor instance 
            The NearestNeighbor instance.
        """
        self.classes_ = y
        self.covmeans_ = X

        return self

    def predict(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        neighbors_location = self.classes_[np.argsort(dist)][:, 0:self.n_neighbors]
        
        #Maybe do weighted mean?
        return np.mean(neighbors_location,axis=1)

    def predict_weight(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        neighbors_location = self.classes_[np.argsort(dist)][:, 0:self.n_neighbors]
        avg_loc = np.mean(neighbors_location,axis=1)
        sum = np.sum(avg_loc,axis=0)
        weights = avg_loc/sum

        
        #Maybe do weighted mean?
        return np.sum(avg_loc*weights,axis=0)

class TSRegressor(BaseEstimator, ClassifierMixin):

    """Classification in the tangent space.

    Project data in the tangent space and apply a classifier on the projected
    data. This is a simple helper to pipeline the tangent space projection and
    a classifier. Default classifier is LogisticRegression

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    clf: sklearn classifier (default LogisticRegression)
        The classifier to apply in the tangent space

    See Also
    --------
    TangentSpace

    Notes
    -----
    .. versionadded:: 0.2.4
    """

    def __init__(self, metric='riemann', tsupdate=False,
                 clf=SGDRegressor()):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate
        self.clf = clf

    def fit(self, X, y):
        """Fit TSclassifier.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : TSclassifier. instance
            The TSclassifier. instance.
        """
        self.classes_ = np.unique(y)
        ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._pipe = make_pipeline(ts, self.clf)
        self._pipe.fit(X, y)
        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of ifloat, shape (n_trials, n_classes)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict_proba(X)


