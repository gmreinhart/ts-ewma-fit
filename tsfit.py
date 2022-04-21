'''Implemention of TSFit'''

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from beta import optimal_h, allbetas

class TSFit(BaseEstimator):
    '''Time Series estimation by iterated EWMAs'''

    def __init__(self,
            n_iterations=1,
            h=None,
            period=None, # period for scoring and setting optimal h
            metric='L1', # point (begin of 'period'), L1, L2 or Linf
            rw_type='Gaussian'): # 'Gaussian' or 'Geometric'

        # performance issues if > 20
        # assert(n_iterations <= 20)
        self.n_iterations = n_iterations

        if h is None:
            self.h = optimal_h(n_iterations, metric) * period
            self.optimized = True
        else:
            self.h = h
            self.optimized = False
        self.period = period
        self.metric = metric
        self.rw_type = rw_type
        self.t0 = 0
        self.est_ = None
        self.slope_ = None
        self.values = None

    def fit(self, X, y=None):
        '''Computes the EWMAs, attaches them as columns to self.values
        '''
        if isinstance(X, np.ndarray):
            assert len(X.shape) == 1 # 1-dim time series only
            self.values = pd.DataFrame(X.copy(), columns=['values'])
        elif isinstance(X, pd.DataFrame):
            assert X.shape[1] == 1
            self.values = X.copy()
            self.values.columns = ['values']
        elif isinstance(X, pd.Series):
            self.values = pd.DataFrame(X.copy(), columns=['values'])
        self._ewma()
        return self

    def transform(self, drift=0, t0=None):
        '''Computes the estimates ending at t0.
            Populates the attributes self.est_ and self.slope_

            Parameters:
                drift: float
                    if non-zero, adjust the estimates for the given drift
                t0: int
                    the largest index for which the estimates are to be
                    computed. Note 0 <= t0 <= self.values.shape[0] - 1

            Returns: pd.Series of the estimates
        '''
        if t0 is None:
            t0 = self.values.shape[0] - 1
        self.t0 = t0
        m = self.n_iterations
        h = self.h
        betas = allbetas(m)

        df = self.values.iloc[:t0+1] # creates a copy, self.values safe
        df = df.reset_index(drop=True)

        # variables with a _f suffix are indexed forward, ie 0..t0
        # with a _b sufix backward, ie index 0 corresponds to t0

        b_m = betas[m].unstack().to_numpy()
        r_f = (df.shape[0] - self.values.index.values - 1) / h
        r_b = r_f[::-1]
        vander_b = np.vander(r_b, m, increasing=True)
        alpha_b = b_m.dot(vander_b.T)
        alpha_b = (alpha_b * np.exp(-r_b)).T
        alpha_b = alpha_b - b_m[:,0]
        alpha_f = alpha_b[::-1]
        sumalpha_f = alpha_f.sum(axis=1)
        diff = self._ewma_diff(t0)

        est = df.loc[df.index[-1], 'values'] + np.sum(alpha_f * diff, axis=1)
        if drift != 0:
            est -= drift * (h * r_f - (h - 0.5) * sumalpha_f)
        est = pd.Series(est)
        self._slope(t0)
        self.est_ = est
        return est

    def delta(self, drift=0):
        '''returns the estimate self.period steps ago. Note that this is similar
           to transform(). transform() has t0 fixed and p is variable
           whereas delta() p=self.period is fixed and t0 is variable (i.e., the
           result gives an estimated fixed delta for each row).

            Parameters:
                drift: float
                    if non-zero, adjust the estimates for the given drift

            Returns: pd.Series of the delta estimates
        '''
        m = self.n_iterations
        p = self.period
        h = self.h
        betas = allbetas(m)
        b_m = betas[m].unstack().to_numpy()

        vander = np.vander([p / h], m, increasing=True)
        alpha = b_m.dot(vander.T) * np.exp(-1 * p / h)
        alpha = alpha.T[0]
        alpha -= b_m[:,0]
        sumalpha = np.sum(alpha)
        ewmadiff = -self.values.diff(axis=1).dropna(axis=1)

        est = ewmadiff.dot(alpha)
        if drift != 0:
            est -= drift * (p - (h - 0.5) * sumalpha)

        return est

    def point_error(self):
        '''Returns the absolute value of the difference of the random walk
            and its estimate at period steps in the past
        '''
        index = self.t0 - self.period
        assert index >= 0
        return abs(self.values.iloc[index, 0] - self.est_[index])

    def geo_point_error(self):
        '''Same as point_score, but assumes that self.values is the log of
            the random walk
        '''
        index = self.t0 - self.period
        assert index >= 0
        return abs(np.exp(self.values.iloc[index, 0]) -
                np.exp(self.est_[index]))

    def lp_error(self, p=2):
        '''Returns the Lp norm over the last self.period values of the random
            walk.
            Parameter: p: float > 0, the 'p' of the Lp norm
        '''
        start = self.t0 - self.period
        end = self.t0 + 1
        score = abs(self.values.iloc[start:end,0] - self.est_[start:end]) ** p
        # strictly speaking we need to divide by self.period + 1, the size of
        # score. The estimate at t0 is accurate and the point "doesn't count".
        return (score.sum() / self.period) ** (1/p)

    def geo_lp_error(self, p=2):
        '''Same as lp_score but assumes that sel.fvalues is the log of the
            random walk'''
        start = self.t0 - self.period
        end = self.t0 + 1
        score = abs(np.exp(self.values.iloc[start:end,0]) -
                        np.exp(self.est_[start:end])) ** p
        return (score.sum() / self.period) ** (1/p)

    def linf_error(self):
        '''Return the Lindinity norm
        '''
        start = self.t0 -self.period
        end = self.t0 + 1
        absval = abs(self.values.iloc[start:end,0] - self.est_[start:end])
        return absval.max()

    def error(self):
        '''Returns the score depending on the value of self.metric. Note that
            the individual x_erro() methods can be called if values for
            'p' other than 1 or 2 are desired (or L infinity). If self.rw_type
            == "Geometric", it is assumed that self.values represents the log
            of the random walk.
        '''
        if self.rw_type == 'Gaussian':
            if self.metric == 'P2':
                return self.point_erro()
            if self.metric == 'L1':
                return self.lp_score(p=1)
            if self.metric == 'L2':
                return self.lp_score(p=2)
            raise ValueError('Metric ' + self.metric + ' unknown')
        if self.rw_type == 'Geometric':
            if self.metric == 'P2':
                return self.geo_point_erro()
            if self.metric == 'L1':
                return self.geo_lp_score(p=1)
            if self.metric == 'L2':
                return self.geo_lp_score(p=2)
            raise ValueError('Metric ' + self.metric + ' unknown')
        raise ValueError('Type ' + self.rw_type + ' unknown')

    # 'h' is usually an integer (measured in periods), but can be any double
    # if a double there may be an issue the way columns are named
    def _ewma(self):
        '''Add EWMAs to self.values with time horizon h periods and
            m iterations'''

        m = self.n_iterations
        h = self.h
        colname = 'ewma_' + str(h) + '_' + str(m)
        # check if EWMA already exists (from a former implementation, currently
        # fit() recreates self.values from scratch and the ewmas do not exist)
        if colname in self.values:
            return

        df = self.values # changing df changes self.values
        w = np.exp(-1 / h)
        prefix = 'ewma_' + str(h) + '_'
        for i in range(1, m+1):
            cn = prefix + str(i)
            if cn in self.values:
                prevcol = cn
                continue
            if i == 1:
                df[cn] = df['values'].ewm(alpha=1-w, adjust=False).mean()
            else:
                df[cn] = df[prevcol].ewm(alpha=1-w, adjust=False).mean()
            prevcol = cn

    def _ewma_diff(self, index):
        m = self.n_iterations
        cols = ['ewma_' + str(self.h) + '_' + str(i) for i in range(1, m+1)]
        cols = ['values'] + cols
        row = self.values[cols].iloc[index:index+1]
        return (row.iloc[-1,:] - row.iloc[-1,:].shift(1)).dropna().to_numpy()

    def _slope(self, index):
        '''Returns the slope of the fitted curve at index=index.
            Populates the attribue self.slope_'''
        m = self.n_iterations
        betas = allbetas(m)
        b_m = betas[m].unstack().to_numpy()
        diff = self._ewma_diff(index)
        if m == 1:
            deriv = diff.dot(b_m[:,0])
        else:
            deriv = diff.dot(b_m[:,0] - b_m[:,1])
        self.slope_ = deriv / self.h
