'''Module simulating random walks both Gaussian and Geometric, drawing from
    a variety of distributions
'''
import numpy as np
import pandas as pd

def simple_rw(n=2000, start_value=100, drift=0):
    '''Simple randomwalk of steps +- 1 with probability 0.5 each.
        Returns a pd.Series indexed from 0 (start_value) to n (--> size=n+1)

        Parameters: n: int: one less than the size of the random walk,
                    i.e., n is the highest index of the returned Series

                    start_value: float (int preferred), the initial value
                    of the random walk

                    drift: float: adds a drift to the random walk
    '''
    rng = np.random.default_rng()
    steps = rng.choice([-1, 1], n) + drift
    steps = np.insert(steps, 0, 0) # so first value will be start_value
    values = start_value + np.cumsum(steps)
    return pd.Series(values)

def norm_rw(n=2000, start_value=100, mu=0, sigma=1, drift=0):
    '''Random walk with step sizes distributed by the normal distribution'''
    rng = np.random.default_rng()
    steps = rng.normal(mu, sigma, n) + drift
    steps = np.insert(steps, 0, 0) # so first value will be start_value
    values = start_value + np.cumsum(steps)
    return pd.Series(values)

def rw_drift_change(n=2000, start_value=100,
        drift=0.2, driftstart=None, sigma=1):
    '''Draw from a standard normal up to driftstart, then apply a drift for
        the remaining indices. If the default arguments are taken then:
        The returned pd.Series will have 2001 elements indexed 0...2000
        As driftstart=1000, the first element that has the drift applied will
        have index 1001.
        (was used experimentally how quickly changes can be detected)
    '''
    rng = np.random.default_rng()
    if driftstart is None:
        driftstart = int(n / 2)
    steps0 = rng.normal(0, sigma, driftstart)
    stepsdrift = rng.normal(drift, sigma, n-driftstart)
    steps = np.concatenate((steps0, stepsdrift))
    steps = np.insert(steps, 0, 0) # so first value will be start_value
    values = start_value + np.cumsum(steps)
    return pd.Series(values)

def uniform_rw(low, high, n=2000, start_value=100):
    '''Random walk with step sizes distributed by the uniform distribution'''
    rng = np.random.default_rng()
    steps = rng.uniform(low, high, n)
    steps = np.insert(steps, 0, 0) # so first value will be start_value
    values = start_value + np.cumsum(steps)
    return pd.Series(values)

def generalized_rw(steps, prob, n=2000, start_value=100):
    '''Takes a probabilty distribution in form of a dictionary having as keys
        step sizes and values the corresponding probabilities
    '''
    rng = np.random.default_rng()
    raw = rng.choice(steps, n, p=prob)
    raw = np.insert(raw, 0, 0) # so first value will be start_value
    values = start_value + np.cumsum(raw)
    return pd.Series(values)

def geo_ratio_rw(n=2000, sigma=0.01, start_value=100):
    '''Random walk where the ratio of successive values are drawn from
        a normal distribution
    '''
    rng = np.random.default_rng()
    steps = rng.normal(0, sigma, n)
    values = start_value * np.cumprod(1 + steps)
    values = np.insert(values, 0, start_value)
    return pd.Series(values)

def geo_log_rw(n=2000, start_value=100):
    '''Geometric random walk without the 1/2 * sigma**2 adjustment, i.e.,
        the log of a Gaussian random walk
    '''
    rng = np.random.default_rng()
    steps = rng.normal(0, 0.01, n)
    values = start_value * np.exp(np.cumsum(steps))
    values = np.insert(values, 0, start_value)
    return pd.Series(values)

def geo_rw(n=2000, drift=0, sigma=None, dt=1, start_value=100):
    '''Geometric random walk with the 1/2 * sigma**2 adjustment, the correct
        simulation of Geometric Brownian Motion
    '''
    if sigma is None:
        # sigma chosen so that Var(S_t) = 1 if t=dt and S_0 = start_value
        sigma = np.sqrt(1 / dt * np.log(1 / (start_value**2) + 1))
    rng = np.random.default_rng()
    steps = rng.normal(0, 1, n+1)
    values = np.exp(np.cumsum((drift - 0.5 * sigma**2) * dt +
                               sigma * np.sqrt(dt) * steps))
    return pd.Series(start_value * values / values[0])
