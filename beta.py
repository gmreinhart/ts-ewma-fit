'''Auxiliary TSFit module handling regression coefficients (beta),
    optimal time horizons and expected square error
'''

import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

_storebeta = {} # needed to make the recursion of beta() efficient

def beta(m, i, q):
    '''Recursively computes the coefficient of beta_i^q using m iterated
        EWMAs
    '''

    global _storebeta
    if (m, i, q) in _storebeta:
        return _storebeta[(m, i, q)]

    if m <= 0 or i < 0 or q < 0:
        return 0
    if i == 0 and q == 0:
        val = ((-1) ** m) - 1
        _storebeta[(m, i, q)] = val
        return val
    if q == 0:
        val = 2 * beta(m-1, i-1, 0) - beta(m-1, i, 0)
        _storebeta[(m, i, q)] = val
        return val
    val =  4 / q * beta(m-1, i-1, q-1) - 2 / q * beta(m-1, i, q-1)\
        - 2 * beta(m-1, i-1, q) + beta(m-1, i, q)
    _storebeta[(m, i, q)] = val
    return val

def allbetas(m):
    '''Wrapper computing all necessary beta for a given m'''
    betas = {}
    for outer in range(1, m+1):
        for i in range(0, outer):
            for q in range(0, outer):
                b = beta(outer, i, q)
                betas[(outer, i, q)] = b
    index = pd.MultiIndex.from_tuples(list(betas.keys()),
                            names=['m', 'i', 'q'])
    pds = pd.Series(betas.values(), index=index)
    #return betas # not sure a dictionary of a multiindexed df is better
    return pds

# L1 and L2 computed using min_lp()
# commented values indicate expected normalized error
def optimal_h(m, metric='L2'):
    '''Returns the optimal time horizon h, depending on m and the
        score metric'''

    if m <= 0 or m > 20:
        raise ValueError('optimal_h available for 1 <= m <= 20')
    if metric == 'P1':
        opt = {
            1: 0.7959,  # 0.1855
            2: 0.3518,  # 0.1139
            3: 0.2204,  # 0.0851
            4: 0.159,   # 0.0691
            5: 0.2153,  # 0.0573
            6: 0.1624,  # 0.048
            7: 0.1295,  # 0.0416
            8: 0.1072,  # 0.0369
            9: 0.0911,  # 0.0333
            10: 0.1064, # 0.0302
            11: 0.0915, # 0.0275
            12: 0.0801, # 0.0253
            13: 0.0711, # 0.0234
            14: 0.0638, # 0.0219
            15: 0.0578, # 0.0205
            16: 0.0638, # 0.0192
            17: 0.0581, # 0.0181
            18: 0.0532, # 0.0171
            19: 0.0491, # 0.0163
            20: 0.0455, # 0.0155
        }
    elif metric == 'L1':
        # commented values are precise expected normalized L1 errors
        opt = {
            1: 0.496,  # 0.2683
            2: 0.266,  # 0.2062
            3: 0.181,  # 0.1742
            4: 0.137,  # 0.1537
            5: 0.11,   # 0.1392
            6: 0.092,  # 0.1282
            7: 0.079,  # 0.1194
            8: 0.07,   # 0.1123
            9: 0.079,  # 0.106
            10: 0.07,  # 0.1006
            11: 0.063, # 0.0959
            12: 0.057, # 0.0918
            13: 0.052, # 0.0883
            14: 0.048, # 0.0851
            15: 0.044, # 0.0822
            16: 0.041, # 0.0797
            17: 0.038, # 0.0773
            18: 0.036, # 0.0752
            19: 0.034, # 0.0732
            20: 0.032, # 0.0714
        }
    elif metric == 'L2':
        # commented values are upper bounds for the expected L2 error
        opt = {
            1: 0.528,  # 0.3448
            2: 0.275,  # 0.2665
            3: 0.186,  # 0.2259
            4: 0.14,   # 0.2
            5: 0.112,  # 0.1814
            6: 0.094,  # 0.1674
            7: 0.11,   # 0.1562
            8: 0.094,  # 0.147
            9: 0.082,  # 0.1376
            10: 0.072, # 0.1306
            11: 0.064, # 0.1246
            12: 0.058, # 0.1193
            13: 0.053, # 0.1148
            14: 0.049, # 0.1107
            15: 0.045, # 0.107
            16: 0.042, # 0.1037
            17: 0.039, # 0.1007
            18: 0.042, # 0.098
            19: 0.039, # 0.0953
            20: 0.037, # 0.0929
        }
    else:
        raise ValueError('Unknown metric', metric)
    return opt[m]

def _highest_beta(m):
    '''internal method needed only for sse()'''
    def func(r):
        val = -beta(m, m-1, 0)
        for i in range(0, m):
            val += beta(m, m-1, i) * (r ** i) * np.exp(-r)
        return val
    return func

def sse(m):
    '''Returns the expected square error function'''
    @np.vectorize
    def func(r):
        if r == 0:
            return 1.0
        val = 0
        for i in range(1, m+1):
            val += (_highest_beta(i)(r) / (2 ** i)) ** 2
        return 1 - 2 / r * val
    return func

def min_lp(m=2, lp=2):
    '''Compute the theoretical Lp error. Return the optimal time horizon and
        the expected normalized error.
        This method was used to fill the dictionaries of 'optimal_h()'
    '''
    def error(p, h):
        return (p * sse(m)(p/h)) ** (lp/2)
    minval = 10000
    for h in np.arange(0.01, 0.9, 0.001):
        fx = lambda x: error(x, h)
        z = integrate.quad(fx, 0, 1)[0]
        z = z ** (1/lp)
        if lp == 1:
            z *= np.sqrt(2 / np.pi)
        if z < minval:
            minval = z
            hmin = h
    return hmin, minval

def min_point(m):
    '''Finds the optimal h and expected minimum SSE'''
    xrange = np.linspace(1, m*2, 10000)
    y = sse(m)(xrange)
    minval = np.argmin(y)
    print(m, 1/xrange[minval], y[minval])

if __name__ == '__main__':
    #pass
    for lp in [2, 1]:
        for m in range(1, 21):
            print(lp, m, min_lp(m, lp))
