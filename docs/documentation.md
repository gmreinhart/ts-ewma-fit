# class TSFit documentation

Regression model estimating values of a time series (or random walk) using its iterated exponentially weighted moving averages. The results are deterministic (using explicit formulae rather than performing regressions). The mathematical background can be found in this ![paper](https://github.com/gmreinhart/ts-ewma-fit/tree/master/docs/documentation.md) in the ![docs](https://github.com/gmreinhart/ts-ewma-fit/tree/master/docs) folder. The syntax is adapted to resemble typical syntax of the skikit-learn classes.

<br>

## Parameters

**n_iterations: int, default=1**
- The number of iterated EWMAs used.

**h: float, default=None**
- The time horizon of the EWMAs, given in steps (i.e., not as a percentage). If `h=None` TSFit uses the optimal time horizon that minimizes the error of the given parameters `period` and `metric` (see below).

**period: int, default=None**
- The period (in whole steps) over which the fit is optimized. Mandatory. If `period=None` an error is raised. The output of the various error methods depends on this parameter.

**metric: 'P2', 'L1' or 'L2', default='L1'**
- Used in chosing the optimal time horizon `h` and also used in the `error()` method.
  - 'P2': point approximation, `period` steps in the past. `error()` gives the mean absolute error (MAE).
  - 'L1': Minimizes the `L1` norm over the `period`. 'error()' gives the `L1' norm.
  - 'L2': Minimizes the `L2` norm over the `period`. 'error()' gives the `L2' norm.

**rw_type: 'Gaussian' or 'Geometric', default= 'Gaussian'**
- If 'Gaussian', the time series (or random walk) is assumed to be Gaussian. If 'Geometric', the time series is assumed to be a Geometric Brownian Motion process (e.g., relevant for stock market data). TSFit takes the natural logarithm of the time series before performing the fit. The results of the methods (`transform(), delta()` or any of the error methods) are automatically exponentiated back to the original scale. The point here is that one can either fit the time series directly to the algorithm (by setting `metric='Gaussin'`, or perform a geometric adjustment by setting `metric='Geometric'` (each with its own advantages and disadvantages).

<br>

## Attributes

**values: pandas Dataframe**
- The first column name 'values' is the original time series. The `fit()` method populates the DataFrame with additional columns containing the iterated EWMAs. The column name depends on the time horizon and the iteration. E.g., the column `ewma_100_3` contains the third iterated EWMA with time horizon 100.

**est_: pandas series**
- The fitted values (estimates) of the given time series, populated by the `transform()` method.

**slope_: float**
- The slope of the fitted curve at the value `t0` that was given to the `transform()` method.
- not yet implemented if parameter `rw_type='Geometric`. (`slope_=None`)

<br>

## Methods

**fit(X[, y])**
- Fit the model with time series X, Computes the EWMAs. `X` must be a one-dimensional iterable (a numpy array, a pandas Series, or a pandas DataFrame with a single column named 'values').

**transform(drift=0, t0=None)**
- Performs the regression up to index `t0` (i.e., the time series is truncated at `t0`). Returns a pandas Series with the fitted values and populates the attribute `est_`. If `drift != 0` the method adjusts for the drift given. Note there is no `fit_transform()` method, since the parameters `drift` and `t0` are passed to `transform()` but not `fit()`.

**delta()**
- estimates the delta-operator, i.e., the difference between the time series at `index` and at `index - period`. This is done for each `index > period` of the time series. Note that the `transform()` methods fixes the index `t0` and varies the steps size, whereas `delta()` fixes the step size (`=period`) and varies the index.
- not yet implemented if the time series has a drift

**point_error()**
- returns the absolute value of the time series minus its estimate (mean absolute eroor, MAE) at `period` steps in the past.

**lp_error(p=2)**
- returns the `Lp` error over the last `period` steps of the time series. `p` can be any postive float.

**linf_error()**
- return the L infinity norm (max error) over the last `period` steps. (Note: the parameter `metric` of the class cannot be set to the L infinity norm because the correct optimal time horizon is not known for this metric). 

**error()**
- calls the correct individual error methode depending on how the parameter `metric`. E.g., if `metric='L1'` this method will call `lp_error(p=1)`. 