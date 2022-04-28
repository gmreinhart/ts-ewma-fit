# ts-ewma-fit

Modeling Time Series using Exponentially Weighted Moving Averages (EWMA)

![animation](/docs/readme.gif)

This project generates fitted curves for time series (TS) using their iterated EWMAs. The fits model the recent values of the TS more accurately than the more distant past. This may be an advantage compared to more standard fitting techniques (e.g., polynomial or Gaussian kernel fits). What looks like a significant pattern during the last month may look like noise a year from now. This application also features a data reduction technique (needs a differenent implementation not using pandas). The history of a TS does not need to be stored in memory. The fits are generated using only the last value of the TS and its EWMAs which can be updated online in real-time.

## Doumentation

## Setup

No special installation instructions, standard Python modules. `tsfit.py` and `beta.py` are required. `rw.py` (generates random walks) is optional.

Tested on python3.8.

## Usage

Templates containing code samples for applications can be found in the ![notebooks](https://github.com/gmreinhart/ts-ewma-fit/tree/master/notebooks) folder.
<br>
The syntax is similar to any of the scikit-learn classes as in the example given below:

```
rwalk = rw.norm_rw(1000)
period = 100
model = TSFit(4, period=period, metric='P2')
model.fit(rwalk)
estimate = model.transform()

# now do something with the fit
estindex = rwalk.size - 1 - period
print(rwalk.iloc[estindex], estimate.iloc[estindex], model.error())
```