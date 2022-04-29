# ts-ewma-fit

Modeling Time Series using Exponentially Weighted Moving Averages (EWMA)

![animation](/docs/readme.gif)

This project generates fitted curves for time series (TS) using their iterated EWMAs. The fits model the recent values of the TS more accurately than the more distant past. This may be an advantage compared to more standard fitting techniques (such as polynomial or Gaussian kernel fits). What looks like a significant pattern during the last month may look like noise a year from now. This application also serves as a data reduction technique since the history of a TS does not need to be stored in memory. The fits are generated using only the last value of the TS and its EWMAs which can be updated online in real-time (needs a differenent implementation not using pandas). 

## Doumentation

The ![documentation](https://github.com/gmreinhart/ts-ewma-fit/tree/master/docs/documentation.md) of the class `TSFit` can be found in the ![docs](https://github.com/gmreinhart/ts-ewma-fit/tree/master/docs) folder. 

The mathematical background can be found in this ![paper](https://github.com/gmreinhart/ts-ewma-fit/tree/master/docs/tsfit.pdf) in the ![docs](https://github.com/gmreinhart/ts-ewma-fit/tree/master/docs) folder. 

## Setup

No special installation instructions, standard Python modules. `tsfit.py` and `beta.py` are required. `rw.py` (generates random walks) is optional.

Tested on python3.8.

## Usage

Templates containing code samples for applications can be found in the ![notebooks](https://github.com/gmreinhart/ts-ewma-fit/tree/master/notebooks) folder.
<br>
The syntax is similar to any of the scikit-learn classes as in the example given below:

```
import numpy as np
import pandas as pd
from tsfit import TSFit
import rw

rwalk = rw.norm_rw(1000)
period = 100
model = TSFit(4, period=period, metric='P2')
model.fit(rwalk)
estimate = model.transform()

# now do something with the fit
estindex = rwalk.size - 1 - period
print(rwalk.iloc[estindex], estimate.iloc[estindex], model.error())
```