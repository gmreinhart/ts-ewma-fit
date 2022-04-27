# ts-ewma-fit
Modeling Time Series using Exponentially Weighted Moving Averages (EWMA)

![animation](/docs/readme.gif)

## Usage
Templates containing code sample for usage can be found in the ![notebooks](https://github.com/gmreinhart/ts-ewma-fit/git/notbooks) folder. A typical application usually follows along the basic setup:

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