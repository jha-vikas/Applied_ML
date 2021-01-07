# White Noise
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics


#### seet random number gen
random.seed(1)
### create white noise
series = [random.gauss(0.0, 1.0) for i in range(1000)]
series = pd.Series(series)
print(series.describe())

### line plot
plt.subplot(211)
series.plot()
plt.hlines(0, 0,1000)
#plt.show()

### histogram plot
plt.subplot(212)
series.hist(bins=50)
plt.show()

### autocorrelation
pd.plotting.autocorrelation_plot(series)
plt.show()

#################################################################
# Random Walk
random.seed(1)
random_walk = []
random_walk.append(-1 if random.random() < 0.5 else 1)
for i in range(1, 1000):
    movement = -1 if random.random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)
plt.plot(random_walk)
plt.hlines(0,1,1000)
plt.show()

## Random Walk Autocorrelation
random.seed(1)
random_walk = []
random_walk.append(-1 if random.random() < 0.5 else 1)
for i in range(1, 1000):
    movement = -1 if random.random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)
pd.plotting.autocorrelation_plot(random_walk)
#plt.hlines(0,0,1000)
plt.show()

## Random Walk and Stationarity
from statsmodels.tsa.stattools import adfuller
random.seed(1)
random_walk = []
random_walk.append(-1 if random.random() < 0.5 else 1)
for i in range(1, 1000):
    movement = -1 if random.random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)
pd.plotting.autocorrelation_plot(random_walk)
result = adfuller(random_walk)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

diff = []
for i in range(1, len(random_walk)):
    value = random_walk[i] - random_walk[i - 1]
    diff.append(value)
plt.plot(diff)
plt.show()

pd.plotting.autocorrelation_plot(diff)
plt.show()

## Predicting Random Walk
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]
# persistence
predictions = []
history = train[-1]
for i in range(len(test)):
    yhat = history
    predictions.append(yhat)
    history = test[i]

rmse = np.sqrt(sklearn.metrics.mean_squared_error(test, predictions))
print('Persistence RMSE: %.3f' % rmse)

predictions = []
history = train[-1]
for i in range(len(test)):
    yhat = history + (-1 if random.random() < 0.5 else 1)
    predictions.append(yhat)
    history = test[i]
rmse = np.sqrt(sklearn.metrics.mean_squared_error(test, predictions))
print('Random RMSE: %.3f' % rmse)

#########################################################################
# Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
### Additive
series = [i+random.randrange(10) for i in range(1,100)]
result = seasonal_decompose(series, model='additive', period=1)
result.plot()
plt.show()
### Multiplicative
series = [i**2.0 for i in range(1,100)]
result = seasonal_decompose(series, model='multiplicative', period=1)
result.plot()
plt.show()
### Airline Passengers Dataset
series = pd.read_csv('./Datasets/airline-passengers.csv', header=0, index_col=0, parse_dates=True,
                squeeze=True)
result = seasonal_decompose(series, model='multiplicative')
result.plot()
plt.show()

##########################################################################
# Trends
## Detrend by Differencing
from datetime import datetime
def parser(x):
    return datetime.strptime("190"+x, "%Y-%m")

series = pd.read_csv("./Datasets/shampoo-sales.csv", header=0, index_col=0, 
                    parse_dates=True, date_parser=parser)
X = series.values
diff = list()
for i in range(1, len(X)):
    value = X[i] - X[i - 1]
    diff.append(value)
plt.plot(X, color="red")
plt.plot(diff)
plt.hlines(0,0,36)
plt.show()
## Detrend by Model Fitting
from sklearn.linear_model import LinearRegression
#### fit linear model
X = [i for i in range(0, len(series))]
X = np.reshape(X, (len(X), 1))
y = series.values
model = LinearRegression()
model.fit(X, y)
##### calculate trend
trend = model.predict(X)
#### plot trend
plt.plot(y)
plt.plot(trend)
plt.show()
#### detrend
detrended = [y[i]-trend[i] for i in range(0, len(series))]
#### plot detrended
plt.plot(detrended)
plt.show()

# Seasonality
series = pd.read_csv("./Datasets/daily-minimum-temperatures.csv", header=0, index_col=0,
                        parse_dates=True, squeeze=True)
X = series.values
diff = list()
days_in_year = 365
for i in range(days_in_year, len(X)):
    value = X[i] - X[i - days_in_year]
    diff.append(value)
plt.plot(diff)
plt.show()

#### resampled
resample = series.resample('M')
monthly_mean = resample.mean()
print(monthly_mean.head(13))
monthly_mean.plot()
plt.show()
diff = list()
months_in_year = 12
for i in range(months_in_year, len(monthly_mean)):
    value = monthly_mean[i] - monthly_mean[i - months_in_year]
    diff.append(value)
plt.plot(diff)
plt.show()

#### deaeasonalize using month based differencing
diff = list()
days_in_year = 365
for i in range(days_in_year, len(X)):
    month_str = str(series.index[i].year-1)+'-'+str(series.index[i].month)
    month_mean_last_year = series[month_str].mean()
    value = X[i] - month_mean_last_year
    diff.append(value)
plt.plot(diff)
plt.show()

## Seasonal Adjustment with Modeling
X = [i%365 for i in range(0, len(series))]
y = series.values
degree = 4
coef = np.polyfit(X, y, degree)
print('Coefficients: %s' % coef)
#### create curve
curve = []
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree-d) * coef[d]
    curve.append(value)
# plot curve over original data
curve2 = []
p = np.poly1d(coef)
for i in range(len(X)):
    value = p(X[i])
    curve2.append(value)

plt.plot(series.values)
plt.plot(curve, color='red', linewidth=3)
plt.show()

values = series.values
diff = list()
for i in range(len(values)):
    value = values[i] - curve[i]
    diff.append(value)
plt.plot(diff)
plt.show()

##########################################################################
# Stationarity
## Augmented Dickey-Fuller test
series = pd.read_csv("./Datasets/daily-total-female-births.csv", header=0, index_col=0,
                    parse_dates=True, squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

series = pd.read_csv('./Datasets/airline-passengers.csv', header=0, index_col=0, 
                    parse_dates=True,squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

X = np.log(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
