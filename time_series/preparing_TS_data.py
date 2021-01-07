import numpy as np
import pandas as pd
import math
import sklearn.metrics

series = pd.read_csv("./Datasets/daily-total-female-births.csv", header=0, index_col=0,
                        parse_dates=True, squeeze=True)

print(series.head(10))
print(series.size)
print(series["1959-01"])
print(series.describe())

#############################

# Basic Feature Engineering
series = pd.read_csv("./Datasets/daily-minimum-temperatures.csv", header=0, index_col=0,
                        parse_dates=True, squeeze=True)

dataframe = pd.DataFrame()
dataframe["month"] = series.index.month
dataframe["day"] = series.index.day
dataframe["temperature"] = series.values
print(dataframe.head(5))

## Lag features
### shift
temps = pd.DataFrame(series.values)
dataframe = pd.concat([temps.shift(1), temps], axis=1)
dataframe.columns = ['t', 't+1']
print(dataframe.head(5))

dataframe = pd.concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
dataframe.columns = ['t-2', 't-1', 't', 't+1']
print(dataframe.head(5))

## Rolling Window Statistics
### rolling()
shifted = temps.shift(1)
window = shifted.rolling(window=2)
means = window.mean()
dataframe = pd.concat([means, temps], axis=1)
dataframe.columns = ['mean(t-1,t)', 't+1']
print(dataframe.head(5))

width = 3
shifted = temps.shift(width - 1)
window = shifted.rolling(window=width)
dataframe = pd.concat([window.min(), window.mean(), window.max(), temps], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(5))

## Expanding Window Statistics
## expanding
window = temps.expanding()
dataframe = pd.concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(5))

#######################################################
# Data Visualization
from matplotlib import pyplot as plt
## Line plot
series.plot()
plt.show()

series.plot(style="k.")
plt.show()

### Grouper
groups = series.groupby(pd.Grouper(freq='A'))
years = pd.DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.plot(subplots=True, legend=False)
plt.show()

## Histogram and Density Plots
series.hist()
plt.show()

series.plot(kind="kde")
plt.show()


## Box and Whisker Plots by Interval
years.boxplot()
plt.show()

one_year = series['1990']
groups = one_year.groupby(pd.Grouper(freq='M'))
months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
months = pd.DataFrame(months)
months.columns = range(1,13)
months.boxplot()
plt.show()

## Head Maps
### Yearly
years = years.T
plt.matshow(years, interpolation=None, aspect="auto")
plt.show()

### Monthly
plt.matshow(months, interpolation=None, aspect='auto')
plt.show()

## Lag Scatter Plots
pd.plotting.lag_plot(series)
plt.show()


## Multiple scatter plots
values = pd.DataFrame(series.values)
lags = 7
columns = [values]

for i in range(1,(lags + 1)):
    columns.append(values.shift(i))

dataframe = pd.concat(columns, axis=1)
columns = ['t']

for i in range(1,(lags + 1)):
    columns.append('t-' + str(i))
dataframe.columns = columns

plt.figure(1)
for i in range(1,(lags + 1)):
    ax = plt.subplot(240 + i)
    ax.set_title('t vs t-' + str(i))
    plt.scatter(x=dataframe['t'].values, y=dataframe['t-'+str(i)].values)
plt.show()

## Autocorrelation Plots
pd.plotting.autocorrelation_plot(series)
plt.show()

##########################################################################
# Resampling and Interpolation
from datetime import datetime
## Upsampling Data
def parser(x):
    return datetime.strptime("190"+x, "%Y-%m")

series = pd.read_csv("./Datasets/shampoo-sales.csv", header=0, index_col=0, 
                    parse_dates=True, date_parser=parser)
upsampled = series.resample("D").mean()
print(upsampled.head(32))

### Interpolate
interpolated = upsampled.interpolate(method='linear')
print(interpolated.head(32))
interpolated.plot()
plt.show()

interpolated = upsampled.interpolate(method='spline', order=2)
print(interpolated.head(32))
interpolated.plot()
plt.show()

## Downsampling Data
resample = series.resample('Q')
quarterly_mean_sales = resample.mean()
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
plt.show()

resample = series.resample('A')
yearly_mean_sales = resample.sum()
print(yearly_mean_sales.head())
yearly_mean_sales.plot()
plt.show()

####################################################################
# Power Transforms
series = pd.read_csv('./Datasets/airline-passengers.csv', header=0, index_col=0, 
                    parse_dates=True,squeeze=True)
plt.figure(1)
#### line plot
plt.subplot(211)
plt.plot(series)
#### histogram
plt.subplot(212)
plt.hist(series)
plt.show()

## Square Root Transform
###--- A quadritic time series
series = [i**2 for i in range(1,100)]
plt.figure(1)
####--- line plot
plt.subplot(211)
plt.plot(series)
####--- histogram
plt.subplot(212)
plt.hist(series)
plt.show()

###-- Using square root transformation
series = pd.read_csv('./Datasets/airline-passengers.csv', header=0, index_col=0, 
                    parse_dates=True,squeeze=True)
dataframe = pd.DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = np.sqrt(dataframe['passengers'])
plt.figure(1)
####--- line plot
plt.subplot(211)
plt.plot(dataframe['passengers'])
####--- histogram
plt.subplot(212)
plt.hist(dataframe['passengers'])
plt.show()

## Log Transform
dataframe = pd.DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = np.log(dataframe['passengers'])
plt.figure(1)
####--- line plot
plt.subplot(211)
plt.plot(dataframe['passengers'])
####--- histogram
plt.subplot(212)
plt.hist(dataframe['passengers'])
plt.show()

## Box-Cox Transform
"""The square root transform and log transform belong to a class of transforms called power
transforms. The Box-Cox transform2 is a congurable data transform method that supports
both square root and log transform, as well as a suite of related transforms."""
from scipy.stats import boxcox

dataframe = pd.DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = boxcox(dataframe['passengers'], lmbda=0.0)
plt.figure(1)
####--- line plot
plt.subplot(211)
plt.plot(dataframe['passengers'])
####--- histogram
plt.subplot(212)
plt.hist(dataframe['passengers'])
plt.show()

"""We can set the lambda parameter to None (the default) and let the function 
and a statistically tuned value.
The following example demonstrates this usage, returning both the transformed
dataset and the chosen lambda value."""
dataframe = pd.DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'], lam = boxcox(dataframe['passengers'])
print('Lambda: %f' % lam)
plt.figure(1)
####--- line plot
plt.subplot(211)
plt.plot(dataframe['passengers'])
####--- histogram
plt.subplot(212)
plt.hist(dataframe['passengers'])
plt.show()


# Moving Average Smoothing
series = pd.read_csv("./Datasets/daily-total-female-births.csv", header=0, index_col=0,
                    parse_dates=True, squeeze=True)
#### tail rolling
rolling = series.rolling(window=3)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))
#### plot original and transformed dataset
series.plot()
rolling_mean.plot(color="red")
plt.show()
#### zoomed plot original and transformed dataset
series[:100].plot()
rolling_mean[:100].plot(color="red")
plt.show()

## Moving Average as Feature Engineering
df = pd.DataFrame(series.values)
width = 3
lag1 = df.shift(1)
lag3 = df.shift(2)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = pd.concat([means, lag1, df], axis=1)
dataframe.columns = ['mean', 't', 't+1']
print(dataframe)

## Moving Average as Prediction
###--- prepare situation
X = series.values
window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
###--- walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    #print([history[i] for i in range(length-window,length)])
    yhat = np.mean([history[i] for i in range(length-window,length)])
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
rmse = math.sqrt(sklearn.metrics.mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
# zoom plot
plt.plot(test[:100])
plt.plot(predictions[:100], color='red')
plt.show()

