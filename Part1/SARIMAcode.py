from pandas import Series
from numpy import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yw
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import warnings
from datetime import datetime
import sys
import os
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from itertools import product
from tqdm import tqdm_notebook
from statsmodels.tsa.stattools import adfuller as ADF

from statsmodels.graphics.api import qqplot

#pip install ipywidgets
warnings.filterwarnings("ignore")

"""中文显示问题"""
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']




a = pd.read_csv("Yantai_Short.csv")
# a.plot(figsize=(8,6))
# plt.show()
data = np.array(a)
ts_data=[]
date_data=[]
for i in range(len(data)):
    ts_data.append(data[i][1])
    date_data.append(data[i][0])

plt.plot(date_data,ts_data,'b*-',alpha=0.5,linewidth=1,label="Monthly Average Temperature")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Average temperature")
plt.show()


dif = list()
# Cancel the seasonal item
for i in range(12,len(ts_data)):
    value = ts_data[i]-ts_data[i-12]
    dif.append(value)


# cancel the seasonal term
df = pd.DataFrame(dif[0:24*20]) # 4 years data
# print(df)
# plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
# fig, axes = plt.subplots(3, 2, sharex=True)
# axes[0, 0].plot(df); axes[0, 0].set_title('Original Series')
# plot_acf(df,ax=axes[0, 1],lags=30,alpha=.05)########### this part should be changed


# 1st Differencing
# axes[1, 0].plot(df.diff()); axes[1, 0].set_title('1st Order Differencing')
# plot_acf(df.diff().dropna(),ax=axes[1, 1],alpha=.05)

# 2nd Differencing
# axes[2, 0].plot(df.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
# plot_acf(df.diff().diff().dropna(), ax=axes[2, 1],alpha=.05)

#plt.show()

# PACF
# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def tsplot(y, lags=None, figsize=(10, 6), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f},lags=60'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.show()

# ts_data
dif = pd.Series(dif)
dif = dif.dropna()

tsplot(dif, lags=60)

#  ADF test:
print("Augmented Dickey-Fuller Results (Removed Seasonal):\n\n")
print(ADF(df.dropna()))
print("Augmented Dickey-Fuller Results (First Difference):\n\n")
print(ADF(df.diff().dropna()))
print("Augmented Dickey-Fuller Results (Second Difference):\n\n")
print(ADF(df.diff().diff().dropna()))

# ljung box test
acorr_ljungbox(dif, lags=[1,2,3,4,5,6,7,8,9,10,11,12], boxpierce=True)
print(acorr_ljungbox(dif, lags=[1,2,3,4,5,6,7,8,9,10,11,12], boxpierce=True))
print("======================================================")
## Yule-Walker equation calculate AR(p)
r = yw.cal_my_yule_walker(df,nlags=12)
print(r)

# Choose model automatically
ps = range(2, 5)
d = 0
qs = range(2, 5)
Ps = range(2, 5)
D = 1
Qs = range(0, 2)
s = 12 # season length is still 12

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print(parameters_list)
len(parameters_list)  # 36

ads = pd.read_csv('Yantai_Short.csv', index_col=['dt'], parse_dates=['dt'])
#del ads["AverageTemperatureUncertainty","City","Country","Latitude","Longitude"]
test = ads[int(0.9*len(ads)):]
train = ads[:int(0.9*len(ads))]

def optimizeSARIMA(parameters_list, d, D, s):
    """Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(train, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)

        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        print([param, model.aic])
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table


result_table = optimizeSARIMA(parameters_list, d, D, s)
print(result_table)
"""
      parameters          aic
0   (4, 2, 2, 1)  1339.356210
1   (4, 2, 3, 1)  1341.329314
2   (2, 4, 2, 1)  1341.801876
3   (4, 3, 2, 1)  1343.082512
4   (4, 2, 4, 1)  1343.115353
5   (2, 4, 3, 1)  1343.742555
6   (2, 2, 2, 1)  1344.332050
7   (4, 3, 3, 1)  1344.756806
8   (3, 3, 2, 1)  1344.826381
9   (2, 4, 4, 1)  1345.569709
10  (2, 3, 2, 1)  1346.217163
11  (2, 2, 3, 1)  1346.286497
12  (3, 4, 2, 1)  1346.338698
13  (3, 2, 3, 1)  1346.347639
14  (3, 2, 2, 1)  1346.429246
15  (3, 3, 3, 1)  1346.792198
16  (2, 2, 4, 1)  1347.517047
17  (3, 3, 4, 1)  1348.003068
18  (2, 3, 3, 1)  1348.136503
19  (3, 4, 3, 1)  1348.159808
20  (3, 4, 4, 1)  1348.169784
21  (4, 3, 4, 1)  1348.614976
22  (2, 3, 4, 1)  1349.255494
23  (4, 4, 2, 1)  1349.661341
24  (3, 2, 4, 1)  1349.896990
25  (4, 4, 3, 1)  1351.572859
26  (4, 4, 4, 1)  1352.575999
27  (4, 4, 4, 0)  1370.154310
28  (2, 3, 4, 0)  1371.036588
29  (3, 3, 4, 0)  1371.450873
30  (4, 2, 4, 0)  1372.938058
31  (2, 4, 4, 0)  1373.093131
32  (3, 2, 4, 0)  1373.270205
33  (4, 3, 4, 0)  1374.817768
34  (3, 4, 4, 0)  1374.894373
35  (2, 2, 4, 0)  1376.130413
36  (4, 4, 3, 0)  1400.299915
37  (3, 2, 3, 0)  1403.153321
38  (3, 3, 3, 0)  1403.552135
39  (2, 3, 3, 0)  1406.536949
40  (3, 4, 3, 0)  1406.537583
41  (4, 2, 3, 0)  1408.126025
42  (2, 4, 3, 0)  1408.683238
43  (4, 3, 3, 0)  1410.455827
44  (2, 2, 3, 0)  1413.609555
45  (4, 4, 2, 0)  1425.569695
46  (3, 3, 2, 0)  1432.545189
47  (3, 4, 2, 0)  1435.967327
48  (2, 2, 2, 0)  1435.970743
49  (2, 3, 2, 0)  1437.486593
50  (4, 2, 2, 0)  1437.518919
51  (2, 4, 2, 0)  1439.329701
52  (4, 3, 2, 0)  1441.650236
53  (3, 2, 2, 0)  1448.604835
"""


# set the parameters that give the lowest AIC
p, q, P, Q = result_table.parameters[0]

best_model = sm.tsa.statespace.SARIMAX(train, order=(p, d, q),
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())
"""
                                      SARIMAX Results                                       
============================================================================================
Dep. Variable:                   AverageTemperature   No. Observations:                  471
Model:             SARIMAX(3, 0, 3)x(4, 1, [1], 12)   Log Likelihood                -662.002
Date:                              Mon, 05 Dec 2022   AIC                           1348.003
Time:                                      14:29:30   BIC                           1397.552
Sample:                                  01-01-1970   HQIC                          1367.516
                                       - 03-01-2009                                         
Covariance Type:                                opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.4864      0.111      4.369      0.000       0.268       0.705
ar.L2         -0.8290      0.048    -17.280      0.000      -0.923      -0.735
ar.L3          0.6244      0.098      6.360      0.000       0.432       0.817
ma.L1         -0.2232      0.122     -1.829      0.067      -0.462       0.016
ma.L2          0.9179      0.033     27.589      0.000       0.853       0.983
ma.L3         -0.3542      0.115     -3.093      0.002      -0.579      -0.130
ar.S.L12      -0.1651      0.060     -2.736      0.006      -0.283      -0.047
ar.S.L24      -0.2149      0.062     -3.461      0.001      -0.337      -0.093
ar.S.L36      -0.0457      0.063     -0.720      0.471      -0.170       0.079
ar.S.L48      -0.0494      0.050     -0.978      0.328      -0.148       0.050
ma.S.L12      -0.8362      0.046    -18.262      0.000      -0.926      -0.746
sigma2         0.9925      0.063     15.726      0.000       0.869       1.116
===================================================================================
Ljung-Box (L1) (Q):                   0.12   Jarque-Bera (JB):                 8.23
Prob(Q):                              0.73   Prob(JB):                         0.02
Heteroskedasticity (H):               0.86   Skew:                            -0.20
Prob(H) (two-sided):                  0.37   Kurtosis:                         3.53
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

#tsplot(best_model.resid[12+1:], lags=60)

def plotSARIMA(series, model, n_steps):
    """Plots model vs predicted values

        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
    """

    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['sarima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['sarima_model'][:s + d] = np.NaN

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.sarima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s + d:], data['sarima_model'][s + d:])

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)
    plt.show()
plotSARIMA(ads, best_model, 24)




# https://blog.csdn.net/itnerd/article/details/104715508
