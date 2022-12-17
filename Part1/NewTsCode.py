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

# pip install ipywidgets
warnings.filterwarnings("ignore")

"""中文显示问题"""
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

a = pd.read_csv("Yantai_Short.csv")
# a.plot(figsize=(8,6))
# plt.show()
data = np.array(a)
ts_data = []
date_data = []
for i in range(len(data)):
    ts_data.append(data[i][1])
    date_data.append(data[i][0])

plt.plot(date_data, ts_data, 'b*-', alpha=0.5, linewidth=1, label="Monthly Average Temperature")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Average temperature")
plt.show()

dif = list()
# Cancel the seasonal item
for i in range(12, len(ts_data)):
    value1 = ts_data[i] - ts_data[i - 12]
    dif.append(value1)
print(dif)


# cancel the seasonal term


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

# plt.show()

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
print(dif)
dif = pd.Series(dif)

dif = dif.dropna()

tsplot(dif, lags=60)

df = pd.DataFrame(dif[0:24 * 20])  # 4 years data
#  ADF test:
print("Augmented Dickey-Fuller Results (Removed Seasonal):\n\n")
print(ADF(df.dropna()))
print("Augmented Dickey-Fuller Results (First Difference):\n\n")
print(ADF(df.diff().dropna()))
print("Augmented Dickey-Fuller Results (Second Difference):\n\n")
print(ADF(df.diff().diff().dropna()))

# ljung box test
# acorr_ljungbox(dif, lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], boxpierce=True)
# print(acorr_ljungbox(dif, lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], boxpierce=True))
# print("======================================================")

## Yule-Walker equation calculate AR(p)
r = yw.cal_my_yule_walker(df, nlags=12)
# print(r)

# Choose model automatically
ps = range(1, 5)
d = 0
qs = range(1, 5)
Ps = range(2, 5)
D = 1
Qs = range(0, 5)
s = 12  # season length is still 12

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print(parameters_list)
len(parameters_list)  # 36

ads = pd.read_csv('Yantai_Short.csv', index_col=['dt'], parse_dates=['dt'])
# del ads["AverageTemperatureUncertainty","City","Country","Latitude","Longitude"]
test = ads[int(0.9 * len(ads)):]
train = ads[:int(0.9 * len(ads))]

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
D=1,d=0
1, (1,4,2, 1),1335.757431
2, (1,4,3, 2),1338.205024
3, (1,4,3, 1),1338.311986
4, (2,1,2, 1),1338.810774
5, (4,2,2, 1),1339.35621
6, (1,4,4, 1),1339.554042
7, (2,1,3, 2),1340.00159
8, (4,2,3, 2),1340.501258
9, (1,4,2, 2),1340.501393
10, (2,1,2, 2),1340.812784
11, (1,4,3, 3),1340.928507
12, (4,2,3, 1),1341.329314
13, (2,4,2, 1),1341.801876
14, (4,2,2, 2),1341.834231
15, (2,1,4, 3),1342.000198
16, (2,1,2, 4),1342.075799
17, (1,1,4, 3),1342.088363
18, (2,1,3, 1),1342.198768
19, (4,2,3, 3),1342.293406
20, (1,4,4, 3),1342.776693
21, (1,4,2, 3),1342.794436
22, (1,4,2, 4),1342.883536
23, (1,3,3, 2),1342.931907
24, (2,1,2, 3),1342.994483
25, (4,3,2, 1),1343.082512
26, (4,2,4, 1),1343.115353
27, (1,4,4, 2),1343.20905
28, (1,1,2, 1),1343.321181
29, (4,2,4, 2),1343.424093
30, (2,4,2, 2),1343.527476
31, (2,1,3, 3),1343.643045
32, (2,4,3, 4),1343.721418
33, (2,4,3, 1),1343.742555
34, (2,4,3, 2),1343.807231
35, (1,2,4, 3),1343.810234
36, (4,3,4, 3),1344.204028
37, (2,2,2, 1),1344.33205
38, (3,2,3, 2),1344.459208
39, (4,2,4, 3),1344.54569
40, (3,4,4, 3),1344.576054
41, (1,1,3, 2),1344.660125
42, (4,2,2, 4),1344.720661
43, (4,3,3, 1),1344.756806
44, (1,1,3, 4),1344.768748
45, (3,1,4, 3),1344.814299
46, (3,3,2, 1),1344.826381
47, (4,3,2, 2),1344.920615
48, (2,1,3, 4),1345.156213
49, (1,1,2, 2),1345.163889
50, (2,4,2, 3),1345.168141
51, (4,2,2, 3),1345.214438
52, (1,1,3, 1),1345.221464
53, (3,1,2, 1),1345.226438
54, (1,2,2, 1),1345.263375
55, (1,1,4, 4),1345.521985
56, (2,4,4, 1),1345.569709
57, (4,2,3, 4),1345.611863
58, (2,4,3, 3),1345.612036
59, (2,2,3, 2),1345.784573
60, (2,2,4, 3),1345.928783
61, (1,3,4, 3),1346.035055
62, (1,1,3, 3),1346.073669
63, (4,1,2, 1),1346.172826
64, (3,3,4, 3),1346.199978
65, (2,3,2, 1),1346.217163
66, (2,2,3, 1),1346.286497
67, (2,3,4, 3),1346.2973
68, (4,3,3, 2),1346.330531
69, (3,4,2, 1),1346.338698
70, (1,4,4, 4),1346.347182
71, (3,2,3, 1),1346.347639
72, (2,2,2, 2),1346.379569
73, (3,2,2, 1),1346.429246
74, (1,1,4, 1),1346.440066
75, (3,1,3, 2),1346.502846
76, (1,1,2, 3),1346.539695
77, (2,4,4, 3),1346.593011
78, (2,4,4, 2),1346.616198
79, (3,2,4, 3),1346.652005
80, (1,2,3, 4),1346.727056
81, (3,3,3, 1),1346.792198
82, (1,4,3, 4),1346.856857
83, (3,1,2, 2),1347.057265
84, (1,2,4, 4),1347.060846
85, (3,1,3, 1),1347.126177
86, (1,2,2, 2),1347.146307
87, (2,4,2, 4),1347.172045
88, (1,2,3, 1),1347.192383
89, (1,2,3, 2),1347.248529
90, (2,4,4, 4),1347.261619
91, (3,1,4, 4),1347.278973
92, (4,1,3, 2),1347.356764
93, (4,1,4, 3),1347.390959
94, (1,3,2, 1),1347.470231
95, (4,3,3, 3),1347.506819
96, (2,2,4, 1),1347.517047
97, (4,3,3, 4),1347.576831
98, (1,1,4, 2),1347.647636
99, (4,3,2, 3),1347.798294
100, (1,2,3, 3),1347.99322
101, (3,3,4, 1),1348.003068
102, (4,1,2, 2),1348.040728
103, (3,1,3, 3),1348.055976
104, (4,1,3, 1),1348.062539
105, (3,4,4, 2),1348.080914
106, (3,2,3, 3),1348.123835
107, (2,3,3, 1),1348.136503
108, (3,1,4, 1),1348.154679
109, (3,4,3, 1),1348.159808
110, (3,4,4, 1),1348.169784
111, (3,3,3, 2),1348.259124
112, (2,2,4, 4),1348.281545
113, (2,2,2, 3),1348.31385
114, (3,1,2, 3),1348.320398
115, (3,3,2, 2),1348.385695
116, (2,1,4, 1),1348.439833
117, (1,2,4, 1),1348.439867
118, (3,2,2, 2),1348.520021
119, (1,2,2, 3),1348.529875
120, (1,1,2, 4),1348.538664
121, (2,2,3, 3),1348.546276
122, (3,4,3, 4),1348.587347
123, (2,3,4, 4),1348.597091
124, (4,3,4, 1),1348.614976
125, (3,3,3, 4),1348.651575
126, (2,2,3, 4),1348.68797
127, (4,1,3, 3),1348.71731
128, (4,3,4, 2),1348.786793
129, (2,2,4, 2),1348.867523
130, (3,4,3, 3),1348.871555
131, (2,3,3, 2),1348.976479
132, (3,1,3, 4),1349.064969
133, (1,3,2, 2),1349.092304
134, (2,1,4, 4),1349.109047
135, (4,1,4, 1),1349.150663
136, (3,4,2, 2),1349.193421
137, (1,3,3, 1),1349.24189
138, (2,3,4, 1),1349.255494
139, (3,1,4, 2),1349.274966
140, (3,2,4, 4),1349.295983
141, (2,3,3, 4),1349.329012
142, (2,3,2, 2),1349.35875
143, (4,4,3, 2),1349.452894
144, (4,1,2, 3),1349.513514
145, (2,2,2, 4),1349.561919
146, (2,1,4, 2),1349.642996
147, (1,2,4, 2),1349.643345
148, (4,4,2, 1),1349.661341
149, (4,3,2, 4),1349.701361
150, (2,3,3, 3),1349.857696
151, (3,2,4, 1),1349.89699
152, (1,3,3, 4),1349.928714
153, (1,3,4, 4),1350.02588
154, (1,3,3, 3),1350.153885
155, (1,3,4, 1),1350.202053
156, (3,4,3, 2),1350.218882
157, (3,1,2, 4),1350.263473
158, (3,3,4, 2),1350.273845
159, (2,3,2, 3),1350.283532
160, (4,1,4, 2),1350.305205
161, (3,2,3, 4),1350.411429
162, (3,4,2, 3),1350.434633
163, (2,3,4, 2),1350.489827
164, (4,4,4, 3),1350.496251
165, (1,3,2, 3),1350.501598
166, (1,2,2, 4),1350.529302
167, (4,1,4, 4),1350.529482
168, (3,4,2, 4),1350.757491
169, (4,2,4, 4),1350.799844
170, (3,2,4, 2),1350.904882
171, (4,1,3, 4),1350.925414
172, (3,3,3, 3),1351.049999
173, (4,1,2, 4),1351.264048
174, (3,2,2, 3),1351.274161
175, (4,4,2, 2),1351.303684
176, (3,3,2, 3),1351.320455
177, (1,3,4, 2),1351.430507
178, (4,4,3, 1),1351.572859
179, (4,4,3, 4),1351.638396
180, (2,3,2, 4),1351.721317
181, (3,2,2, 4),1351.81774
182, (3,3,2, 4),1351.899221
183, (4,3,4, 4),1352.363611
184, (1,3,2, 4),1352.475765
185, (4,4,4, 1),1352.575999
186, (3,4,4, 4),1352.649987
187, (4,4,2, 3),1352.668529
188, (3,3,4, 4),1353.340938
189, (4,4,4, 2),1353.797865
190, (4,4,3, 3),1354.617192
191, (4,4,2, 4),1354.851006
192, (4,4,4, 4),1355.530731
193, (3,1,4, 0),1369.420215
194, (4,4,4, 0),1370.15431
195, (2,3,4, 0),1371.036588
196, (4,1,4, 0),1371.195429
197, (3,3,4, 0),1371.450873
198, (4,2,4, 0),1372.938058
199, (2,4,4, 0),1373.093131
200, (3,2,4, 0),1373.270205
201, (1,1,4, 0),1373.486367
202, (1,2,4, 0),1374.21873
203, (2,1,4, 0),1374.468874
204, (4,3,4, 0),1374.817768
205, (3,4,4, 0),1374.894373
206, (1,3,4, 0),1375.374955
207, (2,2,4, 0),1376.130413
208, (1,4,4, 0),1376.174223
209, (4,4,3, 0),1400.299915
210, (3,2,3, 0),1403.153321
211, (3,3,3, 0),1403.552135
212, (3,1,3, 0),1404.713956
213, (2,3,3, 0),1406.536949
214, (3,4,3, 0),1406.537583
215, (4,1,3, 0),1406.553936
216, (4,2,3, 0),1408.126025
217, (2,4,3, 0),1408.683238
218, (1,1,3, 0),1409.783211
219, (4,3,3, 0),1410.455827
220, (1,2,3, 0),1411.643329
221, (2,1,3, 0),1411.66949
222, (1,3,3, 0),1412.980209
223, (2,2,3, 0),1413.609555
224, (1,4,3, 0),1414.284747
225, (4,4,2, 0),1425.569695
226, (3,3,2, 0),1432.545189
227, (3,1,2, 0),1435.460858
228, (3,4,2, 0),1435.967327
229, (2,2,2, 0),1435.970743
230, (4,1,2, 0),1437.369576
231, (2,3,2, 0),1437.486593
232, (4,2,2, 0),1437.518919
233, (2,4,2, 0),1439.329701
234, (4,3,2, 0),1441.650236
235, (1,1,2, 0),1442.975097
236, (1,2,2, 0),1444.965211
237, (2,1,2, 0),1444.96669
238, (1,3,2, 0),1446.631799
239, (1,4,2, 0),1448.252625
240, (3,2,2, 0),1448.604835
"""

# set the parameters that give the lowest AIC

p, q, P, Q = 2, 1, 2, 1


# result_table.parameters[0]

best_model = sm.tsa.statespace.SARIMAX(train, order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())
"""
                                     SARIMAX Results                                      
==========================================================================================
Dep. Variable:                 AverageTemperature   No. Observations:                  471
Model:             SARIMAX(2, 0, 1)x(2, 1, 1, 12)   Log Likelihood                -662.405
Date:                            Sat, 17 Dec 2022   AIC                           1338.811
Time:                                    23:15:15   BIC                           1367.714
Sample:                                01-01-1970   HQIC                          1350.193
                                     - 03-01-2009                                         
Covariance Type:                              opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          1.2780      0.044     29.371      0.000       1.193       1.363
ar.L2         -0.2781      0.043     -6.405      0.000      -0.363      -0.193
ma.L1         -0.9820      0.017    -58.203      0.000      -1.015      -0.949
ar.S.L12      -0.1290      0.047     -2.771      0.006      -0.220      -0.038
ar.S.L24      -0.1665      0.048     -3.492      0.000      -0.260      -0.073
ma.S.L12      -0.9695      0.036    -26.971      0.000      -1.040      -0.899
sigma2         0.9676      0.057     16.870      0.000       0.855       1.080
===================================================================================
Ljung-Box (L1) (Q):                   0.33   Jarque-Bera (JB):                15.38
Prob(Q):                              0.57   Prob(JB):                         0.00
Heteroskedasticity (H):               0.88   Skew:                            -0.19
Prob(H) (two-sided):                  0.44   Kurtosis:                         3.81
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

tsplot(best_model.resid[12 + 1:], lags=60)


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
    forecast = model.predict(start=473, end=524)
    # data.shape[473], end=data.shape[473] + 52)
    forecast = data.sarima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s + d:], data['sarima_model'][s + d:])
    # print(forecast)
    # print("%%%%%%%%%%")
    # print(data)
    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    index = list(pd.date_range(start='2009-04-01', end='2013-09-01', freq='M'))
    # plt.axvspan(473,524,alpha=0.5,color='lightgrey')
    plt.axvspan(index[0], index[-1], alpha=0.5, color='lightgrey')
    # plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    # print(data.index[-1])
    # print(forecast.index[-1])
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)
    plt.show()


plotSARIMA(ads, best_model, 24)

# https://blog.csdn.net/itnerd/article/details/104715508
