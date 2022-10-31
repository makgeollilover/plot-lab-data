from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uncertainties
import uncertainties.unumpy as unp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

file = "~/Documents/KFLAB/WAERMEPUMPE/waermepumpe_MESSUNG.csv"
infile = pd.read_csv(file, sep = ';')#3630 rows

one = np.ones(20)
#one = pd.DataFrame(one)
#print(one)
dT = 1.5 #Celsius
dT = np.full(20, dT)#.reshape(-1,1)

tges = np.linspace(0, 3600)
t1 = np.linspace(0, 3600, 20, dtype = int).reshape(-1,1)
t2 = np.linspace(0, 3600, 20, dtype = int).reshape(-1,1)
#print(t)
T1 = infile[infile.columns[1]]
T1 = [T1[i] for i in t1]
T1 = np.array(T1)
#T1 = unp.uarray(T1, dT)
#print(T1)
T2 = infile[infile.columns[2]]
T2 = [T2[i] for i in t2]
T2 = np.array(T2)
T2e = unp.uarray(T2, dT)
#print(T1[0])

#
polynomial_features = PolynomialFeatures(2)
tp2 = polynomial_features.fit_transform(t2)
tp1 = polynomial_features.fit_transform(t1)


nominal = unp.nominal_values
stdevs = unp.std_devs

#create polynomial model
polyreg = make_pipeline(PolynomialFeatures(2), LinearRegression())
fitT1 = polyreg.fit(t1, T1)
#print(fitT1.named_steps['linearregression'].intercept_)
#print(fitT1.named_steps['linearregression'].coef_)
pred1 = polyreg.predict(t1)
#print(pred1.mean())
#print(stdevs(fitT1.named_steps['linearregression'].coef_))
mse1 = mean_squared_error(T1, pred1)

fitT2 = polyreg.fit(t2, T2)
model2 = sm.OLS(T2, tp2).fit()
#print(model2.summary())
fitT1 = polyreg.fit(t2, T1)
model1 = sm.OLS(T1, tp1).fit()
#print(model1.summary())
#print(fitT2.named_steps['linearregression'].intercept_)
#print(fitT2.named_steps['linearregression'].score)
pred2 = polyreg.predict(t2)
mse2 = mean_squared_error(T2, pred2)

#[[ 0.00000000e+00 -1.07317861e-02  2.11548735e-06]] T1 X0 = 0, X1 ~ -1.07e-02, X2 ~ 2.12e-06
#[[ 0.00000000e+00  1.33799510e-02 -1.80525325e-06]] T2 X0 = 0, X1 ~ 1.34e-02, X2 ~ -1.81e-06

T1_p1 = T1[0]
T1_p2 = T1[-1]
T2_p1 = T2[0]
T2_p2 = T2[-1]
fig, ax = plt.subplots()
#fig.set_figwidth(12)
#print(T1_p2)
ax.scatter(t2, T2, color = "red", label = "T2")
ax.scatter(t1, T1, color = "blue", label = "T1")

ax.scatter([0, 3600], [T1_p1, T1_p2], color = "black",
label = "T zu p1 = 3,75 +/- 0,05 bar\n und p2 = 1,83 +/- 0,05 bar")

ax.scatter([0, 3600], [T2_p1, T2_p2], color = "pink",
label = "T zu p1 = 5,30 +/- 0,05 bar\n und p2 = 16,00 +/- 0,05 bar")


ax.errorbar(t2.flatten(), T2.flatten(), dT, linestyle = "none") #x,y must be 1d arrays
ax.errorbar(t1.flatten(), T1.flatten(), dT, linestyle = "none")


ax.plot(t2, pred2, label = "Polynomial Regression T2")
ax.plot(t1, pred1, label = "Polynomial regression T1")
#
ax.set_xlabel("t / s")
ax.set_ylabel("T / °C")
ax.legend()
fig.savefig("temperaturenverlauf.png")
#plt.show()

#
#print(dT.dtype)
#print(T2.dtype)

#coefficients are [ 0.00000000e+00  1.33799510e-02 -1.80525325e-06] for F = A + BX * Cx²

#LEISTUNGSZAHL

dm = 0.01 #Kg
dm = np.full(20, dm)#.reshape(-1,1)
c = 4.19E+3
c = np.full(20, c)#.reshape(-1,1)
m=4
m = np.full(20, m)


b1 = 1.33799510e-02
b2 = -1.80525325e-06

db1 = 0.0001
db2 = 1.1e-07

koeff = [2 * b2, b1]
dkoeff = [2 * db2, db1]
Qdot = c * m * np.polyval(koeff, t2)
epsilon = Qdot / 127

#depsilon = c * dm * abs(np.polyval(koeff, t2)[:,0]) + c * m * (db1+db2)*np.ones(20)

#print(epsilon)

#GÜTEGRAD

emax = T2 / (T2 - T1)

eta = epsilon / emax

deltaT = T2 - T1

fig1, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title("Leistungszahl")
ax2.set_title("Gütegrad")
ax1.set_xlabel(r"$\Delta$ T / °C")
ax1.set_ylabel(r"Leistungszahl $\epsilon$")
ax2.set_xlabel(r"$Delta T$ / °C")
ax2.set_ylabel(r"Gütegrad $\eta$")
ax1.plot(deltaT, epsilon[:,0], label = "Leistungszahl")
ax2.plot(deltaT, eta[:,0], label = "Gütegrad")
#ax1.errorbar(deltaT, epsilon[:,0], depsilon, linestyle = "none")
ax1.legend()
ax2.legend()
plt.show()
#print(deltaT.shape)
#print(epsilon[:,0])
#print(depsilon.shape)
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.996
Model:                            OLS   Adj. R-squared:                  0.996
Method:                 Least Squares   F-statistic:                     2206.
Date:                Mon, 31 Oct 2022   Prob (F-statistic):           2.91e-21
Time:                        18:11:34   Log-Likelihood:                -13.771
No. Observations:                  20   AIC:                             33.54
Df Residuals:                      17   BIC:                             36.53
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         21.2085      0.318     66.693      0.000      20.538      21.879
x1             0.0134      0.000     32.670      0.000       0.013       0.014
x2         -1.805e-06    1.1e-07    -16.436      0.000   -2.04e-06   -1.57e-06
==============================================================================
Omnibus:                        3.600   Durbin-Watson:                   0.412
Prob(Omnibus):                  0.165   Jarque-Bera (JB):                1.432
Skew:                           0.200   Prob(JB):                        0.489
Kurtosis:                       1.752   Cond. No.                     1.64e+07
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.64e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.974
Model:                            OLS   Adj. R-squared:                  0.971
Method:                 Least Squares   F-statistic:                     317.3
Date:                Mon, 31 Oct 2022   Prob (F-statistic):           3.46e-14
Time:                        18:11:34   Log-Likelihood:                -20.313
No. Observations:                  20   AIC:                             46.63
Df Residuals:                      17   BIC:                             49.61
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         18.9210      0.441     42.899      0.000      17.990      19.852
x1            -0.0107      0.001    -18.893      0.000      -0.012      -0.010
x2          2.115e-06   1.52e-07     13.887      0.000    1.79e-06    2.44e-06
==============================================================================
Omnibus:                        1.690   Durbin-Watson:                   0.347
Prob(Omnibus):                  0.430   Jarque-Bera (JB):                0.989
Skew:                          -0.135   Prob(JB):                        0.610
Kurtosis:                       1.945   Cond. No.                     1.64e+07
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.64e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
"""