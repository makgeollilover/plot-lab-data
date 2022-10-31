from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uncertainties
import uncertainties.unumpy as unp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import scipy.odr

file = "~/Documents/KFLAB/WAERMEPUMPE/waermepumpe_MESSUNG.csv"
infile = pd.read_csv(file, sep = ';')#for the terrible German standard #3630 rows

one = np.ones(20)
#one = pd.DataFrame(one)
#print(one)
dT = 1.5 #Celsius
dT = np.full(20, dT)#.reshape(-1,1)
dm = 0.01 #Kg
dm = np.full(20, dm)#.reshape(-1,1)
c = 4.19E+3
c = np.full(20, c)#.reshape(-1,1)
m=4
m = np.full(20, m)
tges = np.linspace(0, 3600)
t = np.linspace(0, 3600, 20, dtype = int).reshape(-1,1)
#print(t)
T1 = infile[infile.columns[1]]
T1 = [T1[i] for i in t]
T1 = np.array(T1)
#T1 = unp.uarray(T1, dT)
#print(T1)
T2 = infile[infile.columns[2]]
T2 = [T2[i] for i in t]
T2 = np.array(T2)
T2e = unp.uarray(T2, dT)
#print(T2)


nominal = unp.nominal_values
stdevs = unp.std_devs

#create polynomial model
polyreg = make_pipeline(PolynomialFeatures(2), LinearRegression())
fitT1 = polyreg.fit(t, T1)
fitT2 = polyreg.fit(t, T2)

fig, ax = plt.subplots()
#fig.set_figwidth(12)

ax.scatter(t, T2, color = "red", label = "T2")
ax.errorbar(t.flatten(), T2.flatten(), dT) #x,y must be 1d arrays
ax.plot(t, polyreg.predict(t), label = "Polynomial Regression")
#
ax.set_xlabel("t / s")
ax.set_ylabel("T / °C")
ax.legend()
plt.show()

#
#print(dT.dtype)
#print(T2.dtype)