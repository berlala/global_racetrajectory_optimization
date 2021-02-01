import numpy
import matplotlib.pyplot as plt
import math 
import csv

# -------------------------------------------------------
# This script is used to review the vehicle configuration
# -------------------------------------------------------

# GGV Data
csv_file = csv.reader(open('ggv.csv'))

v = [] # km/h
ax_max = []
ay_max = []
index  = 0

for item in csv_file:
    if index > 1:
        v.append(float(item[0]))
        ax_max.append(float(item[1]))
        ay_max.append(float(item[2]))
    index=index+1;

plt.plot(v, ax_max)
plt.plot(v, ay_max)
plt.legend(['Ax Max','Ay Max'])
plt.show()