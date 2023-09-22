import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数
import math 
import csv

# -------------------------------------------------------
# This script is used to review the vehicle configuration
# -------------------------------------------------------
# 对高性能轮胎和赛道，附着系数可以大于1。

# GGV Data
csv_file_ggv = csv.reader(open('ggv.csv'))
csv_file_motor = csv.reader(open('ax_max_machines.csv'))

v_ggv = [] # m/s
v_motor = []
ax_ggv_max = []
ay_ggv_max = []
ax_motor = []
index_1  = 0
index_2 = 0

for item in csv_file_ggv:
    if index_1 > 1:
        v_ggv.append(float(item[0]))
        ax_ggv_max.append(float(item[1]))
        ay_ggv_max.append(float(item[2]))
    index_1=index_1+1;

for item in csv_file_motor:
    if index_2 > 1:
        v_motor.append(float(item[0]))
        ax_motor.append(float(item[1]))
    index_2=index_2+1;

plt.plot(v_ggv, ax_ggv_max)
plt.plot(v_ggv, ay_ggv_max)
plt.plot(v_motor, ax_motor)
plt.legend(['Ax GGV Max','Ay GGV Max','Ax Powertrain'])
plt.xlabel('Spd[m/s]')
plt.ylabel('Acc[m/s^2]')
plt.show()

fig2 = plt.figure()
plt.plot(ax_ggv_max,ay_ggv_max,'o')
plt.xlabel('Acc X [m/s]')
plt.ylabel('Acc Y [m/s^2]')
plt.show()