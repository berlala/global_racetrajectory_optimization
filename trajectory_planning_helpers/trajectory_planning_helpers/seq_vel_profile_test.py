# Sequence Method   Ref DSCC2015 
import numpy as np
import math  
import matplotlib.pyplot as  plt



data = np.load('../../kanel_cl.npz') # 读取上赛道闭环轨迹结果
kappa = data['kappa']
el_lengths = data['el_lengths']

mu = 1.0 # constant mu
grav = 9.8
v_max = 75 # [m/s]
mass = 1500

def _fcn_acc(v):
    m = 1500
    r = 0.25
    T_acc_engine = max(0,480-v*10)
    acc_max = T_acc_engine*13/(m*r)
    return acc_max

def _fcn_del(v):
    m = 1500
    g = 9.8
    mu = 1 - v*0.005
    F_brake = mu*m*g
    dec_max = F_brake/m
    return dec_max


vel_kappa = []
# first process: curvature limit
for i in kappa:
    vel_kappa.append(min(v_max, math.sqrt(mu*grav/abs(i))))

plt.plot(vel_kappa)

# second process: acceleration limit
vel_acc =[]
vel_acc.append(vel_kappa[0])

for i in range(len(vel_kappa)-1):
    vel_acc.append(min(vel_kappa[i+1], math.sqrt(pow(vel_acc[i],2) + 2*_fcn_acc(vel_acc[i])*el_lengths[i])))

plt.plot(vel_acc)

# third process: deceleration limit
vel_dec = vel_acc

for i in range(len(vel_acc)-1,1,-1):
    vel_dec[i-1] = min(vel_acc[i-1], math.sqrt(max(0.0, pow(vel_dec[i],2) + 2*_fcn_del(vel_dec[i])*el_lengths[i])) )

plt.plot(vel_dec)
plt.legend(['Curv Cons','Acc cons','Dec cons'])
plt.ylabel('Spd[m/s]')
plt.show()