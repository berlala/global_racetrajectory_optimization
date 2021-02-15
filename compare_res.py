# Compare Trajectory Result
# Bolin ZHAO

import numpy as np
import matplotlib.pyplot as plt
import csv

# Result 1: original 
# index: # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
csv_data = np.loadtxt('outputs/traj_race_cl.csv',comments='#',delimiter=';')
#csv_time = np.loadtxt('outputs/lap_time_matrix.csv',comments='#',delimiter=',') # 并不是圈速数据
csv_time = np.loadtxt('outputs/t_profile_cl.csv',comments='#',delimiter=',')
res_x = []
res_y = []
vel_org = []
s_org = []
t_org = []
for i in csv_data:
    res_x.append(i[1])
    res_y.append(i[2])
    vel_org.append(i[5])
    s_org.append(i[0])
for i in csv_time:
    t_org.append(i)

#print(t_org)


# Result 2: Bolin Curv
data = np.load('outputs/kanel_cl.npz')
kappa = data['kappa']
el_lengths = data['el_lengths']
race_x = data['raceline_x']
race_y = data['raceline_y']
vel_crv = data['vx_profile_opt']
t_crv = data['t_profile_cl']

# Result 2: Bolin Shorest
data = np.load('outputs/shorest_cl.npz')
race_sx = data['raceline_x']
race_sy = data['raceline_y']
vel_sht = data['vx_profile_opt']
t_sht = data['t_profile_cl']

# Track info (from Result 2)
bond_ux = data['bond_up_x']
bond_uy = data['bond_up_y']
bond_dx = data['bond_down_x']
bond_dy = data['bond_down_y']

#Plot Result
plt.figure(1)
plt.plot(res_x,res_y)
plt.plot(race_x,race_y)
plt.plot(race_sx,race_sy)
plt.plot(bond_ux,bond_uy,'k')
plt.plot(bond_dx,bond_dy,'k')
plt.axis('equal')
plt.legend(['Origianl','New Curv','Shortest'])

plt.figure(2)
plt.plot(t_org, vel_org)
plt.plot(t_crv[:-1],vel_crv)
plt.plot(t_sht[:-1],vel_sht)
plt.legend(['Origianl','New Curv','Shortest'])
plt.xlabel('Time[s]')
plt.ylabel('Spd[m/s]')
plt.show()
