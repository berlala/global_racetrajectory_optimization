# Compare Trajectory Result
# Bolin ZHAO

import numpy as np
import matplotlib.pyplot as plt
import csv

# Result 1: original 
csv_data = np.loadtxt('outputs/traj_race_cl.csv',comments='#',delimiter=';')
res_x = []
res_y = []
for i in csv_data:
    res_x.append(i[1])
    res_y.append(i[2])

# Result 2: Bolin Curv
data = np.load('outputs/kanel_cl.npz')
kappa = data['kappa']
el_lengths = data['el_lengths']
race_x = data['raceline_x']
race_y = data['raceline_y']

# Result 2: Bolin Shorest
data = np.load('outputs/shorest_cl.npz')
race_sx = data['raceline_x']
race_sy = data['raceline_y']

# Track info (from Result 2)
bond_ux = data['bond_up_x']
bond_uy = data['bond_up_y']
bond_dx = data['bond_down_x']
bond_dy = data['bond_down_y']

#Plot Result
plt.plot(res_x,res_y)
plt.plot(race_x,race_y)
plt.plot(race_sx,race_sy)
plt.plot(bond_ux,bond_uy,'k')
plt.plot(bond_dx,bond_dy,'k')
plt.axis('equal')
plt.legend(['Origianl','New Curv','Shortest'])
plt.show()
