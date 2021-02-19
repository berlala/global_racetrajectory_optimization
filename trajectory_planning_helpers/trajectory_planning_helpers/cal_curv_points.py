# cal_curv
# 散点法计算曲率

import numpy as np 

def cal_curv_points(x,y):
    t_a = np.linalg.norm([x[1]-x[0] , y[1]-y[0]])
    t_b = np.linalg.norm([x[2]-x[1],y[2]-y[1]])

    M = np.array([
        [1, -t_a, t_a**2],
        [1,0,0],
        [1, t_b, t_b**2]
    ])

    a = np.matmul(np.linalg.inv(M),x)
    b = np.matmul(np.linalg.inv(M),y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    norm = [b[1], -a[1]]/np.sqrt(a[1]**2. + b[1]**2.)

    return kappa