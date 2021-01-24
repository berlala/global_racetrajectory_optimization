import numpy as np
import math
import quadprog
# import cvxopt
import time
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
#import helper_funcs_glob


# required input
# reftrack, normvectors, A, kappa_bound, w_veh

#Test Case 1
#reftrack = np.array([(0,-1,4,4), # -1 instead of 0 in case norvec cross
#                    (2,2,4,4),
#                    (4,4,4,4),
#                    (6,6,4,4),
#                    (10,8,4,4),
#                    (15,8,4,4),
#                    (20,8,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
#refpath_interp_cl = np.vstack((reftrack[:, :2],[25,8])) 

#Test Case 2
reftrack = np.array([(0,0,4,4), 
                     (4,0,4,4),
                     (6,0,4,4),
                     (9,0,4,4),
                     (14,2,4,4),
                     (24,8,4,4),
                     (30,13,4,4),
                     (40,14,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
refpath_interp_cl = np.vstack((reftrack[:, 0:2],[50,12])) 
# TODO:normverctor 与直接用calc_splines_test.py算出的有差异，差异来源是+math.pi/2导致的。
# 但是其对结果ratio不影响

#reftrack_interp, coeffs_y, A, coeffs_x, coeffs_y = helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=reftrack,
#                                                reg_smooth_opts=pars["reg_smooth_opts"],
#                                                stepsize_opts=pars["stepsize_opts"],
#                                                debug=debug,
#                                                min_width=imp_opts["min_track_width"])

[coeffs_x, coeffs_y, A, normvectors] = tph.calc_splines.calc_splines(path = refpath_interp_cl,el_lengths= None,
                            psi_s= 0.00,psi_e= 0,use_dist_scaling=False)
print_debug = False;
plot_debug = False;

kappa_bound = 0.4;
w_veh = 2.0;

# ------------------------------------------------------------------------------------------------------------------
# PREPARATIONS -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

no_points = reftrack.shape[0]
print("========")
print("num of points")
print(no_points)
print("norm vectors")
print(normvectors) #注意normvector的方向，可能会莫名反向
print("========")

# check inputs
if no_points != normvectors.shape[0]:
    raise ValueError("Array size of reftrack should be the same as normvectors!")

if no_points * 4 != A.shape[0] or A.shape[0] != A.shape[1]:
    raise ValueError("Spline equation system matrix A has wrong dimensions!")

# create extraction matrix -> only b_i coefficients of the solved linear equation system are needed for gradient
# information
A_ex_b = np.zeros((no_points, no_points * 4), dtype=np.int)

for i in range(no_points):
    A_ex_b[i, i * 4 + 1] = 1    # 1 * b_ix = E_x * x

#e.g., 当no_points = 3 时:
#A_ex_b = array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

# create extraction matrix -> only c_i coefficients of the solved linear equation system are needed for curvature
# information
A_ex_c = np.zeros((no_points, no_points * 4), dtype=np.int)

for i in range(no_points):
    A_ex_c[i, i * 4 + 2] = 2    # 2 * c_ix = D_x * x
#A_ex_c = array([[0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]])

# invert matrix A resulting from the spline setup linear equation system and apply extraction matrix
A_inv = np.linalg.inv(A)   #A 矩阵从外部直接输入，这里只负责抽取
T_c = np.matmul(A_ex_c, A_inv)

# set up M_x and M_y matrices including the gradient information, i.e. bring normal vectors into matrix form
M_x = np.zeros((no_points * 4, no_points))
M_y = np.zeros((no_points * 4, no_points))

for i in range(no_points):
    j = i * 4

    if i < no_points - 1:
        M_x[j, i] = normvectors[i, 0]
        M_x[j + 1, i + 1] = normvectors[i + 1, 0]

        M_y[j, i] = normvectors[i, 1]
        M_y[j + 1, i + 1] = normvectors[i + 1, 1]
    else:
        M_x[j, i] = normvectors[i, 0]
        M_x[j + 1, 0] = normvectors[0, 0]  # close spline

        M_y[j, i] = normvectors[i, 1]
        M_y[j + 1, 0] = normvectors[0, 1]

# set up q_x and q_y matrices including the point coordinate information
q_x = np.zeros((no_points * 4, 1))
q_y = np.zeros((no_points * 4, 1))

for i in range(no_points):
    j = i * 4

    if i < no_points - 1:
        q_x[j, 0] = reftrack[i, 0]
        q_x[j + 1, 0] = reftrack[i + 1, 0]

        q_y[j, 0] = reftrack[i, 1]
        q_y[j + 1, 0] = reftrack[i + 1, 1]
    else:
        q_x[j, 0] = reftrack[i, 0]
        q_x[j + 1, 0] = reftrack[0, 0]

        q_y[j, 0] = reftrack[i, 1]
        q_y[j + 1, 0] = reftrack[0, 1]

# set up P_xx, P_xy, P_yy matrices
x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x)
y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)

x_prime_sq = np.power(x_prime, 2)
y_prime_sq = np.power(y_prime, 2)
x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)

curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)                   # calculate curvature denominator
curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                        where=curv_den != 0)                          # divide where not zero (diag elements)
curv_part_sq = np.power(curv_part, 2)

P_xx = np.matmul(curv_part_sq, y_prime_sq)
P_yy = np.matmul(curv_part_sq, x_prime_sq)
P_xy = np.matmul(curv_part_sq, x_prime_y_prime)

# ------------------------------------------------------------------------------------------------------------------
# SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

T_nx = np.matmul(T_c, M_x)
T_ny = np.matmul(T_c, M_y)

H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
H = H_x + H_xy + H_y
H = (H + H.T) / 2   # make H symmetric

f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
        + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
f = f_x + f_xy + f_y
f = np.squeeze(f)   # remove non-singleton dimensions

# ------------------------------------------------------------------------------------------------------------------
# KAPPA CONSTRAINTS ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

Q_x = np.matmul(curv_part, y_prime)
Q_y = np.matmul(curv_part, x_prime)

# this part is multiplied by alpha within the optimization (variable part)
E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)

# original curvature part (static part)
k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)  # multiplied by -1 as only LE conditions are poss.
con_stack = np.append(con_ge, con_le)

# ------------------------------------------------------------------------------------------------------------------
# CALL QUADRATIC PROGRAMMING ALGORITHM -----------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

"""
quadprog interface description taken from 
https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/quadprog_.py

Solve a Quadratic Program defined as:

    minimize
        (1/2) * alpha.T * H * alpha + f.T * alpha

    subject to
        G * alpha <= h
        A * alpha == b

using quadprog <https://pypi.python.org/pypi/quadprog/>.

Parameters
----------
H : numpy.array
    Symmetric quadratic-cost matrix.
f : numpy.array
    Quadratic-cost vector.
G : numpy.array
    Linear inequality constraint matrix.
h : numpy.array
    Linear inequality constraint vector.
A : numpy.array, optional
    Linear equality constraint matrix.
b : numpy.array, optional
    Linear equality constraint vector.
initvals : numpy.array, optional
    Warm-start guess vector (not used).

Returns
-------
alpha : numpy.array
        Solution to the QP, if found, otherwise ``None``.

Note
----
The quadprog solver only considers the lower entries of `H`, therefore it
will use a wrong cost function if a non-symmetric matrix is provided.
"""

# calculate allowed deviation from refline
dev_max_right = reftrack[:, 2] - w_veh / 2
dev_max_left = reftrack[:, 3] - w_veh / 2

# check that there is space remaining between left and right maximum deviation (both can be negative as well!)
if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
    raise ValueError("Problem not solvable, track might be too small to run with current safety distance!")

# consider value boundaries (-dev_max_left <= alpha <= dev_max_right)
G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
h = np.append(dev_max_right, dev_max_left)
h = np.append(h, con_stack)

# save start time
t_start = time.perf_counter()

# solve problem (CVXOPT) -------------------------------------------------------------------------------------------
# args = [cvxopt.matrix(H), cvxopt.matrix(f), cvxopt.matrix(G), cvxopt.matrix(h)]
# sol = cvxopt.solvers.qp(*args)
#
# if 'optimal' not in sol['status']:
#     print("WARNING: Optimal solution not found!")
#
# alpha_mincurv = np.array(sol['x']).reshape((H.shape[1],))

# solve problem (quadprog) -----------------------------------------------------------------------------------------
eigenvalue, featurevector = np.linalg.eig(H) #矩阵特征值，判断正定
print("===Real Part of Eigen Value of A===")
print(eigenvalue.real)
if eigenvalue.real.all() < 0:
    print("A矩阵负定！")

alpha_mincurv = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

# print runtime into console window
if print_debug:
    print("Solver runtime opt_min_curv: " + "{:.3f}".format(time.perf_counter() - t_start) + "s")

# ------------------------------------------------------------------------------------------------------------------
# CALCULATE CURVATURE ERROR ----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# calculate curvature once based on original linearization and once based on a new linearization around the solution
q_x_tmp = q_x + np.matmul(M_x, np.expand_dims(alpha_mincurv, 1))
q_y_tmp = q_y + np.matmul(M_y, np.expand_dims(alpha_mincurv, 1))

x_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x_tmp)
y_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y_tmp)

x_prime_prime = np.squeeze(np.matmul(T_c, q_x) + np.matmul(T_nx, np.expand_dims(alpha_mincurv, 1)))
y_prime_prime = np.squeeze(np.matmul(T_c, q_y) + np.matmul(T_ny, np.expand_dims(alpha_mincurv, 1)))

curv_orig_lin = np.zeros(no_points)
curv_sol_lin = np.zeros(no_points)

for i in range(no_points):
    curv_orig_lin[i] = (x_prime[i, i] * y_prime_prime[i] - y_prime[i, i] * x_prime_prime[i]) \
                        / math.pow(math.pow(x_prime[i, i], 2) + math.pow(y_prime[i, i], 2), 1.5)
    curv_sol_lin[i] = (x_prime_tmp[i, i] * y_prime_prime[i] - y_prime_tmp[i, i] * x_prime_prime[i]) \
                        / math.pow(math.pow(x_prime_tmp[i, i], 2) + math.pow(y_prime_tmp[i, i], 2), 1.5)

if plot_debug:
    plt.plot(curv_orig_lin)
    plt.plot(curv_sol_lin)
    plt.legend(("original linearization", "solution based linearization"))
    plt.show()

# calculate maximum curvature error
curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))

#-----------Result and Plot-------------#

print('==ratio==')
print(alpha_mincurv)
#print(curv_error_max)

# Plot the final result on the global frame
result_pos = []
result_x = []
result_y = []
track_x = []
track_y = []
bond_up_x = []
bond_up_y = []
bond_down_x = []
bond_down_y = []
int_index = 0  #第一个和最后一个normvector似乎不对
for i in alpha_mincurv[int_index:]: #[:-1]和[-1]用法不一样，注意
    vec = normvectors[int_index]
    base = reftrack[int_index]
    track_x.append(base[0])
    track_y.append(base[1])

    bond_up_x.append(base[0] + vec[0]*-1*base[2])
    bond_up_y.append(base[1] + vec[1]*-1*base[2])
    bond_down_x.append(base[0] + vec[0]*+1*base[3])
    bond_down_y.append(base[1] + vec[1]*+1*base[3])

    result_x.append(vec[0]*i + base[0])
    result_y.append(vec[1]*i + base[1])

    int_index = int_index+1
plt.plot(reftrack[:,0],reftrack[:,1])
plt.plot(track_x, track_y)
plt.plot(bond_up_x, bond_up_y)
plt.plot(bond_down_x, bond_down_y)
plt.plot(result_x,result_y)
plt.legend(['track center','Up','Down','Opt Res'])
plt.plot(track_x, track_y,'o')
for i in range(len(track_x)):
    plt.plot([bond_down_x[i],track_x[i],bond_up_x[i]],[bond_down_y[i],track_y[i],bond_up_y[i]],'k--')
plt.axis('equal')
plt.show()
