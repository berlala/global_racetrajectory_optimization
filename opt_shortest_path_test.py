import numpy as np
import math
import quadprog
import time
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
import helper_funcs_glob
import os 


def opt_shortest_path(reftrack: np.ndarray,
                      normvectors: np.ndarray,
                      w_veh: float,
                      print_debug: bool = False) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    This function uses a QP solver to minimize the summed length of a path by moving the path points along their
    normal vectors within the track width.

    Please refer to the following paper for further information:
    Braghin, Cheli, Melzi, Sabbioni
    Race Driver Model
    DOI: 10.1016/j.compstruc.2007.04.028

    .. inputs::
    :param reftrack:        array containing the reference track, i.e. a reference line and the according track widths
                            to the right and to the left [x, y, w_tr_right, w_tr_left] (unit is meter, must be unclosed)
    :type reftrack:         np.ndarray
    :param normvectors:     normalized normal vectors for every point of the reference track [x_component, y_component]
                            (unit is meter, must be unclosed!)
    :type normvectors:      np.ndarray
    :param w_veh:           vehicle width in m. It is considered during the calculation of the allowed deviations from
                            the reference line.
    :type w_veh:            float
    :param print_debug:     bool flag to print debug messages.
    :type print_debug:      bool

    .. outputs::
    :return alpha_shpath:   solution vector of the optimization problem containing lateral shift in m for every point.
    :rtype alpha_shpath:    np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = reftrack.shape[0]

    # check inputs
    if no_points != normvectors.shape[0]:
        raise ValueError("Array size of reftrack should be the same as normvectors!")

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    H = np.zeros((no_points, no_points))
    f = np.zeros(no_points)

    for i in range(no_points):
        if i < no_points - 1:
            H[i, i] = H[i, i] + 2 * (math.pow(normvectors[i, 0], 2) + math.pow(normvectors[i, 1], 2)) #*2 due to 1/2*H
            H[i, i + 1] = -2 * normvectors[i, 0] * normvectors[i + 1, 0] - 2 * normvectors[i, 1] * normvectors[i + 1, 1]
            H[i + 1, i] = H[i, i + 1]   # same as *2
            H[i + 1, i + 1] = 2 * (math.pow(normvectors[i + 1, 0], 2) + math.pow(normvectors[i + 1, 1], 2))

            f[i] =f[i] + 2 * normvectors[i, 0] * reftrack[i, 0] - 2 * normvectors[i, 0] * reftrack[i + 1, 0] \
                    + 2 * normvectors[i, 1] * reftrack[i, 1] - 2 * normvectors[i, 1] * reftrack[i + 1, 1]
            f[i + 1] = -2 * normvectors[i + 1, 0] * reftrack[i, 0] \
                       - 2 * normvectors[i + 1, 1] * reftrack[i, 1] \
                       + 2 * normvectors[i + 1, 0] * reftrack[i + 1, 0] \
                       + 2 * normvectors[i + 1, 1] * reftrack[i + 1, 1]

        else:
            H[i, i] = H[i, i]+ 2 * (math.pow(normvectors[i, 0], 2) + math.pow(normvectors[i, 1], 2))
            H[i, 0] = 0.5 * 2 * (-2 * normvectors[i, 0] * normvectors[0, 0] - 2 * normvectors[i, 1] * normvectors[0, 1])
            H[0, i] = H[i, 0]
            H[0, 0] =H[0, 0] + 2 * (math.pow(normvectors[0, 0], 2) + math.pow(normvectors[0, 1], 2))

            f[i] += 2 * normvectors[i, 0] * reftrack[i, 0] - 2 * normvectors[i, 0] * reftrack[0, 0] \
                    + 2 * normvectors[i, 1] * reftrack[i, 1] - 2 * normvectors[i, 1] * reftrack[0, 1]
            f[0] = f[0] -2 * normvectors[0, 0] * reftrack[i, 0] - 2 * normvectors[0, 1] * reftrack[i, 1] \
                    + 2 * normvectors[0, 0] * reftrack[0, 0] + 2 * normvectors[0, 1] * reftrack[0, 1]

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

    # set minimum deviation to zero
    dev_max_right[dev_max_right < 0.001] = 0.001
    dev_max_left[dev_max_left < 0.001] = 0.001

    # consider value boundaries (-dev_max <= alpha <= dev_max)
    G = np.vstack((np.eye(no_points), -np.eye(no_points)))
    h = np.ones(2 * no_points) * np.append(dev_max_right, dev_max_left)

    # save start time
    t_start = time.perf_counter()

    # solve problem
    alpha_shpath = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

    # print runtime into console window
    if print_debug:
        print("Solver runtime opt_shortest_path: " + "{:.3f}".format(time.perf_counter() - t_start) + "s")

    return alpha_shpath #宽度方向上的位移


# --------------testing ------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    w_veh = 2.0
    print_debug = False

    #Input Track
    imp_opts = {"flip_imp_track": False,                # flip imported track to reverse direction
                "set_new_start": False,                 # set new starting point (changes order, not coordinates)
                "new_start": np.array([0.0, -47.0]),    # [x_m, y_m], set new starting point
                "min_track_width": None,                # [m] minimum enforced track width (set None to deactivate)
                "num_laps": 1}   
    file_paths = {"veh_params_file": "racecar.ini"}
    file_paths["track_name"] = "shanghai"    # berlin_2018 
    file_paths["module"] = os.path.dirname(os.path.abspath(__file__))
    file_paths["track_file"] = os.path.join(file_paths["module"], "inputs", "tracks", file_paths["track_name"] + ".csv")
    reftrack_imp = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
                                                                file_path=file_paths["track_file"],
                                                                width_veh=w_veh)

    #-----相当于pre_track-----------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------
    start_index = 50
    end_index = 600
    #reftrack=reftrack_imp[start_index:end_index,:] #截取开环轨迹
    reftrack=reftrack_imp #原始闭环轨迹
    reftrack_interp = tph.spline_approximation.spline_approximation(track=reftrack,
                                                                    k_reg= 3,
                                                                    s_reg= 10,
                                                                    stepsize_prep= 1.0,
                                                                    stepsize_reg= 3.0,
                                                                    debug=False)    #[2]进行近似处理，注意spline_approximation会强行闭环
    #reftrack_interp = reftrack  #不进行近似处理
    #refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_imp[end_index+1, 0:2]))  # 开环,补全最后一个
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0,0:2]))  #闭环，使收尾相等

    #Show the original track 
    # plt.figure(9)
    # plt.plot(reftrack[:,0],reftrack[:,1])
    # plt.axis('equal')
    # plt.title('Original Track')
    # plt.show()

    #-----end of pre_track-----------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------

    # check if path is closed 判断是否闭环
    if np.all(np.isclose(refpath_interp_cl[0], refpath_interp_cl[-1])):
        closed = True
        psi_s = None
        psi_e = None
        print("The track is closed.")
    else:
        closed = False
        psi_s = math.atan((refpath_interp_cl[1,1] - refpath_interp_cl[0,1])/(refpath_interp_cl[1,0] - refpath_interp_cl[0,0])) # 开环必须提供首尾航向角
        psi_e = math.atan((refpath_interp_cl[-1,1] - refpath_interp_cl[-2,1])/(refpath_interp_cl[-1,0] - refpath_interp_cl[-2,0]))
        #print(psi_s/math.pi*180)
        #print(psi_e/math.pi*180)
        print("The track is NOT closed.")

    #print("The track is closed? " +str(closed))

    [coeffs_x_refline, coeffs_y_refline, A, normvectors_refline] = tph.calc_splines.calc_splines(path = refpath_interp_cl,el_lengths= None,
                                            psi_s= psi_s, psi_e= psi_e,use_dist_scaling=False)
    #--------结束pretrack-------#

    # The track cannot be straight
    #reftrack = np.array([(0,-1,4,4),
    #                     (2,2,4,4),
    #                     (4,4,4,4),
    #                     (6,6,4,4),
    #                    (10,8,4,4),
    #                    (15,8,4,4),
    #                    (20,8,4,4)]) # （中心坐标x, 中心坐标y, 左侧宽度，右侧宽度）
    #normvectors = np.array([(0.7071,-0.7071),
    #                    (0.7071,-0.7071),
    #                    (0.7071,-0.7071),
    #                    (0.7071,-0.7071),
    #                    (0,-1),
    #                    (0,-1),
    #                    (0,-1)])  # 如何避免弯心处的坐标交汇的问题？？？

    print(np.size(reftrack_interp,0))
    print(np.size(normvectors_refline,0))   

    ratio = opt_shortest_path(reftrack_interp,normvectors_refline,w_veh,print_debug)
    
    # print (ratio) # positive is to right hand direction
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
    int_index = 0
    raceline = []
    for i in ratio:
        vec = normvectors_refline[int_index]
        base = reftrack_interp[int_index]
        track_x.append(base[0])
        track_y.append(base[1])

        bond_up_x.append(base[0] + vec[0]*-1*base[2])
        bond_up_y.append(base[1] + vec[1]*-1*base[2])
        bond_down_x.append(base[0] + vec[0]*+1*base[2])
        bond_down_y.append(base[1] + vec[1]*+1*base[2])

        result_x.append(vec[0]*i + base[0])
        result_y.append(vec[1]*i + base[1])
        raceline.append([result_x[-1],result_y[-1]])

        int_index = int_index+1
        
    raceline = np.asarray(raceline) # List转np.rray

    # plt.figure(1) # 图1.轨迹结果
    # #plt.plot(reftrack[:,0],reftrack[:,1])
    # plt.plot(track_x, track_y,'--',linewidth=0.6)
    # plt.plot(bond_up_x, bond_up_y)
    # plt.plot(bond_down_x, bond_down_y)
    # plt.plot(result_x,result_y)
    # plt.legend(['Track Center','Up','Low','Shortest Res'])
    # plt.title('Shortest Result')
    # plt.axis('equal')

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE SPEED --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # 基于优化结果alpha构建赛车线，将其处理成与refline一致的格式
    #psi_vel_opt, kappa_opt = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=coeffs_x_opt,
    #                      coeffs_y=coeffs_y_opt,
    #                      ind_spls=spline_inds_opt_interp,
    #                      t_spls=t_vals_opt_interp)

    #-----相当于pre_track-----------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------
    raceline_interp = raceline  #不进行近似处理
    if closed is True:
        raceline_interp_interp_cl = np.vstack((raceline_interp[:, :2], raceline_interp[0,0:2]))  #闭环，使收尾相等
    else:
        raceline_interp_interp_cl = np.vstack((raceline_interp[:, :2], reftrack_imp[end_index+1, 0:2]))  # 开环,补全最后一个

    #-----end of pre_track-----------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------

    #对raceline,重新进行样条曲线拟合处理，获取参数
    [coeffs_x_raceline, coeffs_y_raceline, A, normvectors_raceline] = tph.calc_splines.calc_splines(path = raceline_interp_interp_cl,el_lengths= None,
                                            psi_s= psi_s, psi_e= psi_e,use_dist_scaling=False)

    # calculate new spline lengths
    spline_lengths_raceline = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x_raceline,
                            coeffs_y=coeffs_y_raceline)
    print('==size of origin spline length ==')
    print(np.size(spline_lengths_raceline)) # size check ok


    # interpolate splines for evenly spaced raceline points
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.\
        interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                        coeffs_x=coeffs_x_raceline,
                                        coeffs_y=coeffs_y_raceline,
                                        incl_last_point=False,
                                        stepsize_approx=1.0)

    print('==size of interp raceline length ==')
    print(np.size(t_values_raceline_interp)) # size check ok

    # calculate element lengths
    s_tot_raceline = float(np.sum(spline_lengths_raceline))
    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    if closed is True:
        el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])
    else:
        el_lengths_raceline_interp_cl = el_lengths_raceline_interp

    print("====t_values_raceline_interp====")
    print(np.size(t_values_raceline_interp)) #t的总长度，t为1维,0<t<1
    #print(t_values_raceline_interp)

    # calculate heading and curvature, 对生成的raceline,计算各点的航向和曲率
    psi_vel_opt, kappa_opt = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=coeffs_x_raceline,
                        coeffs_y=coeffs_y_raceline,ind_spls=spline_inds_raceline_interp,
                        t_spls=t_values_raceline_interp)

    #print('== == kappa size == ==')
    #print(np.size(kappa_opt))

    ggv, ax_max_machines = tph.import_veh_dyn_info.\
        import_veh_dyn_info(ggv_import_path='./inputs/veh_dyn_info/ggv.csv',
                            ax_max_machines_import_path='./inputs/veh_dyn_info/ax_max_machines.csv')

    vx_profile_opt = tph.calc_vel_profile.calc_vel_profile(ggv=ggv,
                            ax_max_machines=ax_max_machines,
                            v_max=70,
                            kappa=kappa_opt,
                            el_lengths=el_lengths_raceline_interp_cl,
                            closed=closed,
                            dyn_model_exp=1.0,
                            drag_coeff=0.75,
                            v_start = 20,
                            m_veh=1200)

    # calculate longitudinal acceleration profile
    if closed is True:  
        vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
    else:
        vx_profile_opt_cl = vx_profile_opt

    ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                        el_lengths=el_lengths_raceline_interp_cl,
                                                        eq_length_output=False)

    # calculate laptime
    t_profile_cl = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_opt,
                                                    ax_profile=ax_profile_opt,
                                                    el_lengths=el_lengths_raceline_interp_cl)

    # Additional: 2015DSCC method
    #TODO kappa_opt 比 el_lengths多一个
    el_lengths_raceline_interp_fix_unclosed= np.append(el_lengths_raceline_interp,el_lengths_raceline_interp[-1] )
    vx_profile_dscc = tph.seq_vel_profile.seq_vel_profile(kappa  = kappa_opt, 
                                                         el_lengths = el_lengths_raceline_interp_fix_unclosed)
    if closed is True:  
        vx_profile_dscc_cl = np.append(vx_profile_dscc, vx_profile_dscc[0])
    else:
        vx_profile_dscc_cl = vx_profile_dscc
            
    ax_profile_dscc = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_dscc_cl,
                                                        el_lengths=el_lengths_raceline_interp_cl,
                                                        eq_length_output=False)
    t_profile_dscc = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_dscc_cl,
                                                    ax_profile=ax_profile_dscc,
                                                    el_lengths=el_lengths_raceline_interp_cl)
    # ----------------- End of DSCC --------------------

    plt.figure(2)  #速度分析
    plt.subplot(2,1,1)
    plt.plot(s_raceline_interp, vx_profile_dscc)
    plt.plot(s_raceline_interp, vx_profile_opt)
    plt.xlabel('Distance[m]')
    plt.ylabel('Spd[m/s]')
    plt.legend(['DSCC','Origin'])
    plt.title('Velocity Result Compare')
    plt.subplot(2,1,2)
    plt.plot(t_profile_dscc, vx_profile_dscc_cl)
    plt.plot(t_profile_cl, vx_profile_opt_cl)
    plt.xlabel('time[s]')
    plt.ylabel('Spd[m/s]')

    # == Lap Time == 
    laptime_origin_part = math.modf(max(t_profile_cl))
    laptime_dscc_part = math.modf(max(t_profile_dscc))

    print('== == == Lap Time == == ==')
    print('Original Method Lap Time: '+ str(int(laptime_origin_part[1]/60)) + ' min ' + 
            str(int(laptime_origin_part[1]%60))+' sec '+str(int(1000*round(laptime_origin_part[0],3))))
    print('DSCC Method Lap Time: '+ str(int(laptime_dscc_part[1]/60)) + ' min ' + 
            str(int(laptime_dscc_part[1]%60))+' sec '+str(int(1000*round(laptime_dscc_part[0],3))))

    plt.figure(3)  # 轨迹线带速度信息
    cm = plt.cm.get_cmap('cool')
    plt.plot(track_x, track_y,'--',linewidth=0.6)
    plt.plot(bond_up_x, bond_up_y)
    plt.plot(bond_down_x, bond_down_y)
    sc = plt.scatter(raceline_interp[:,0],raceline_interp[:,1], s= 6, c = vx_profile_opt*3.6, cmap = cm)
    cbar = plt.colorbar(sc) #添加速度颜色条
    cbar.set_label('Spd[km/h]') # 颜色条的单位
    plt.legend(['Track Center','Up Bound','Low Bound','Opt Res'])
    plt.xlabel("East[m]")
    plt.ylabel("North[m]")
    plt.title("Optimal Result with Speed for Shortest Path")
    plt.axis('equal')
    plt.show()


    # 保存轨迹和速度结果 
    np.savez('outputs/shorest_cl.npz',raceline_x = result_x, raceline_y = result_y,
                            bond_up_x = bond_up_x, bond_up_y = bond_up_y,
                            bond_down_x =bond_down_x, bond_down_y = bond_down_y,
                            vx_profile_opt = vx_profile_opt, t_profile_cl = t_profile_cl)   