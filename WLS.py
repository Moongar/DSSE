"""
Created on Dec 12 2019

@author: Moosa Moghimi
"""
import numpy as np
import math


# import scipy.io


def get_measurements(vnoisy, inoisy, line_name, load_power, load_nodes, pmu_loc, include_current):
    # we add each measurement type one by one
    # types: 1:Pij 2:Pi 3:Qij 4:Qi 5:|Vi| 6:theta Vi 7:|Ireal| 8:|Iimag|
    # active power from pseudo measurements
    z = -load_power[0, :]  # - sign because we want injected power
    z_type = np.array([2 * np.ones(len(load_nodes)), load_nodes,
                       np.zeros(len(load_nodes)), np.zeros(len(load_nodes))]).astype(int)

    # reactive power from pseudo measurements
    z = np.concatenate((z, -load_power[1, :]))
    z_type_temp = np.array([4 * np.ones(len(load_nodes)), load_nodes,
                            np.zeros(len(load_nodes)), np.zeros(len(load_nodes))])
    z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

    # voltage phasor magnitudes (PMU measurements)
    # first we find the pmu locations on the 3-phase system (we have 3*99 single phase nodes)
    pmu_nodes = np.sort(np.concatenate((3 * pmu_loc - 2, 3 * pmu_loc - 1, 3 * pmu_loc)))
    z = np.concatenate((z, np.absolute(vnoisy[pmu_nodes - 1])))
    z_type_temp = np.array([5 * np.ones(len(pmu_nodes)), pmu_nodes,
                            np.zeros(len(pmu_nodes)), np.zeros(len(pmu_nodes))])
    z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

    # voltage phasor phase angles (PMU measurements)
    z = np.concatenate((z, np.angle(vnoisy[pmu_nodes[1:] - 1]) - np.angle(vnoisy[0])))
    z_type_temp = np.array([6 * np.ones(len(pmu_nodes) - 1), pmu_nodes[1:],
                            np.zeros(len(pmu_nodes) - 1), np.zeros(len(pmu_nodes) - 1)])
    z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

    if include_current:
        # first we add the real part of the line current phasors
        for pm in pmu_loc:
            z = np.concatenate((z, np.real(inoisy[pm - 1])))
            z_type_temp = np.array([7 * np.ones(3), line_name[pm - 1, 0] * np.ones(3),
                                    line_name[pm - 1, 1] * np.ones(3), [1, 2, 3]])
            z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

        # now we add the imaginary part of the line current phasors
        for pm in pmu_loc:
            z = np.concatenate((z, np.imag(inoisy[pm - 1])))
            z_type_temp = np.array([8 * np.ones(3), line_name[pm - 1, 0] * np.ones(3),
                                    line_name[pm - 1, 1] * np.ones(3), [1, 2, 3]])
            z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

    return z, z_type


def state_estimation(ybus, z, ztype, err_cov, iter_max, threshold, vtrue):
    n = len(ybus)  # number of single phase nodes
    g = np.real(ybus)  # real part of the admittance matrix
    b = np.imag(ybus)  # imaginary art of the admittance matrix
    x = np.concatenate(
        ([-2 * math.pi / 3, -4 * math.pi / 3], np.tile([0, -2 * math.pi / 3, -4 * math.pi / 3], math.floor(n / 3) - 1),
         np.ones(n) * (1 + .000001 * np.random.randn(n))))  # our initial guess fot the voltage phasors
    # x = np.concatenate((np.angle(vtrue[1:]), np.abs(vtrue)))
    k = 0
    cont = True
    while k < iter_max and cont:
        v = x[n - 1:]  # voltage magnitudes
        th = np.concatenate(([0], x[0: n - 1]))  # voltage angles. we add a 0 for the reference bus
        # calculating the measurement functions h(x)
        h = np.zeros(len(z))
        for m in range(0, len(z)):
            if ztype[0, m] == 2:  # Pi active load demand at node i
                i = ztype[1, m] - 1
                for jj in range(n):
                    h[m] += v[i] * v[jj] * (g[i, jj] * math.cos(th[i] - th[jj]) + b[i, jj] * math.sin(th[i] - th[jj]))
            elif ztype[0, m] == 4:  # Qi reactive load demand at node i
                i = ztype[1, m] - 1
                for jj in range(n):
                    h[m] += v[i] * v[jj] * (g[i, jj] * math.sin(th[i] - th[jj]) - b[i, jj] * math.cos(th[i] - th[jj]))
            elif ztype[0, m] == 5:  # |Vi| voltage phasor magnitude at bus i
                i = ztype[1, m] - 1
                h[m] = v[i]
            elif ztype[0, m] == 6:  # Theta Vi voltage phasor phase angle at bus i
                i = ztype[1, m] - 1
                h[m] = th[i]
            elif ztype[0, m] == 7 or ztype[0, m] == 8:
                i = ztype[1, m] - 1  # sending node
                jj = ztype[2, m] - 1  # receiving node
                ph = ztype[3, m] - 1  # phase
                a1, b1, c1 = 3 * i + [0, 1, 2]
                a2, b2, c2 = 3 * jj + [0, 1, 2]
                yline = -ybus[np.array([a1, b1, c1])[:, None], np.array([a2, b2, c2])]
                gline = np.real(yline)
                bline = np.imag(yline)
                if ztype[0, m] == 7:  # real part of Iij phasor
                    h[m] = gline[ph, 0] * (v[a1] * math.cos(th[a1]) - v[a2] * math.cos(th[a2])) - \
                           bline[ph, 0] * (v[a1] * math.sin(th[a1]) - v[a2] * math.sin(th[a2])) + \
                           gline[ph, 1] * (v[b1] * math.cos(th[b1]) - v[b2] * math.cos(th[b2])) - \
                           bline[ph, 1] * (v[b1] * math.sin(th[b1]) - v[b2] * math.sin(th[b2])) + \
                           gline[ph, 2] * (v[c1] * math.cos(th[c1]) - v[c2] * math.cos(th[c2])) - \
                           bline[ph, 2] * (v[c1] * math.sin(th[c1]) - v[c2] * math.sin(th[c2]))
                else:  # imaginary part of Iij phasor
                    h[m] = gline[ph, 0] * (v[a1] * math.sin(th[a1]) - v[a2] * math.sin(th[a2])) + \
                           bline[ph, 0] * (v[a1] * math.cos(th[a1]) - v[a2] * math.cos(th[a2])) + \
                           gline[ph, 1] * (v[b1] * math.sin(th[b1]) - v[b2] * math.sin(th[b2])) + \
                           bline[ph, 1] * (v[b1] * math.cos(th[b1]) - v[b2] * math.cos(th[b2])) + \
                           gline[ph, 2] * (v[c1] * math.sin(th[c1]) - v[c2] * math.sin(th[c2])) + \
                           bline[ph, 2] * (v[c1] * math.cos(th[c1]) - v[c2] * math.cos(th[c2]))
            else:
                print("Measurement type not defined!")
        # print(h-z)
        # calculating the jacobian of h
        h_jacob = np.zeros([len(z), len(x)])
        for m in range(0, len(z)):
            if ztype[0, m] == 2:  # Pi active load demand at node i
                i = ztype[1, m] - 1
                for jj in range(n):
                    if jj != i:
                        if jj > 0:
                            h_jacob[m, jj - 1] = v[i] * v[jj] * (g[i, jj] * math.sin(th[i] - th[jj]) -
                                                                 b[i, jj] * math.cos(th[i] - th[jj]))
                        h_jacob[m, jj + n - 1] = v[i] * (g[i, jj] * math.cos(th[i] - th[jj]) +
                                                         b[i, jj] * math.sin(th[i] - th[jj]))
                if i > 0:
                    h_jacob[m, i - 1] = -v[i] ** 2 * b[i, i]
                    for jj in range(n):
                        h_jacob[m, i - 1] += v[i] * v[jj] * (-g[i, jj] * math.sin(th[i] - th[jj]) +
                                                             b[i, jj] * math.cos(th[i] - th[jj]))
                h_jacob[m, i + n - 1] = v[i] * g[i, i]
                for jj in range(n):
                    h_jacob[m, i + n - 1] += v[jj] * (g[i, jj] * math.cos(th[i] - th[jj]) +
                                                      b[i, jj] * math.sin(th[i] - th[jj]))

            elif ztype[0, m] == 4:  # Qi reactive load demand at node i
                i = ztype[1, m] - 1
                for jj in range(n):
                    if jj != i:
                        if jj > 0:
                            h_jacob[m, jj - 1] = v[i] * v[jj] * (-g[i, jj] * math.cos(th[i] - th[jj]) -
                                                                 b[i, jj] * math.sin(th[i] - th[jj]))
                        h_jacob[m, jj + n - 1] = v[i] * (g[i, jj] * math.sin(th[i] - th[jj]) -
                                                         b[i, jj] * math.cos(th[i] - th[jj]))
                if i > 0:
                    h_jacob[m, i - 1] = -v[i] ** 2 * g[i, i]
                    for jj in range(n):
                        h_jacob[m, i - 1] += v[i] * v[jj] * (g[i, jj] * math.cos(th[i] - th[jj]) +
                                                             b[i, jj] * math.sin(th[i] - th[jj]))
                h_jacob[m, i + n - 1] = -v[i] * b[i, i]
                for jj in range(n):
                    h_jacob[m, i + n - 1] += v[jj] * (g[i, jj] * math.sin(th[i] - th[jj]) -
                                                      b[i, jj] * math.cos(th[i] - th[jj]))

            elif ztype[0, m] == 5:  # |Vi| voltage phasor magnitude at bus i
                i = ztype[1, m] - 1
                h_jacob[m, i + n - 1] = 1

            elif ztype[0, m] == 6:  # Theta Vi voltage phasor phase angle at bus i
                i = ztype[1, m] - 1
                h_jacob[m, i - 1] = 1

            elif ztype[0, m] == 7 or ztype[0, m] == 8:
                i = ztype[1, m] - 1  # sending node
                jj = ztype[2, m] - 1  # receiving node
                ph = ztype[3, m] - 1  # phase
                a1, b1, c1 = 3 * i + [0, 1, 2]
                a2, b2, c2 = 3 * jj + [0, 1, 2]
                yline = -ybus[np.array([a1, b1, c1])[:, None], np.array([a2, b2, c2])]
                gline = np.real(yline)
                bline = np.imag(yline)
                if ztype[0, m] == 7:  # real part of Iij phasor
                    # derivatives with respect to voltage phase angles
                    if a1 > 0:
                        h_jacob[m, a1-1] = -gline[ph, 0] * v[a1] * math.sin(th[a1]) - bline[ph, 0] * v[a1] * math.cos(th[a1])
                    h_jacob[m, b1-1] = -gline[ph, 1] * v[b1] * math.sin(th[b1]) - bline[ph, 1] * v[b1] * math.cos(th[b1])
                    h_jacob[m, c1-1] = -gline[ph, 2] * v[c1] * math.sin(th[c1]) - bline[ph, 2] * v[c1] * math.cos(th[c1])
                    h_jacob[m, a2-1] = gline[ph, 0] * v[a2] * math.sin(th[a2]) + bline[ph, 0] * v[a2] * math.cos(th[a2])
                    h_jacob[m, b2-1] = gline[ph, 1] * v[b2] * math.sin(th[b2]) + bline[ph, 1] * v[b2] * math.cos(th[b2])
                    h_jacob[m, c2-1] = gline[ph, 2] * v[c2] * math.sin(th[c2]) + bline[ph, 2] * v[c2] * math.cos(th[c2])
                    # derivatives with respect to voltage magnitudes
                    h_jacob[m, a1+n-1] = gline[ph, 0] * math.cos(th[a1]) - bline[ph, 0] * math.sin(th[a1])
                    h_jacob[m, b1+n-1] = gline[ph, 1] * math.cos(th[b1]) - bline[ph, 1] * math.sin(th[b1])
                    h_jacob[m, c1+n-1] = gline[ph, 2] * math.cos(th[c1]) - bline[ph, 2] * math.sin(th[c1])
                    h_jacob[m, a2+n-1] = -gline[ph, 0] * math.cos(th[a2]) + bline[ph, 0] * math.sin(th[a2])
                    h_jacob[m, b2+n-1] = -gline[ph, 1] * math.cos(th[b2]) + bline[ph, 1] * math.sin(th[b2])
                    h_jacob[m, c2+n-1] = -gline[ph, 2] * math.cos(th[c2]) + bline[ph, 2] * math.sin(th[c2])
                else:  # imaginary part of Iij phasor
                    if a1 > 0:
                        h_jacob[m, a1-1] = gline[ph, 0] * v[a1] * math.cos(th[a1]) - bline[ph, 0] * v[a1] * math.sin(th[a1])
                    h_jacob[m, b1-1] = gline[ph, 1] * v[b1] * math.cos(th[b1]) - bline[ph, 1] * v[b1] * math.sin(th[b1])
                    h_jacob[m, c1-1] = gline[ph, 2] * v[c1] * math.cos(th[c1]) - bline[ph, 2] * v[c1] * math.sin(th[c1])
                    h_jacob[m, a2-1] = -gline[ph, 0] * v[a2] * math.cos(th[a2]) + bline[ph, 0] * v[a2] * math.sin(th[a2])
                    h_jacob[m, b2-1] = -gline[ph, 1] * v[b2] * math.cos(th[b2]) + bline[ph, 1] * v[b2] * math.sin(th[b2])
                    h_jacob[m, c2-1] = -gline[ph, 2] * v[c2] * math.cos(th[c2]) + bline[ph, 2] * v[c2] * math.sin(th[c2])
                    # derivatives with respect to voltage magnitudes
                    h_jacob[m, a1+n-1] = gline[ph, 0] * math.sin(th[a1]) + bline[ph, 0] * math.cos(th[a1])
                    h_jacob[m, b1+n-1] = gline[ph, 1] * math.sin(th[b1]) + bline[ph, 1] * math.cos(th[b1])
                    h_jacob[m, c1+n-1] = gline[ph, 2] * math.sin(th[c1]) + bline[ph, 2] * math.cos(th[c1])
                    h_jacob[m, a2+n-1] = -gline[ph, 0] * math.sin(th[a2]) - bline[ph, 0] * math.cos(th[a2])
                    h_jacob[m, b2+n-1] = -gline[ph, 1] * math.sin(th[b2]) - bline[ph, 1] * math.cos(th[b2])
                    h_jacob[m, c2+n-1] = -gline[ph, 2] * math.sin(th[c2]) - bline[ph, 2] * math.cos(th[c2])

            else:
                print("Measurement type not defined!")
        # the right hand side of the equation
        rhs = h_jacob.transpose() @ np.linalg.inv(err_cov) @ (z - h)
        # d1 = h_jacob.transpose() @ np.linalg.inv(err_cov)
        # d2 = np.linalg.inv(err_cov) @ (z-h)
        # saving to mat file
        # scipy.io.savemat('C:/Users/Moosa Moghimi/Desktop/testArrays.mat', {'d11': d1, 'd22': d2})
        # print("Array saved")
        # the gain matrix
        gain = h_jacob.transpose() @ np.linalg.inv(err_cov) @ h_jacob

        delta_x = np.linalg.solve(gain, rhs)

        x += delta_x
        if np.max(np.absolute(delta_x)) < threshold:
            cont = False
        k += 1
    v = x[n - 1:]  # voltage magnitudes
    th = np.concatenate(([0], x[0: n - 1]))  # voltage angles. we add a 0 for the reference bus
    v_phasor = v * (np.cos(th) + 1j * np.sin(th))
    return v_phasor, k
