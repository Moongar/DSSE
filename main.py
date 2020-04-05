"""
Created on Dec 9 2019

@author: Moosa Moghimi
"""

import scipy.io as spio
import math
import numpy as np
import DSSStartup
import RunPF
import WLS
from matplotlib import pyplot as plt

# parameters for different runs
delay = 1  # number of semples that PMU data is taking to be processed and communicated to us
currentPhasor = True  # if you want to consider the current phasor readings from PMU
resol = 60  # data resolution in sec. we take a PMU reading every resol seconds
PMUnumber = 5  # number of installed PMUs
sigmaPMU = .01 * .05 / 3  # micro-PMU accuracy

# defining parameters
priNodes = 33  # primary nodes of the original system
secNodes = 55  # number of load nodes in the secondary power system
dur = 3600  # the pseudo meas. will be available for every dur seconds
PMUmap = [1, 33, 32, 31, 18, 17, 30, 16, 29, 15, 14, 13, 28, 12, 11,
          10, 9, 8, 27, 26, 7, 6, 25, 24, 5, 4, 23, 3, 22, 21, 20, 19, 2]
PMUnodes = np.sort(np.array(PMUmap[0:PMUnumber + 1]))

# perunit values
sBase = 100e3 / 3  # single phase power base
vBasePri = 12.66e3 / math.sqrt(3)  # single phase primary network voltage base
vBaseSec = .416 / math.sqrt(3)  # single phase secondary network voltage base
yBase = sBase / vBasePri ** 2  # base admittance
zBase = 1 / yBase  # base impedance
Ibase = sBase / vBasePri  # base current

# loading Ybus
mat = spio.loadmat('C:/OpenDSS/DSSE33DetailedMultiPhase/Ybus99.mat', squeeze_me=True)
Ybus = np.array(mat['Y'])/yBase  # adimittance matrix of the system for primary nodes only

# loading load demand data
caseName = 'loadR' + str(resol) + 'Dur3600'
mat = spio.loadmat('C:/OpenDSS/DSSE33DetailedMultiPhase/' + caseName, squeeze_me=True)
pSec = np.array(mat['P'])  # active load demands at the secondary nodes
qSec = np.array(mat['Q'])  # reactive load demands at the secondary nodes
pPmean = np.array(mat['PpMean']) / sBase * 1000  # pseudo active power at the primary nodes
qPmean = np.array(mat['QpMean']) / sBase * 1000  # pseudo reactive power at the primary nodes
pPstd = np.array(mat['PpStd'])  # standard deviation of the pseudo active power at primary nodes
qPstd = np.array(mat['QpStd'])  # standard deviation of the pseudo reactive power at primary nodes

# setting up opendss
dss_path = 'C:/OpenDSS/DSSE33DetailedMultiPhase/master33Full.dss'
print("openDss path: ", dss_path)
dssObj = DSSStartup.dssstartup(dss_path)

# parametrs of the main loop
maxIter = math.floor(len(pSec[0]) / 24)  # maximum loop iteration
v_wls = np.zeros(priNodes)
mse_wls = np.zeros(maxIter)

# main loop
for k in range(0, maxIter):
    print('time step: ' + str(k + 1))

    # running power flow
    dssCircuit = RunPF.run_pf(dssObj, pSec[:, k], qSec[:, k], priNodes)
    v_true = np.array(RunPF.get_bus_voltages(dssCircuit, priNodes)) / vBasePri  # per unit voltage phasors
    line_name, i_true = RunPF.get_line_currents(dssCircuit, priNodes)
    i_true = np.array(i_true) / Ibase  # per unit current phasors

    # adding noise to the measuremnets
    v_noisy = v_true + np.absolute(v_true) * np.random.normal(0, sigmaPMU, len(v_true)) * \
              np.exp(1j * np.random.normal(0, 1, len(v_true)))
    i_noisy = i_true + np.absolute(i_true) * np.random.normal(0, sigmaPMU, (len(i_true), 3)) * \
              np.exp(1j * np.random.normal(0, 1, (len(i_true), 3)))

    # getting the psuedo powers corresponding to the current time
    load_nodes = np.array(range(4, 3 * priNodes + 1))
    pseudo_power = np.array([pPmean[:, math.floor(k * resol / dur)], qPmean[:, math.floor(k * resol / dur)]])

    # creating covariance matrix of the error for WLS
    if currentPhasor:
        R = np.diag(np.concatenate(
            [np.square(pPstd[:, math.floor(k * resol / dur)]), np.square(qPstd[:, math.floor(k * resol / dur)]),
             sigmaPMU ** 2 * np.ones(3 * 2 * len(PMUnodes) - 1),
             sigmaPMU ** 2 * np.ones(3 * 2 * len(PMUnodes))]))
    else:
        R = np.diag(np.concatenate(
            [np.square(pPstd[:, math.floor(k * resol / dur)]), np.square(qPstd[:, math.floor(k * resol / dur)]),
             sigmaPMU ** 2 * np.ones(3 * 2 * len(PMUnodes) - 1)]))

    # organizing measurement data for WLS
    z, ztype = WLS.get_measurements(v_noisy, i_noisy, line_name, pseudo_power, load_nodes, PMUnodes, currentPhasor)

    # implementing WLS for SE
    iter_max = 10  # maximum number of iteration for WLS to converge
    threshold = 1e-7
    stop_counter = 0
    iter_number = 0
    while stop_counter < 5:
        v_wls, iter_number = WLS.state_estimation(Ybus, z, ztype, R, iter_max, threshold, v_true)
        if iter_number > 1 & iter_number < iter_max:
            stop_counter = 5
        else:
            stop_counter += 1
    # print("Iterations: ", iter_number)
    mse_wls[k] = np.sum(np.square(np.abs(v_true - v_wls))) / (3 * priNodes)
    print(mse_wls[k])

print(np.sqrt(np.mean(mse_wls)))
plt.plot(mse_wls)
plt.show()
