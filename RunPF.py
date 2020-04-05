"""
Created on Fri Nov 15 11:15:35 2019

@author: Moosa Moghimi
"""
import numpy as np
# import math


def run_pf(dss_obj, p, q, n_nodes):
    dss_circuit = dss_obj.ActiveCircuit
    dss_text = dss_obj.Text

    for ind in range(2, n_nodes + 1, 1):
        for secInd in range(56, 110+1):
            node_number = str(ind*1000+secInd)
            load_idx = (ind - 2) * 55 + secInd - 55
            loadchangecommand = 'Edit Load.' + str(node_number) + ' kW= ' + \
                                str(p[load_idx-1]) + ' kvar= ' + str(q[load_idx-1])
            dss_text.Command = loadchangecommand

    dss_text.Command = 'Set mode=snapshot'
    dss_text.Command = 'solve'
#    dss_text.Command = 'show voltage'
#    dss_text.Command = 'show power'
    return dss_circuit


def get_bus_voltages(dss_circuit, n_nodes):
    v = dss_circuit.YnodeVarray[0: 6*n_nodes]
    v_real = np.array(v[0::2])
    v_imag = np.array(v[1::2])
    return v_real + 1j * v_imag


def get_load_powers(dss_circuit, pri_nodes):
    load_power = np.zeros([3 * (pri_nodes - 1), 3])
    load_index = dss_circuit.Loads.First
    load_count = -1
    while load_index != 0:
        load_count += 1
        e_powers = dss_circuit.ActiveCktElement.Powers
        load_power[load_count, 0] = float(dss_circuit.ActiveCktElement.BusNames[0])
        load_power[load_count, 1:3] = 1e3 * np.array(e_powers[0:2])
        load_index = dss_circuit.Loads.Next
    return load_power


def get_line_currents(dss_circuit, pri_nodes):
    line_name = []
    line_current = []
    line_index = dss_circuit.Lines.First
    while line_index < pri_nodes:
        if line_index == 1:
            line_name.append([int(dss_circuit.ActiveCktElement.BusNames[0]),
                              int(dss_circuit.ActiveCktElement.BusNames[1])])
            e_currents = dss_circuit.ActiveCktElement.Currents[0:6]
            line_current.append([complex(e_currents[0], e_currents[1]),
                                 complex(e_currents[2], e_currents[3]),
                                 complex(e_currents[4], e_currents[5])])

        line_name.append([int(dss_circuit.ActiveCktElement.BusNames[1]),
                          int(dss_circuit.ActiveCktElement.BusNames[0])])
        e_currents = dss_circuit.ActiveCktElement.Currents[6:12]
        line_current.append([complex(e_currents[0], e_currents[1]),
                             complex(e_currents[2], e_currents[3]),
                             complex(e_currents[4], e_currents[5])])
        line_index = dss_circuit.Lines.Next
    return np.array(line_name), line_current
