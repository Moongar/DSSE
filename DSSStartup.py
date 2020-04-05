"""
Created on Thu Nov 14 18:34:45 2019

@author: Moosa Moghimi
"""

import win32com.client


def dssstartup(path):
    # 2. Define the com engine.
    engine = win32com.client.Dispatch("OpenDSSEngine.DSS")
    # 3. Start the engine.
    engine.Start("0")
    # 4. Command can be given via text as shown below
    engine.Text.Command = 'clear'
    # 5. Prepare the circuit
    # circuit = engine.ActiveCircuit
    # 6. Then we can compile a DSS file
    engine.Text.Command = 'compile ' + path
    return engine
