import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Hodgkin-Huxley model parameters
C_m = 1.0  # membrane capacitance (uF/cm^2)
g_Na = 120.0  # maximum conductance of sodium channels (mS/cm^2)
g_K = 36.0  # maximum conductance of potassium channels (mS/cm^2)
g_L = 0.3  # maximum leak conductance (mS/cm^2)
E_Na = 50.0  # sodium reversal potential (mV)
E_K = -77.0  # potassium reversal potential (mV)
E_L = -54.387  # leak reversal potential (mV)

def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80.0)

def hh_model(y, t):
    V, m, h, n = y
    
    # Membrane current components
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # External current, for example, a current injection
    I_ext = 10.0 if 5 <= t <= 45 else 0.0  # current injection from 5 ms to 45 ms
    
    # Membrane potential differential equation
    dVdt = (I_ext - I_Na - I_K - I_L) / C_m
    
    # Gates differential equations
    dm_dt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh_dt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dn_dt = alpha_n(V) * (1 - n) - beta_n(V) * n
    
    return [dVdt, dm_dt, dh_dt, dn_dt]

# Initial conditions
V0 = -65.0  # initial membrane potential (mV)
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))  # initial m gate value
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))  # initial h gate value
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))  # initial n gate value
y0 = [V0, m0, h0, n0]

# Time vector
t = np.linspace(0, 50, 1000)  # 50 ms simulation

# Solve differential equations
sol = odeint(hh_model, y0, t)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], label='V(t)')
plt.title('Hodgkin-Huxley Model Simulation')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.grid(True)
plt.show()
