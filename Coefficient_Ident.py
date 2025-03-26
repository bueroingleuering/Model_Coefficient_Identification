"""@author: robin"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""Load measurment"""
measure = pd.read_csv('rollout_1850.csv', delimiter=';')
v_meas = measure.v.values  # [km/h]
t_meas = measure.t.values  # [s]
"""Parameters"""
m = 1850  # Vehicle mass
k = 50  # Slope calc value
e = np.inf  # Error value init
"""Variable Init"""
coeff = np.zeros(3)
index_f = len(measure)
v_appr = np.polyval(np.polyfit(t_meas, v_meas / 3.6, 7), t_meas)  # [km/h --> m/s]
v_part = np.zeros([len(coeff), index_f])
t_part = np.zeros([len(coeff), index_f])
v = np.zeros(len(coeff))
t = np.zeros(len(coeff))
dvdt = np.zeros(len(coeff))
index = np.int64(np.zeros(len(coeff)))
"""Parameter Identification"""
while (e > len(v_meas) * 0.01):
    for i in range(0, len(coeff)):
        index[i] = np.random.randint(k, index_f - k)
        v[i] = v_appr[index[i]]
        t[i] = t_meas[index[i]]
        dvdt[i] = (v_appr[index[i] + k] - v_appr[index[i] - k]) / (t_meas[index[i] + k] - t_meas[index[i] - k])
    if (index[0] == index[1] or index[0] == index[2] or index[1] == index[2]): continue
    T = np.matrix([[v[0] ** 2, v[0], 1],
                   [v[1] ** 2, v[1], 1],
                   [v[2] ** 2, v[2], 1]])
    z = np.matrix(-m * dvdt.reshape(3, 1))  # -m * dv/dt
    x = np.matmul(T.I, z)
    A = x[2].item()  # [N]
    B = x[1].item()  # [N/m/s]
    C = x[0].item()  # [N/(m/s)^2]
    if (4 * A * C - B ** 2) > 0:
        D = np.sqrt(4 * A * C - B ** 2)
        v_temp = (D * np.tan((-t_meas * D / (2 * m)) + np.arctan((2 * C * (v_appr[0]) + B) / D)) - B) / (2 * C)
        e_temp = np.sum(np.power(v_appr - v_temp, 2))
        if e_temp < e:
            e = e_temp
            v_calc = v_temp
            coeff = np.array([A, B, C])

v_kph = 3.6 * v_calc
"""Plot"""
plt.figure(1);
plt.rcParams['figure.figsize'] = [16, 12]
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Velocity [kph]', fontsize=16)
plt.plot(t_meas, v_meas, c='#284b64', label='v_meas', linewidth=3.0)
plt.plot(t_meas, v_kph, label='v_calc', linewidth=3.0)
plt.title('Roll out', fontsize=20)
plt.legend(loc='upper right', fontsize=16)
plt.grid(True)
plt.show()
plt.savefig("rollout_new.png")