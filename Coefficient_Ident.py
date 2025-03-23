"""@author: robin"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""Load measurment"""
measure = pd.read_csv('rollout_1850.csv', delimiter=';')
vmeas = measure.v.values
tmeas = measure.t.values
"""Parameters"""
m = 1850  # Vehicle mass
k = 50  # Slope calc value
e = np.inf  # Error value init
"""Variable Init"""
coeff = np.zeros(3)
index_f = len(measure)
vappr = np.polyval(np.polyfit(tmeas, vmeas, 7), tmeas)
vpart = np.zeros([len(coeff), index_f])
tpart = np.zeros([len(coeff), index_f])
v = np.zeros(len(coeff))
t = np.zeros(len(coeff))
dvdt = np.zeros(len(coeff))
index = np.int64(np.zeros(len(coeff)))
"""Parameter Identification"""
while (e > len(vmeas) * 0.01):
    for i in range(0, len(coeff)):
        index[i] = np.random.randint(k, index_f - k)
        v[i] = vappr[index[i]]
        t[i] = tmeas[index[i]]
        dvdt[i] = (vappr[index[i] + k] - vappr[index[i] - k]) / (tmeas[index[i] + k] - tmeas[index[i] - k])
    if (index[0] == index[1] or index[0] == index[2] or index[1] == index[2]): continue
    T = np.matrix([[v[0] ** 2, v[0], 1],
                   [v[1] ** 2, v[1], 1],
                   [v[2] ** 2, v[2], 1]])
    z = np.matrix(-m * dvdt.reshape(3, 1) / 3.6)  # -m * dv/dv , km/h -> m/s
    x = np.matmul(T.I, z)
    A = x[2].item()  # [N]
    B = x[1].item() * 3.6  # [N/km/h]
    C = x[0].item() * 3.6 ** 2  # [N/(km/h)^2]
    if (4 * A * C - B ** 2) > 0:
        D = np.sqrt(4 * A * C - B ** 2)
        vtemp = 3.6 * (D * np.tan((-tmeas * D / (2 * m)) + np.arctan((2 * C * (vmeas[0] / 3.6) + B) / D)) - B) / (2 * C)
        etemp = np.sum(np.power(vmeas - vtemp, 2))
        if etemp < e:
            e = etemp
            vcalc = vtemp
            coeff = np.array([A, B, C])
"""Plot"""
plt.figure(1);
plt.rcParams['figure.figsize'] = [16, 12]
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Velocity [kph]', fontsize=16)
plt.plot(tmeas, vmeas, c='#284b64', label='v_meas', linewidth=3.0)
plt.plot(tmeas, vcalc, label='v_calc', linewidth=3.0)
plt.title('Roll out', fontsize=20)
plt.legend(loc='upper right', fontsize=16)
plt.grid(True)
plt.show()
plt.savefig("rollout_new.png")

