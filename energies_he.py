import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import root, bisect
from scipy.interpolate import CubicSpline

def PHI(R, x):
    phi = np.zeros_like(R)
    for i in range(len(x)):
        if i == 0:
            phi[i] = -trapezoid(R[i:]**2/x[i:], x[i:])
        elif i > 0 and i < len(x)-1:
            phi[i] = -trapezoid(R[i:]**2/x[i:], x[i:]) - trapezoid(R[:i]**2, x[:i])/x[i]
        else:
            phi[i] = - trapezoid(R[:i]**2, x[:i])/x[i]
    return phi

def R_0(x, Z):
    Z_1 = Z - 5/16
    R = 2*np.sqrt(Z_1) * Z_1 * x * np.exp(-Z_1*x)
    return R

def phi_fun(x, phi, x_unknown):
    phi_fun = CubicSpline(x, phi)
    return phi_fun(x_unknown)

def schrod(t, y, x, phi, e, Z):
    R = y[0]
    dR = y[1]
    ddR = -(2*Z/t + 2*phi_fun(x, phi, t) + e)*R
    return [dR, ddR]

def find_y12(phi, x, e, Z, d_R, h):
    sol = solve_ivp(schrod, [1e-8, h], [0, d_R], args=(x, phi, e, Z), method='DOP853', t_eval=np.array([h]), rtol=1e-12, atol=1e-12)
    print(sol.y)
    return 0, sol.y[0, 0]

def calc_R(x, e, N_iter, Z=2):
    # R_0
    R = R_0(x, Z)
    h = x[1] - x[0]

    for i in range(N_iter):
        # calculate phi
        phi = PHI(R, x)

        # calculate y_1 and y_2
        d_R = (R[1] - R[0])/h
        R[0], R[1] = find_y12(phi, x, e, Z, d_R, h)

        # calculate k
        k = 2*Z/x - 2*phi/x + e*np.ones_like(x)

        # calc new R
        for j in range(2, len(x)):
            R[j] = (2*(1-5/12*h**2*k[j-1])*R[j-1] - (1+1/12*h**2*k[j-2])*R[j-2])/(1+1/12*h**2*k[j])

    return R

# plot R for N_iter from 0 to 5
x = np.linspace(1e-8, 20, 10000)
for i in range(5, 8):
    R = calc_R(x, -0.2, i)
    plt.plot(x, R, label=f'N_iter = {i}')
plt.legend()
plt.show()
