import numpy as np
import matplotlib.pyplot as plt

# Problem definition
def f(t, y):
    return (t / 2) * (2 - y)

def exact(t):
    # Analytical solution: y = 2 - exp(-t^2/4)
    return 2 - np.exp(-t**2 / 4)

t0, tf, y0 = 0.0, 4.0, 1.0

# Fourth-order Runge-Kutta
def rk4(h):
    n = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y

# Trapezoidal method (Exercise 2)
def trapezoidal(h):
    n = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        # Implicit step solved exactly (f is linear in y)
        fi = f(t[i], y[i])
        y[i + 1] = (y[i] + h / 2 * (fi + t[i + 1])) / (1 + h * t[i + 1] / 4)
    return t, y

# Run both methods
steps_rk = [1, 0.1, 0.01]            # Exercise 4
steps_tr = [0.1, 0.01, 0.001]        # Exercise 2
res_rk = {h: rk4(h) for h in steps_rk}
res_tr = {h: trapezoidal(h) for h in steps_tr}

def max_err(t, y):
    return np.max(np.abs(y - exact(t)))

# Error tables
print("RK4 - max absolute error and final value at t = 4:")
prev = None
for h, (t, y) in res_rk.items():
    e = max_err(t, y)
    order = f"  observed order: {np.log10(prev / e):.2f}" if prev else ""
    print(f"  h = {h:<6} -> max error = {e:.3e}  y(4) = {y[-1]:.8f}{order}")
    prev = e

print("\nTrapezoidal - max absolute error and final value at t = 4:")
prev = None
for h, (t, y) in res_tr.items():
    e = max_err(t, y)
    order = f"  observed order: {np.log10(prev / e):.2f}" if prev else ""
    print(f"  h = {h:<6} -> max error = {e:.3e}  y(4) = {y[-1]:.8f}{order}")
    prev = e

print(f"\nExact y(4) = {exact(tf):.8f}")

# Direct comparison at equal step size
print("\nComparison at the same step size (max error):")
print(f"  {'h':<6} {'RK4':>12} {'Trapezoidal':>14} {'Trap/RK4':>10}")
for h in [0.1, 0.01]:
    e_rk = max_err(*res_rk[h])
    e_tr = max_err(*res_tr[h])
    print(f"  {h:<6} {e_rk:>12.3e} {e_tr:>14.3e} {e_tr / e_rk:>10.1f}")

# Comparison at similar accuracy (cost)
print("\nCost comparison (number of f evaluations, RK4 = 4 per step, trapezoidal = 1 per step):")
for h, (t, y) in res_rk.items():
    print(f"  RK4  h = {h:<6} evals = {4 * (len(t) - 1):>6}  max error = {max_err(t, y):.3e}")
for h, (t, y) in res_tr.items():
    print(f"  Trap h = {h:<6} evals = {len(t) - 1:>6}  max error = {max_err(t, y):.3e}")

# Plots
t_fine = np.linspace(t0, tf, 1000)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) RK4 solutions
ax = axes[0]
ax.plot(t_fine, exact(t_fine), 'k-', lw=2, label='Analytical')
st_rk = {1: 'r--o', 0.1: 'g-.', 0.01: 'b:'}
for h, (t, y) in res_rk.items():
    ax.plot(t, y, st_rk[h], lw=1.8, label=f'RK4 h = {h}')
ax.set_xlabel('t'); ax.set_ylabel('y(t)')
ax.set_title('RK4 vs analytical solution')
ax.grid(True); ax.legend()

# 2) Absolute error: RK4 vs trapezoidal at equal h
ax = axes[1]
ax.semilogy(*(lambda t, y: (t, np.abs(y - exact(t)) + 1e-18))(*res_rk[0.1]), 'r-', lw=1.8, label='RK4 h = 0.1')
ax.semilogy(*(lambda t, y: (t, np.abs(y - exact(t)) + 1e-18))(*res_tr[0.1]), 'r--', lw=1.8, label='Trap h = 0.1')
ax.semilogy(*(lambda t, y: (t, np.abs(y - exact(t)) + 1e-18))(*res_rk[0.01]), 'b-', lw=1.8, label='RK4 h = 0.01')
ax.semilogy(*(lambda t, y: (t, np.abs(y - exact(t)) + 1e-18))(*res_tr[0.01]), 'b--', lw=1.8, label='Trap h = 0.01')
ax.set_xlabel('t'); ax.set_ylabel('|error|')
ax.set_title('Absolute error: RK4 vs trapezoidal')
ax.grid(True, which='both'); ax.legend()

# 3) Convergence: max error vs h (log-log)
ax = axes[2]
h_rk = np.array(steps_rk); e_rk = np.array([max_err(*res_rk[h]) for h in steps_rk])
h_tr = np.array(steps_tr); e_tr = np.array([max_err(*res_tr[h]) for h in steps_tr])
ax.loglog(h_rk, e_rk, 'bo-', label='RK4')
ax.loglog(h_tr, e_tr, 'rs-', label='Trapezoidal')
hh = np.array([1e-3, 1])
ax.loglog(hh, e_rk[1] * (hh / 0.1)**4, 'b:', alpha=0.5, label='slope 4')
ax.loglog(hh, e_tr[0] * (hh / 0.1)**2, 'r:', alpha=0.5, label='slope 2')
ax.set_xlabel('h'); ax.set_ylabel('max |error|')
ax.set_title('Convergence order')
ax.grid(True, which='both'); ax.legend()

plt.tight_layout()
plt.show()