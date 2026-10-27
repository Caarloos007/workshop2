import numpy as np
import matplotlib.pyplot as plt

# Problem definition
def f(t, y):
    return (t / 2) * (2 - y)

def exact(t):
    # Analytical solution: y = 2 - exp(-t^2/4)
    return 2 - np.exp(-t**2 / 4)

t0, tf, y0 = 0.0, 4.0, 1.0

# Trapezoidal method
def trapezoidal(h):
    n = int(round((tf - t0) / h))
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        # Implicit step solved exactly (f is linear in y):
        # y_{i+1} = y_i + h/2 * (f_i + t_{i+1} - t_{i+1}*y_{i+1}/2)
        fi = f(t[i], y[i])
        y[i + 1] = (y[i] + h / 2 * (fi + t[i + 1])) / (1 + h * t[i + 1] / 4)
    return t, y

# Run for each step size
steps = [0.1, 0.01, 0.001]
results = {h: trapezoidal(h) for h in steps}

# Error table
print("Maximum absolute error vs analytical solution:")
prev = None
for h, (t, y) in results.items():
    err = np.max(np.abs(y - exact(t)))
    ratio = f"  (ratio to previous: {prev / err:.1f})" if prev else ""
    print(f"  h = {h:<6} -> max error = {err:.3e}{ratio}")
    prev = err

# Plot
t_fine = np.linspace(t0, tf, 1000)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Solutions
ax = axes[0]
ax.plot(t_fine, exact(t_fine), 'k-', lw=2, label='Analytical')
styles = {0.1: 'r--', 0.01: 'g-.', 0.001: 'b:'}
for h, (t, y) in results.items():
    ax.plot(t, y, styles[h], lw=1.8, label=f'Trapezoidal h = {h}')
ax.set_xlabel('t')
ax.set_ylabel('y(t)')
ax.set_title('Trapezoidal method vs analytical solution')
ax.grid(True)
ax.legend()

# Absolute errors (log scale)
ax = axes[1]
for h, (t, y) in results.items():
    ax.semilogy(t, np.abs(y - exact(t)) + 1e-18, styles[h], lw=1.8, label=f'h = {h}')
ax.set_xlabel('t')
ax.set_ylabel('|error|')
ax.set_title('Absolute error')
ax.grid(True, which='both')
ax.legend()

plt.tight_layout()
plt.show()