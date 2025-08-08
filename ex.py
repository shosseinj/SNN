import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# پارامترهای نورون‌ها
params = [
    {"A": 0.5, "B": 0.3, "t_min": 3, "t_max": 5, "threshold": 2.0},
    {"A": 0.4, "B": 0.2, "t_min": 7, "t_max": 10, "threshold": 1.5},
    {"A": 0.3, "B": 0.1, "t_min": 12, "t_max": 15, "threshold": 1.0},
    {"A": 0.2, "B": 0.05, "t_min": 17, "t_max": 20, "threshold": 0.5}
]

colors = ['blue', 'green', 'red', 'purple']
t = np.linspace(0, 20, 2000)
dt = t[1] - t[0]
layers = 4

V = [np.zeros_like(t) for _ in range(layers)]
spike_times = [[] for _ in range(layers)]
input_spikes = {0: [1.0, 2.0]}  # اسپایک‌های اولیه برای لایه ۱

# شبیه‌سازی ولتاژ و شلیک نورون‌ها
for i in range(1, len(t)):
    for layer in range(layers):
        p = params[layer]
        prev_V = V[layer][i-1]
        spikes = input_spikes.get(layer, [])
        spike_input = sum(1.0 for ts in spikes if abs(t[i] - ts) < dt)
        if t[i] < p["t_min"]:
            dV = p["A"] + spike_input
        elif p["t_min"] <= t[i] <= p["t_max"]:
            dV = p["B"]
        else:
            dV = 0
        V[layer][i] = prev_V + dV * dt
        if V[layer][i] >= p["threshold"]:
            spike_times[layer].append(t[i])
            V[layer][i] = 0
            if layer + 1 < layers:
                input_spikes.setdefault(layer + 1, []).append(t[i])

# رسم انیمیشن
fig, ax = plt.subplots(figsize=(14, 9))
lines = [ax.plot([], [], label=f'لایه {i+1}', color=colors[i])[0] for i in range(layers)]
vlines = []

ax.set_xlim(0, 20)
ax.set_ylim(0, 2.5)
ax.set_xlabel('زمان (ثانیه)')
ax.set_ylabel('ولتاژ (V)')
ax.set_title('انیمیشن شلیک نورون‌ها')
ax.grid(True)
ax.legend()

# خطوط t_min/t_max و نواحی سایه‌دار
for i in range(layers):
    ax.axvline(x=params[i]['t_min'], linestyle=':', color=colors[i], alpha=0.4)
    ax.axvline(x=params[i]['t_max'], linestyle=':', color=colors[i], alpha=0.4)
    ax.axvspan(0, params[i]['t_min'], facecolor='pink', alpha=0.05)
    ax.axvspan(params[i]['t_min'], 20, facecolor='gray', alpha=0.03)
    ax.axhline(y=params[i]['threshold'], linestyle='--', color=colors[i], alpha=0.3)

# تابع به‌روزرسانی فریم‌ها
def update(frame):
    for i in range(layers):
        lines[i].set_data(t[:frame], V[i][:frame])
    for v in vlines:
        v.remove()
    vlines.clear()
    for i in range(layers):
        for st in spike_times[i]:
            if st <= t[frame]:
                vline = ax.axvline(x=st, color=colors[i], linestyle='-', ymax=0.1, alpha=0.6)
                vlines.append(vline)
    return lines + vlines

ani = animation.FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)

# ذخیره فایل ویدیو
ani.save("neuron_spike_animation.gif", writer='pillow', fps=20)
print("✔️ ویدیو ذخیره شد: neuron_spike_animation.mp4")
