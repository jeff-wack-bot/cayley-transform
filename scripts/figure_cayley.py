"""Figure 1 of the Cayley note: the map, its half-angle, and the warping it causes.

(a) upper half-plane with a coordinate grid; (b) its image in the disk under w=(z-i)/(z+i);
(c) the inscribed-angle construction behind e^{i theta} = (1+i tan(theta/2))/(1-i tan(theta/2));
(d) the warping 2 arctan(x/2) versus x (Tustin prewarping, Crank-Nicolson phase error).

Run:  ~/miniforge3/envs/wield/bin/python figure_cayley.py   (writes ../cayley.pdf figure file)
Palette: categorical slots 1-2 of the dataviz reference palette, validated (CVD dE 24.7).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#222222", "#8a8a85"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6})

cay = lambda z: (z - 1j) / (z + 1j)
fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.4))
(a, b), (c, d) = axs

# ---------------------------------------------------------------- (a) half-plane grid
xs = np.linspace(-3, 3, 7)
ys = [0.25, 0.5, 1, 2, 4]
tt = np.linspace(-60, 60, 4000)
for x0 in xs:
    a.plot([x0, x0], [0, 3.2], color=BLUE, lw=0.8)
for y0 in ys:
    a.plot([-3.2, 3.2], [y0, y0], color=ORANGE, lw=0.8)
a.axhline(0, color=INK, lw=1.6)
a.plot(0, 1, "o", color=INK, ms=4)
a.annotate("$z=i$", (0, 1), (0.25, 1.25), color=INK)
a.text(1.5, 0.08, r"$\mathbb{R}$: boundary", color=INK, fontsize=8)
a.set_xlim(-3.2, 3.2); a.set_ylim(-0.3, 3.2); a.set_aspect("equal")
a.set_title(r"(a) upper half-plane $\mathbb{H}$", fontsize=9)
a.text(-3.1, 2.85, r"Re $z$ = const", color=BLUE, fontsize=8)
a.text(-3.1, 2.2, r"Im $z$ = const", color=ORANGE, fontsize=8)
a.set_xticks([]); a.set_yticks([])

# ---------------------------------------------------------------- (b) disk image
for x0 in xs:
    w = cay(x0 + 1j * np.concatenate([np.linspace(1e-4, 1, 400), np.logspace(0, 4, 800)]))
    b.plot(w.real, w.imag, color=BLUE, lw=0.8)
for y0 in ys:
    w = cay(tt + 1j * y0)
    b.plot(w.real, w.imag, color=ORANGE, lw=0.8)
th = np.linspace(0, 2 * np.pi, 400)
b.plot(np.cos(th), np.sin(th), color=INK, lw=1.6)
b.plot(0, 0, "o", color=INK, ms=4)
b.annotate("$w=0$", (0, 0), (0.08, 0.1), color=INK)
b.plot(1, 0, "o", color=INK, ms=3)
b.annotate(r"$z=\infty$", (1, 0), (0.62, -1.1), color=INK, fontsize=8,
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
b.set_xlim(-1.15, 1.15); b.set_ylim(-1.15, 1.15); b.set_aspect("equal")
b.set_title(r"(b) disk, $w=(z-i)/(z+i)$", fontsize=9)
b.set_xticks([]); b.set_yticks([])

# ---------------------------------------------------------------- (c) inscribed angle
theta = 1.9
c.plot(np.cos(th), np.sin(th), color=INK, lw=1.2)
c.axvline(0, color=MUTED, lw=0.6, ls="--")
c.axhline(0, color=MUTED, lw=0.6)
P = np.array([np.cos(theta), np.sin(theta)])
c.plot([0, P[0]], [0, P[1]], color=BLUE, lw=1.2)
c.plot([-1, P[0]], [0, P[1]], color=ORANGE, lw=1.2)
t2 = np.tan(theta / 2)
# extend chord to meet the imaginary axis at i tan(theta/2)
c.plot([-1, 0], [0, t2], color=ORANGE, lw=1.2, ls=":")
c.plot(0, t2, "o", color=ORANGE, ms=4)
c.plot(*P, "o", color=BLUE, ms=4)
c.plot(-1, 0, "o", color=INK, ms=3)
arc = np.linspace(0, theta, 60)
c.plot(0.28 * np.cos(arc), 0.28 * np.sin(arc), color=BLUE, lw=0.8)
c.text(0.18, 0.32, r"$\theta$", color=INK)
arc2 = np.linspace(0, theta / 2, 60)
c.plot(-1 + 0.45 * np.cos(arc2), 0.45 * np.sin(arc2), color=ORANGE, lw=0.8)
c.text(-0.5, 0.12, r"$\theta/2$", color=INK)
c.annotate(r"$e^{i\theta}$", P, P + np.array([-0.35, 0.05]), color=INK)
c.annotate(r"$i\tan\frac{\theta}{2}$", (0, t2), (0.1, t2 + 0.05), color=INK)
c.text(-1.25, -0.25, "$-1$", color=INK)
c.set_xlim(-1.4, 1.4); c.set_ylim(-1.15, 2.1); c.set_aspect("equal")
c.set_title("(c) stereographic projection from $-1$", fontsize=9)
c.set_xticks([]); c.set_yticks([])

# ---------------------------------------------------------------- (d) warping
x = np.linspace(0, 12, 400)
d.plot(x, x, color=MUTED, lw=1, ls="--")
d.plot(x, 2 * np.arctan(x / 2), color=BLUE, lw=2)
d.axhline(np.pi, color=MUTED, lw=0.6)
d.text(7.0, 7.6, r"exact: $\Omega T$", color=INK, fontsize=8, rotation=0)
d.text(5.3, 1.75, r"Cayley: $2\arctan(\Omega T/2)$", color=INK, fontsize=8)
d.text(0.3, np.pi + 0.15, r"$\pi$ (Nyquist)", color=INK, fontsize=8)
d.set_xlim(0, 12); d.set_ylim(0, 9)
d.set_xlabel(r"$x=\Omega T$  or  $E\,\Delta t/\hbar$")
d.set_ylabel("phase per step")
d.set_title("(d) the half-angle as frequency warping", fontsize=9)
for sp in ("top", "right"):
    d.spines[sp].set_visible(False)

fig.tight_layout()
fig.savefig("../cayley-fig.pdf")
fig.savefig("../cayley-fig.png", dpi=170)
print("wrote ../cayley-fig.pdf")
