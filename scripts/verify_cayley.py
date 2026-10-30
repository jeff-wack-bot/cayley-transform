"""Numerical checks for the Cayley-transform note.

Each block checks one claim and prints the worst-case error. Run with the `wield` env:
    ~/miniforge3/envs/wield/bin/python verify_cayley.py > verify_cayley.out

Conventions:
  * Cayley matrix  C = (1/sqrt2) [[1, i], [1, -i]]  (quadrature (x,p) -> ladder (a, a^dagger)).
  * Scalar Cayley map  c(x) = (1 + i x)/(1 - i x) = exp(2 i arctan x)  sends R -> unit circle.
"""
import numpy as np
from numpy.linalg import inv, det, eigvals, norm, solve
from scipy.linalg import expm, solve_continuous_lyapunov, solve_discrete_lyapunov, schur

rng = np.random.default_rng(1)
SQ2 = np.sqrt(2.0)
C = np.array([[1, 1j], [1, -1j]]) / SQ2
results = []


def report(name, err, tol=1e-9):
    ok = err < tol
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max err = {err:.2e}")


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def rand_sl2r():
    M = rng.normal(size=(2, 2))
    if det(M) < 0:
        M[:, 0] *= -1
    return M / np.sqrt(det(M))


def mobius(M, z):
    return (M[0, 0] * z + M[0, 1]) / (M[1, 0] * z + M[1, 1])


cay = lambda x: (1 + 1j * x) / (1 - 1j * x)

# ---------------------------------------------------------------- 1. scalar map and half angle
x = rng.normal(size=1000) * 5
report("1a  (1+ix)/(1-ix) = exp(2i arctan x)", np.max(abs(cay(x) - np.exp(2j * np.arctan(x)))))
report("1b  |c(x)| = 1 on the real line", np.max(abs(abs(cay(x)) - 1)))
zc = rng.normal(size=1000) + 1j * rng.normal(size=1000)
report("1c  Im x > 0  <=>  |c(x)| < 1  (c maps UHP onto the unit disk)",
       float(np.any((zc.imag > 0) != (abs(cay(zc)) < 1))))

# ---------------------------------------------------------------- 2. double-angle phasor chi
err = 0
for _ in range(200):
    z = rng.normal(size=2)
    a = rng.uniform(-np.pi, np.pi)
    chi = lambda v: (v[0] + 1j * v[1]) / (v[0] - 1j * v[1])
    err = max(err, abs(chi(rot(a) @ z) - np.exp(2j * a) * chi(z)), abs(chi(-z) - chi(z)))
    t = z[0] / z[1]
    err = max(err, abs(chi(z) - (t + 1j) / (t - 1j)))
report("2   chi(R(a)z)=e^{2ia}chi(z), chi(-z)=chi(z), chi=(t+i)/(t-i) with t=z1/z2", err)

# ---------------------------------------------------------------- 3. SL(2,R) -> SU(1,1) by C
err = 0
for _ in range(200):
    M = rand_sl2r()
    N = C @ M @ inv(C)
    a, b = N[0, 0], N[0, 1]
    err = max(err, abs(N[1, 0] - np.conj(b)), abs(N[1, 1] - np.conj(a)), abs(abs(a)**2 - abs(b)**2 - 1))
    # rotations become diagonal: C R(al) C^-1 = diag(e^{-i al}, e^{i al})
al = 0.37
err = max(err, norm(C @ rot(al) @ inv(C) - np.diag([np.exp(1j * al), np.exp(-1j * al)])))
report("3a  C SL(2,R) C^-1 = SU(1,1) = {[[a,b],[b*,a*]]}; rotations diagonal", err)
# Mobius picture: C maps lower half plane (physical q, Im q<0) into the unit disk; rotation -> e^{-2i al}
q = rng.normal(size=500) - 1j * rng.uniform(0.01, 5, size=500)
w = mobius(C, q)
report("3b  C maps Im q<0 into unit disk", float(np.any(abs(w) >= 1)))
qr = mobius(rot(al), q)
report("3c  rotation by al acts on disk coordinate w=(q+i)/(q-i) as w -> e^{2i al} w",
       np.max(abs(mobius(C, qr) - np.exp(2j * al) * w)))

# ---------------------------------------------------------------- 4. squeezed vacuum: Fock space -> Gaussian
Nf = 160
a_op = np.diag(np.sqrt(np.arange(1, Nf)), 1)
ad = a_op.T.conj()
r, th = 0.6, 0.9
xi = r * np.exp(1j * th)
S = expm(0.5 * (np.conj(xi) * a_op @ a_op - xi * ad @ ad))
vac = np.zeros(Nf); vac[0] = 1
psi_n = S @ vac
xs = np.linspace(-2.5, 2.5, 41)
herm = np.zeros((Nf, xs.size))            # normalized Hermite functions by stable recursion
herm[0] = np.exp(-xs**2 / 2) / np.pi**0.25
herm[1] = SQ2 * xs * herm[0]
for k in range(2, Nf):
    herm[k] = np.sqrt(2.0 / k) * xs * herm[k - 1] - np.sqrt((k - 1) / k) * herm[k - 2]
psi_x = psi_n @ herm
# fit log psi = const - kappa x^2/2
A = np.vstack([np.ones_like(xs), -xs**2 / 2]).T
coef = np.linalg.lstsq(A, np.log(abs(psi_x)) + 1j * np.unwrap(np.angle(psi_x)), rcond=None)[0]
kappa = coef[1]
cpar = np.exp(1j * th) * np.tanh(r)  # disk coordinate (Perelomov)
report("4a  squeezed vacuum = exp(-kappa x^2/2), kappa=(1+c)/(1-c), c=e^{i th}tanh r (Cayley)",
       abs(kappa - (1 + cpar) / (1 - cpar)), tol=1e-6)
# phase-space rotation exp(-i phi n) multiplies c by e^{-2 i phi}
phi = 0.41
Rphi = np.diag(np.exp(-1j * phi * np.arange(Nf)))
psi2 = (Rphi @ psi_n) @ herm
k2 = np.linalg.lstsq(A, np.log(abs(psi2)) + 1j * np.unwrap(np.angle(psi2)), rcond=None)[0][1]
c2 = (k2 - 1) / (k2 + 1)
report("4b  rotation exp(-i phi N): c -> e^{-2i phi} c (double angle)",
       abs(c2 - np.exp(-2j * phi) * cpar), tol=1e-6)

# ---------------------------------------------------------------- 5. Tustin / bilinear transform
n, T = 4, 0.3
A = rng.normal(size=(n, n)); A -= (max(eigvals(A).real) + 0.5) * np.eye(n)
B = rng.normal(size=(n, 2)); Cm = rng.normal(size=(2, n)); D = np.zeros((2, 2))
Ei = inv(np.eye(n) - A * T / 2)
Ad = (np.eye(n) + A * T / 2) @ Ei
Bd = np.sqrt(T) * Ei @ B
Cd = np.sqrt(T) * Cm @ Ei
Dd = D + Cm @ Ei @ B * T / 2
lam = eigvals(A)
report("5a  eig(Ad) = (1+lam T/2)/(1-lam T/2); stable -> |.|<1",
       np.max(abs(np.sort_complex(eigvals(Ad)) - np.sort_complex((1 + lam * T / 2) / (1 - lam * T / 2)))))
G = lambda s: Cm @ solve(s * np.eye(n) - A, B) + D
Gd = lambda z: Cd @ solve(z * np.eye(n) - Ad, Bd) + Dd
err = 0
for Om in np.linspace(-20, 20, 81):
    zz = (1 + 1j * Om * T / 2) / (1 - 1j * Om * T / 2)
    err = max(err, norm(Gd(zz) - G(1j * Om)), abs(np.angle(zz) - 2 * np.arctan(Om * T / 2)))
report("5b  Gd(z)=G(s) at z=(1+sT/2)/(1-sT/2); jw-axis phase = 2 arctan(OmT/2)", err)
P = solve_continuous_lyapunov(A, -B @ B.T)
Pd = solve_discrete_lyapunov(Ad, Bd @ Bd.T)
Q = solve_continuous_lyapunov(A.T, -Cm.T @ Cm)
Qd = solve_discrete_lyapunov(Ad.T, Cd.T @ Cd)
hsv = np.sort(np.sqrt(abs(eigvals(P @ Q))))
hsvd = np.sort(np.sqrt(abs(eigvals(Pd @ Qd))))
report("5c  CT Lyapunov = DT Stein gramians; Hankel singular values equal",
       max(norm(P - Pd), norm(Q - Qd), norm(hsv - hsvd)))
# Hamiltonian -> symplectic; stable invariant subspace preserved => CARE X = DARE X
J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
Hm = np.block([[A, -B @ B.T], [-Cm.T @ Cm, -A.T]])
Sm = (np.eye(2 * n) + Hm * T / 2) @ inv(np.eye(2 * n) - Hm * T / 2)
report("5d  Hamiltonian H (JH symmetric) -> Cayley(H) symplectic", norm(Sm.T @ J @ Sm - J))
Tc, Zc, sdim = schur(Hm, output="complex", sort="lhp")
X1 = Zc[n:, :n] @ inv(Zc[:n, :n])
Td, Zd, sdimd = schur(Sm, output="complex", sort="iuc")
X2 = Zd[n:, :n] @ inv(Zd[:n, :n])
report("5e  stabilizing Riccati solution identical from H (lhp) and Cayley(H) (unit disk)",
       norm(X1 - X2) + norm(A.T @ X1 + X1 @ A - X1 @ B @ B.T @ X1 + Cm.T @ Cm))

# ---------------------------------------------------------------- 6. Crank-Nicolson = Cayley
Nh, dt = 30, 0.2
Hh = rng.normal(size=(Nh, Nh)) + 1j * rng.normal(size=(Nh, Nh)); Hh = (Hh + Hh.conj().T) / 2
Ucn = solve(np.eye(Nh) + 0.5j * dt * Hh, np.eye(Nh) - 0.5j * dt * Hh)
E = np.linalg.eigvalsh(Hh)
ph = np.sort(np.angle(eigvals(Ucn)))
report("6   Crank-Nicolson unitary; eigenphase = -2 arctan(E dt/2) (Tustin warping)",
       max(norm(Ucn.conj().T @ Ucn - np.eye(Nh)), norm(ph - np.sort(-2 * np.arctan(E * dt / 2)))))

# ---------------------------------------------------------------- 7. K-matrix -> S-matrix, Blaschke product
Nr = 6
H0 = rng.normal(size=(Nr, Nr)); H0 = (H0 + H0.T) / 2
W = rng.normal(size=(Nr, 1)) * 0.4
zres = eigvals(H0 - 1j * np.pi * W @ W.T)   # resonance poles, Im < 0
err = 0
Es = np.linspace(-5, 5, 401)
Sv = []
for En in Es:
    K = (np.pi * W.T @ solve(En * np.eye(Nr) - H0, W))[0, 0]
    Sk = (1 - 1j * K) / (1 + 1j * K)
    Sb = np.prod((En - np.conj(zres)) / (En - zres))
    err = max(err, abs(Sk - Sb), abs(abs(Sk) - 1))
    Sv.append(Sk)
report("7a  S=(1-iK)/(1+iK), K=pi W^T(E-H)^-1 W, equals Blaschke prod (E-z*)/(E-z) over H-i pi WW^T", err)
Ebig = np.linspace(-200, 200, 400001)
Sbig = np.array([np.prod((e - np.conj(zres)) / (e - zres)) for e in Ebig[::10]])
wind = np.sum(np.diff(np.unwrap(np.angle(Sbig)))) / (2 * np.pi)
report("7b  phase of S winds -(number of resonances) times (each resonance: delta += pi)",
       abs(abs(wind) - Nr), tol=0.02)

# ---------------------------------------------------------------- 8. Smith chart / impedance half-plane
def d_rhp(z1, z2):  # hyperbolic distance in right half plane, metric |dz|^2/(Re z)^2
    return np.arccosh(1 + abs(z1 - z2)**2 / (2 * z1.real * z2.real))

zl = rng.uniform(0.05, 5, 500) + 1j * rng.normal(size=500) * 3
G_ = (zl - 1) / (zl + 1)
vswr = (1 + abs(G_)) / (1 - abs(G_))
report("8a  Gamma=(z-1)/(z+1): |Gamma|=tanh(d/2), VSWR = exp(d(z,1))",
       max(np.max(abs(abs(G_) - np.tanh(d_rhp(zl, 1 + 0j) / 2))), np.max(abs(np.log(vswr) - d_rhp(zl, 1 + 0j)))))
bl = 0.7
zin = (zl + 1j * np.tan(bl)) / (1 + 1j * zl * np.tan(bl))
report("8b  lossless line: Gamma_in = Gamma_L e^{-2i beta l} (double angle)",
       np.max(abs((zin - 1) / (zin + 1) - G_ * np.exp(-2j * bl))))
zs = rng.uniform(0.05, 5, 500) + 1j * rng.normal(size=500) * 3
eta = 4 * zs.real * zl.real / abs(zs + zl)**2
report("8c  power transfer 4RsRL/|Zs+ZL|^2 = sech^2(d(ZL, Zs*)/2)",
       np.max(abs(eta - 1 / np.cosh(d_rhp(zl, np.conj(zs)) / 2)**2)))

# ---------------------------------------------------------------- 9. polarization
err = 0
for _ in range(200):
    E2 = rng.normal(size=2) + 1j * rng.normal(size=2)
    E2 /= norm(E2)
    ER = (E2[0] - 1j * E2[1]) / SQ2
    EL = (E2[0] + 1j * E2[1]) / SQ2
    w_ = EL / ER                                  # circular-basis ratio = Cayley of the Jones ratio
    s = np.array([abs(E2[0])**2 - abs(E2[1])**2, 2 * (np.conj(E2[0]) * E2[1]).real,
                  2 * (np.conj(E2[0]) * E2[1]).imag])
    st = np.array([2 * w_.real, 2 * w_.imag, abs(w_)**2 - 1]) / (abs(w_)**2 + 1)  # inverse stereographic
    err = max(err, norm(st - np.array([s[0], s[1], -s[2]])))
    a = rng.uniform(-3, 3)
    E3 = rot(a) @ E2
    err = max(err, abs((E3[0] + 1j * E3[1]) / (E3[0] - 1j * E3[1]) - np.exp(2j * a) * w_))
report("9   circular ratio E_L/E_R = chi; inverse stereographic -> Stokes; rotator -> e^{2ia}", err)

# ---------------------------------------------------------------- 10. Cayley's rotations, Gibbs vector
err = 0
for _ in range(200):
    nvec = rng.normal(size=3); nvec /= norm(nvec)
    t = rng.uniform(-3, 3)
    g = nvec * np.tan(t / 2)
    Gk = np.array([[0, -g[2], g[1]], [g[2], 0, -g[0]], [-g[1], g[0], 0]])
    Rc = (np.eye(3) + Gk) @ inv(np.eye(3) - Gk)
    Kn = np.array([[0, -nvec[2], nvec[1]], [nvec[2], 0, -nvec[0]], [-nvec[1], nvec[0], 0]])
    err = max(err, norm(Rc - expm(t * Kn)))
report("10  (I+G)(I-G)^-1 with Gibbs vector n tan(t/2) = rotation by t about n", err)

# ---------------------------------------------------------------- 11. KLMTV filter target = Heitler form
gam, Io = 1.0, 1.0
Lam4 = 2 * Io * gam**4
Kk = lambda s: Lam4 / (s**2 + gam**2 * s)
p = np.roots([1, gam**2, -1j * Lam4])     # roots of D - iN
err = 0
for Om in np.linspace(0.05, 5, 200):
    s = Om**2
    target = (1 + 1j * Kk(s)) / (1 - 1j * Kk(s))
    chain = np.prod((s - np.conj(p)) / (s - p))
    err = max(err, abs(target - chain))
report("11  KLMTV filter all-pass (1+iK)/(1-iK) = two Blaschke factors in s=Om^2", err)

# ---------------------------------------------------------------- 12. Jacobian / positivity identity (scalar and matrix)
zz = rng.normal(size=500) + 1j * rng.normal(size=500)
ww = (zz - 1j) / (zz + 1j)
err = np.max(abs((1 - abs(ww)**2) - 4 * zz.imag / abs(zz + 1j)**2))
for _ in range(50):
    Zm = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Sm_ = (Zm - np.eye(3)) @ inv(Zm + np.eye(3))
    Fi = inv(Zm + np.eye(3))
    err = max(err, norm((np.eye(3) - Sm_.conj().T @ Sm_) - 2 * Fi.conj().T @ (Zm + Zm.conj().T) @ Fi))
report("12  1-|w|^2 = 4 Im z/|z+i|^2 ; I - S^H S = 2 (Z+I)^-H (Z+Z^H)(Z+I)^-1", err)

# ---------------------------------------------------------------- 13. wave digital filter: C -> z^-1, L -> -z^-1
err = 0
Tw, Cc, Ll = 0.01, 3e-3, 2e-3
for th in np.linspace(0.1, 3.0, 30):
    zq = np.exp(1j * th)
    s_ = (2 / Tw) * (1 - 1 / zq) / (1 + 1 / zq)          # bilinear
    RC, RL = Tw / (2 * Cc), 2 * Ll / Tw
    ZC, ZL = 1 / (s_ * Cc), s_ * Ll
    err = max(err, abs((ZC - RC) / (ZC + RC) - 1 / zq), abs((ZL - RL) / (ZL + RL) + 1 / zq))
report("13  wave digital filter: reflectance of C -> z^-1, L -> -z^-1 (two Cayley maps cancel)", err)

# ---------------------------------------------------------------- 14. CUE -> Cauchy ensemble (Brouwer)
from scipy.stats import unitary_group
Mdim, Nsamp = 8, 4000
Ev = []
for _ in range(Nsamp):
    U = unitary_group.rvs(Mdim, random_state=rng)
    Hc = 1j * (np.eye(Mdim) - U) @ inv(np.eye(Mdim) + U)   # inverse Cayley: U = (1+iH)/(1-iH)... check hermiticity
    Ev.extend(np.linalg.eigvalsh((Hc + Hc.conj().T) / 2))
Ev = np.array(Ev)
U0 = unitary_group.rvs(Mdim, random_state=rng)
H0c = 1j * (np.eye(Mdim) - U0) @ inv(np.eye(Mdim) + U0)
herm_err = norm(H0c - H0c.conj().T)
# compare empirical CDF of eigenvalues with Cauchy CDF 1/2 + arctan(E)/pi
Es_ = np.sort(Ev)
cdf_emp = np.arange(1, Es_.size + 1) / Es_.size
ks = np.max(abs(cdf_emp - (0.5 + np.arctan(Es_) / np.pi)))
report("14  H = i(1-U)(1+U)^-1 Hermitian for U in CUE; level density = Cauchy 1/(pi(1+E^2)) (KS stat)",
       max(herm_err * 1e-3, ks), tol=0.02)

# ---------------------------------------------------------------- 15. relativity: half rapidity and Doppler factor
xi_ = rng.normal(size=200) * 2
v_, w_ = np.tanh(xi_), np.tanh(xi_ / 2)
err = max(np.max(abs(v_ - 2 * w_ / (1 + w_**2))), np.max(abs((1 + w_) / (1 - w_) / np.exp(xi_) - 1)),
          np.max(abs(np.sqrt((1 + v_) / (1 - v_)) / np.exp(xi_) - 1)))
report("15  v = 2w/(1+w^2) with w = tanh(xi/2); Doppler k = (1+w)/(1-w) = e^xi (relative; sqrt((1+v)/(1-v)) loses digits as v->1)", err, tol=1e-7)

# ---------------------------------------------------------------- 16. Crank-Nicolson phase error series
xx = np.linspace(1e-3, 0.2, 50)
report("16  2 arctan(x/2) - x = -x^3/12 + O(x^5)",
       np.max(abs((2 * np.arctan(xx / 2) - xx) + xx**3 / 12) / xx**5), tol=0.02)

# ---------------------------------------------------------------- 17. Robin wall: reflection is a Cayley factor (Bonneau et al.)
err = 0
for lam_ in [-2.0, -0.3, 0.4, 3.0]:
    for k_ in np.linspace(0.1, 5, 20):
        # phi = A e^{-ikx} + B e^{ikx}, phi(0) = lam phi'(0)  -> solve for B/A
        Ar = 1.0
        Br = Ar * (1 + 1j * lam_ * k_) / (1j * lam_ * k_ - 1)
        err = max(err, abs(Br / Ar + (1 + 1j * lam_ * k_) / (1 - 1j * lam_ * k_)),
                  abs((Ar + Br) - lam_ * 1j * k_ * (Br - Ar)))
report("17  Robin wall phi(0)=lam phi'(0): out/in = -(1+i lam k)/(1-i lam k)", err)

# ---------------------------------------------------------------- 18. SLH single cavity transfer function
err = 0
kap = 1.7
for om in np.linspace(-10, 10, 101):
    sv = 1j * om
    Xi = 1 - np.sqrt(kap) * (1 / (sv + kap / 2)) * np.sqrt(kap)
    err = max(err, abs(Xi - (sv - kap / 2) / (sv + kap / 2)), abs(abs(Xi) - 1))
report("18  SLH one-mode cavity: Xi(s) = 1 - kappa/(s+kappa/2) = (s-kappa/2)/(s+kappa/2), |Xi(i om)|=1", err)

print(f"\n{sum(results)}/{len(results)} checks passed")
