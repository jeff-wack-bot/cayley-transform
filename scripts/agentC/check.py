import numpy as np, scipy.linalg as sl
rng=np.random.default_rng(1); n,m,p=5,2,3
A=rng.normal(size=(n,n)); A-= (max(np.linalg.eigvals(A).real)+0.5)*np.eye(n)
B=rng.normal(size=(n,m)); C=rng.normal(size=(p,n)); D=rng.normal(size=(p,m))*0.3
I=np.eye(n); M=np.linalg.inv(I-A)   # alpha=1 (T=2)
Ad=(I+A)@M; Bd=np.sqrt(2)*M@B; Cd=np.sqrt(2)*C@M; Dd=D+C@M@B
P=sl.solve_continuous_lyapunov(A,-B@B.T); Q=sl.solve_continuous_lyapunov(A.T,-C.T@C)
Pd=sl.solve_discrete_lyapunov(Ad,Bd@Bd.T); Qd=sl.solve_discrete_lyapunov(Ad.T,Cd.T@Cd)
print("gramians equal:",np.allclose(P,Pd),np.allclose(Q,Qd))
hc=np.sqrt(np.sort(np.linalg.eigvals(P@Q).real)); print("HSV",hc)
# transfer fn: Gd(z)=G(s) with s=(z-1)/(z+1)
G=lambda s: C@np.linalg.solve(s*I-A,B)+D
Gd=lambda z: Cd@np.linalg.solve(z*I-Ad,Bd)+Dd
z=0.3+0.7j; print("TF match:",np.allclose(Gd(z),G((z-1)/(z+1))))
# warping: z=e^{i th} -> s = i tan(th/2)
th=1.1; print("warp:",np.isclose((np.exp(1j*th)-1)/(np.exp(1j*th)+1),1j*np.tan(th/2)))
# CARE vs DARE: A'X+XA - X B R^-1 B' X + C'C=0  (R=I)
X=sl.solve_continuous_are(A,B,C.T@C,np.eye(m))
# discrete ARE with cross term from bilinear map? test plain DARE on (Ad,Bd,Q?,R?)
# known: with s=(z-1)/(z+1) the CARE is equiv to a DARE with Qd=Cd'Cd,Rd=Dd'Dd... use general spectral factor check
# H-inf norm via grid
w=np.logspace(-3,3,4000); hn=max(np.linalg.svd(G(1j*x),compute_uv=False)[0] for x in w)
th=2*np.arctan(w); hd=max(np.linalg.svd(Gd(np.exp(1j*t)),compute_uv=False)[0] for t in th)
print("Hinf c,d:",hn,hd)
a=3.0; M=np.linalg.inv(a*I-A); Ad=(a*I+A)@M; Bd=np.sqrt(2*a)*M@B; Cd=np.sqrt(2*a)*C@M; Dd=D+C@M@B
Pd=sl.solve_discrete_lyapunov(Ad,Bd@Bd.T); print("alpha=3 gram equal:",np.allclose(P,Pd))
Gd=lambda z: Cd@np.linalg.solve(z*I-Ad,Bd)+Dd
print("TF alpha:",np.allclose(Gd(z),G(a*(z-1)/(z+1))))
