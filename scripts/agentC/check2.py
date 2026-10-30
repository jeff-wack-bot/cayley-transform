import numpy as np
rng=np.random.default_rng(2); n=3
B=rng.normal(size=(n,1)); K=rng.normal(size=(n,n)); K=K-K.T
A=K-0.5*B@B.T; D=np.array([[1.0]]); C=-D@B.T   # CT all-pass with X=Y=I
print("stable",np.linalg.eigvals(A).real.max()<0)
G=lambda s:(C@np.linalg.solve(s*np.eye(n)-A,B)+D)[0,0]
print("|G(jw)|",abs(G(0.7j)),abs(G(3j)))
I=np.eye(n);M=np.linalg.inv(I-A);Ad=(I+A)@M;Bd=np.sqrt(2)*M@B;Cd=np.sqrt(2)*C@M;Dd=D+C@M@B
R=np.block([[Ad,Bd],[Cd,Dd]]); print("R orth:",np.allclose(R.T@R,np.eye(n+1)))
