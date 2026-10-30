"""B-agent checks (Cayley-transform note, area B). Pure Python.
1 Yonte et al. 2002: w=(z+i)/(1+iz) maps disk->UHP and z = i*K(w), K(w)=(w-i)/(w+i).
3 Orfanidis Ch.6: s=(1-z^-1)/(1+z^-1), z=e^{2j d} -> s = j tan d.
4 Kwee 2014 eq.(7), eps=0: r = -(1-i xi)/(1+i xi) = -exp(-2i atan xi).
5 KLMTV eq.(77): b+ = (b1+i b2)/sqrt2, b-^dag = (b1-i b2)/sqrt2, so chi(b)=b+/b-^dag; chi(R(a)z)=e^{2ia}chi(z).
6 Lossless cavity: e^{i(alpha+ + alpha-)} = (s-conj p)/(s-p) (sum, NOT difference).
7 Scutaru 1998: |<psi1|psi2>|^2 = 1/cosh(d/2) for exp(i tau x^2/2), d = curvature -1 distance in UHP.
"""
import cmath, math, random
random.seed(1)
K=lambda w:(w-1j)/(w+1j)
# 1 Yonte: z -> w=(z+i)/(1+iz) maps disk to UHP, and z = i K(w)
for _ in range(5):
    z=cmath.rect(random.random(),random.uniform(0,6.28)); w=(z+1j)/(1+1j*z)
    assert w.imag>0 and abs(z-1j*K(w))<1e-12
# 3 Orfanidis: s=(1-1/z)/(1+1/z), z=e^{2jd} -> j tan d
d=0.37; z=cmath.exp(2j*d); s=(1-1/z)/(1+1/z); assert abs(s-1j*math.tan(d))<1e-12
# 4 Kwee lossless: r=(-1+i xi)/(1+i xi) = -exp(-2i atan xi)
xi=0.8; r=(-1+1j*xi)/(1+1j*xi); assert abs(r+cmath.exp(-2j*math.atan(xi)))<1e-12
# 5 KLMTV: b+=(b1+i b2)/sqrt2, b-^dag=(b1-i b2)/sqrt2 -> ratio = chi; check b1,b2 formulas eq 77
b1,b2=0.3,-1.1; bp=(b1+1j*b2)/2**.5; bmd=(b1-1j*b2)/2**.5
assert abs((bp+bmd)/2**.5-b1)<1e-12 and abs((bp-bmd)/(2**.5*1j)-b2)<1e-12
# rotation R(a) multiplies chi by e^{2ia}
a=0.4; c=lambda x,y:(x+1j*y)/(x-1j*y); x2=math.cos(a)*b1-math.sin(a)*b2; y2=math.sin(a)*b1+math.cos(a)*b2
assert abs(c(x2,y2)-cmath.exp(2j*a)*c(b1,b2))<1e-12
# 6 alpha± = 2 atan(xi ± W/delta), xi=-Delta/delta; e^{i(a+ + a-)} = (s - conj p)/(s-p), p=(Delta - i delta)^2
D,dl,W=0.7,0.3,1.3; ap=2*math.atan((-D+W)/dl); am=2*math.atan((-D-W)/dl); om=D-1j*dl; p=om**2; s=W*W
print("sum:",abs(cmath.exp(1j*(ap+am))-(s-p.conjugate())/(s-p)), " diff:",abs(cmath.exp(1j*(ap-am))-(s-p.conjugate())/(s-p)))
# 2 Chi-Fraas-Tan: free rotation: z -> e^{-2i w t} z for H=w a*a ; exponent (1+z)/(1-z) in right half plane
z=0.5*cmath.exp(0.3j); print("Re (1+z)/(1-z) >0:", ((1+z)/(1-z)).real>0)
# Scutaru: |<psi1|psi2>|^2 vs 1/cosh(d/2) for Gaussians exp(i tau x^2/2) with tau in UHP
t1,t2=0.3+1.0j,-0.5+2.2j
I=lambda a: cmath.sqrt(2*math.pi/a)  # integral of exp(-a x^2/2)
P=abs(I(-1j*(t2-t1.conjugate())))**2/(I(2*t1.imag).real*I(2*t2.imag).real)
d=math.acosh(1+abs(t1-t2)**2/(2*t1.imag*t2.imag))
print("Scutaru: |<1|2>|^2 =",P," 1/cosh(d/2) =",1/math.cosh(d/2))
print("all asserts ok")
