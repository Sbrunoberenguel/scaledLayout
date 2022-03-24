import numpy as np 

#CHANGE OF COORDINATE SYSTEM
varPhi_init = -np.pi
varPhi_end = np.pi
Phi_init = np.pi/2.0
Phi_end = -np.pi/2.0

def xy2angles(x,y,W=1024,H=512):
	varPhi = x*(varPhi_end-varPhi_init)/W + varPhi_init	#np.pi*((2*x)/W - 1.0)
	Phi = 	y*(Phi_end-Phi_init)/H + Phi_init			#np.pi*(0.5 - y/H)
	return varPhi,Phi

def angles2xy(varPhi,Phi,W=1024,H=512):
	x = W*(varPhi-varPhi_init)/(varPhi_end-varPhi_init)	
	y = H*(Phi-Phi_init)/(Phi_end-Phi_init)				
	return x,y

# def angles2xy(varPhi,Phi,W=1024,H=512):
# 	x = (W/2.0)*(varPhi/np.pi + 1.0)
# 	y = H*(0.5 - Phi/np.pi)
# 	return x,y

def xy2plucker(x,y,W=1024,H=512,Rc=1.0):
	varPhi,Phi = xy2angles(x,y,W,H)
	cv,sv = np.cos(varPhi),np.sin(varPhi)
	cp,sp = np.cos(Phi),np.sin(Phi)
	r = np.array([cp*cv,cp*sv,sp]).reshape(3,-1)
	r_b = np.array([Rc*sp*sv,-Rc*sp*cv,np.zeros_like(sp)]).reshape(3,-1)
	ray = np.concatenate((r,r_b),axis=0).reshape(6,-1)
	return ray.T

def angles2plucker(varPhi,Phi,Rc=1.0):
	cv,sv = np.cos(varPhi),np.sin(varPhi)
	cp,sp = np.cos(Phi),np.sin(Phi)
	r = np.array([cp*cv,cp*sv,sp]).reshape(3,)
	r_b = np.array([Rc*sp*sv,-Rc*sp*cv,0]).reshape(3,)
	ray = np.concatenate((r,r_b),axis=0).reshape(6,)
	return ray

def ray2angles(ray):
	r = ray.reshape(-1,)
	phi = np.arcsin(r[2])
	varphi = np.arccos(r[0]/np.cos(phi))
	varphi = varphi if r[1] >= 0 else (-1)*varphi
	return varphi,phi

def xyz2angles(point,Rc=1.0):
	x,y,z = point
	varPhi = np.arctan2(y,x)
	Phi = np.arctan2(z,np.sqrt(x**2+y**2)-Rc)
	return varPhi,Phi

def xyz2uv(point,W=1024,H=512,Rc=1.0):
	varPhi, Phi = xyz2angles(point,Rc)
	u,v = angles2xy(varPhi,Phi,W,H)
	return u,v

# def side(L,M):
# 	L = L.reshape(6,)
# 	M = M.reshape(6,)
# 	ld,lm = L[:3],L[3:]
# 	md,mm = M[:3],M[3:]
# 	a1 = np.dot(ld,mm.T)
# 	a2 =np.dot(md,lm.T)
# 	return a1+a2

def side(L,M):
	W = np.block([[np.zeros((3,3)),np.eye(3)],
				  [np.eye(3),np.zeros((3,3))]])
	return np.dot(L,np.dot(W,M))

def closest_point(L):
	l,l_b = L[:3],L[3:]
	num = np.cross(l,l_b)
	den = float(np.dot(l.reshape(1,3),l))
	return num/den