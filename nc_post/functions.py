import numpy as np
from nc_post.geom import xy2plucker, closest_point, xy2angles, angles2plucker, angles2xy
import scipy.signal as ss


def cluster_peaks(cor):
	delete = True
	cor = cor.reshape(-1,)
	dCor = [cor[i-1]-cor[i] for i in range(cor.size)]
	th = 0.85
	min_gap = 5
	peaks = ss.find_peaks(cor,height=th)
	c_list = peaks[0]
	while delete:
		gaps = c_list-np.roll(c_list,1)
		if min(gaps,key=abs)<min_gap:
			idx = np.where(gaps==min(gaps,key=abs))[0]
			try:
				idx = int(idx)
			except:
				idx = int(idx[0])
			pk1 = peaks[1]['peak_heights'][idx]
			pk2 = peaks[1]['peak_heights'][idx-1]
			if pk1 > pk2:
				c_list = np.delete(c_list,idx-1)
			else:
				c_list = np.delete(c_list,idx)
		else:
			delete=False
	return np.array(c_list),np.array(dCor)*10


def cluster(cor):
	cor = cor.reshape(-1,)
	cor = (cor + np.roll(cor,1))/2
	corner = []
	min_gap = 5
	th = 0.85
	dCor = [cor[i-1]-cor[i] for i in range(cor.size)]
	once = True
	tester = np.concatenate((cor[-5:],cor,cor[:5]))
	for i in range(len(dCor)-2):
		prev1 = dCor[i-1]
#		prev2 = dCor[i-2]
		prev0 = prev1<0 #and prev2<0
		next1 = dCor[i+1]
#		next2 = dCor[i+2]
		next0 = next1>0 #and next2>0
		peak = max(tester[i+5-min_gap:i+5+min_gap])
		if prev0 and next0 and cor[i]>th and cor[i]==peak:
			if once:
				corner.append(i)
				once=False
			elif abs(corner[-1]-i) > min_gap:
				corner.append(i)
	return np.array(corner),np.array(dCor)*10

def SVD(A):
	A = np.array(A,dtype=np.float64)
	u,s,vh = np.linalg.svd(A)
	s_len = s.shape[0]
	v_len = vh.shape[0]
	#S = np.concatenate((np.diag(s),np.zeros((s_len,v_len-s_len))), axis=1)
	return u,s,vh
	
def ray2A(rays):
	A = rays[0].reshape(1,-1)
	W = np.roll(np.eye(6),3,axis=1)	
	once = True
	for r in rays:
		if once:
			once = False
			continue
		A = np.concatenate((A,r.reshape(1,-1)),axis=0)
	A = np.dot(A,W)
	return A

def get_ray(bon,idx=0,W=1024,H=512,Rc=1.0):
	x = idx
	y_ceil = bon[0,x]
	y_floor = bon[1,x]
	r_ceil = xy2plucker(x,y_ceil,W=W,H=H,Rc=Rc)
	r_floor = xy2plucker(x,y_floor,W=W,H=H,Rc=Rc)
	return r_ceil,r_floor

def get_ray_angle(bon,idx=0,W=1024,H=512,Rc=1.0):
	vP,_ = xy2angles(idx,256)
	if vP in bon[0]:
		idx = np.where(bon[0]==vP)[0].reshape(-1,)
		x = bon[0,idx[0]]
		y_ceil = bon[1,idx[0]]
		y_floor = bon[2,idx[0]]
	else:
		aux = min(bon[0,:],key = lambda x:abs(x-vP))
		idx = np.where(bon[0,:]==aux)[0].reshape(-1,)
		x = bon[0,int(idx[0])]
		y_ceil = bon[1,int(idx[0])]
		y_floor = bon[2,int(idx[0])]
	r_ceil = angles2plucker(x,y_ceil,Rc=Rc)
	r_floor = angles2plucker(x,y_floor,Rc=Rc)
	return r_ceil,r_floor, y_ceil,y_floor

def test_roots(roots,u,L0,L1,L2,up):
	x = np.array([1,0,0])
	y = np.array([0,1,0])
	l0,l1,l2 = L0[:3],L1[:3],L2[:3]
	L_ind = (float(np.dot(u.reshape(1,3),l2))*L0
			 - float(np.dot(u.reshape(1,3),l0))*L2)
	L_dep = (float(np.dot(u.reshape(1,3),l2))*L1
			 - float(np.dot(u.reshape(1,3),l1))*L2)
	L_root = np.zeros((2,6))
	votes = np.zeros(2)
	for i in range(roots.size):
		sol = roots[i]
		L_root[i] = L_ind + L_dep*sol
		px = closest_point(L_root[i])
		dx = np.sqrt(px[0]**2+px[1]**2)
		dl = np.absolute(np.linalg.norm(L_root[i][:3]))
		votes[i] += (0.4 if (up==True and px[-1]>0
						 or up==False and px[-1]<0) else -0.8)
		votes[i] += 0.3 if dx < 30.0 else -0.6
		votes[i] += (0.3 if (np.isclose(dl,x,atol=0.01).all()
						 or np.isclose(dl,y,atol=0.01).all()) else -0.6)
	L = L_root[votes.argmax()]
	return L

def are_parallel(L1,L2):
	eps = 0.9
	l1 = L1[:3]
	l2 = L2[:3]
	dot = np.dot(l1,l2)
	l1_n = np.linalg.norm(l1)
	l2_n = np.linalg.norm(l2)
	cos = dot/(l1_n*l2_n)
	if abs(cos) > eps:
		return True
	else:
		return False	

def vertical_line_corners(L,M):
	L, M = L.reshape(6,), M.reshape(6,)
	l,l_b = L[:3],L[3:]
	m,m_b = M[:3],M[3:]
	n = np.cross(l,m)
	A_mat = np.array([l.reshape(1,3),
						m.reshape(1,3),	
						n.reshape(1,3)]).reshape(3,-1)
	try:
		inv = np.linalg.inv(A_mat)
	except:
		N = np.zeros(6)
		X_L = np.array([0,0,0,1])
		X_M = np.array([0,0,0,1])
		return N,X_L,X_M
	B_mat = np.array([	-float(np.dot(l_b.reshape(1,3),n)),
						-float(np.dot(m_b.reshape(1,3),n)),
						0]).reshape(3,-1)
	n_b = np.dot(inv,B_mat).reshape(3,)
	X_L = np.array(np.concatenate((-np.cross(n_b,l_b),
									np.dot(n.reshape(1,3),l_b)), axis=0))
	X_M = np.array(np.concatenate((-np.cross(n_b,m_b),
									np.dot(n.reshape(1,3),m_b)), axis=0))
	N = np.array([	n.reshape(3,),
					n_b.reshape(3,)]).reshape(6,)
	return N, X_L, X_M

def vertical_line_corners_2(L,M):
	L, M = L.reshape(6,), M.reshape(6,)
	l,l_b = L[:3],L[3:]
	m,m_b = M[:3],M[3:]
	n = np.array([0.,0.,1.])
	A_mat = np.array([l.reshape(1,3),
						m.reshape(1,3),
						n.reshape(1,3)]).reshape(3,-1)
	try:
		inv = np.linalg.inv(A_mat)
	except:
		N = np.zeros(6)
		X_L = np.array([0,0,0,1])
		X_M = np.array([0,0,0,1])
		return N,X_L,X_M
	B_mat = np.array([	-float(np.dot(l_b.reshape(1,3),n)),
						-float(np.dot(m_b.reshape(1,3),n)),
						0]).reshape(3,-1)
	n_b = np.dot(inv,B_mat).reshape(3,)
	X_L = np.array(np.concatenate((-np.cross(n_b,l_b),
									np.dot(n.reshape(1,3),l_b)), axis=0))
	X_M = np.array(np.concatenate((-np.cross(n_b,m_b),
									np.dot(n.reshape(1,3),m_b)), axis=0))
	N = np.array([	n.reshape(3,),
					n_b.reshape(3,)]).reshape(6,)
	return N, X_L, X_M

def get_rays(bondaries,corner,idx=0,W=1024,H=512,Rc=1.0):
	if idx+1 >= len(corner):
		x_min,x_max = corner[idx],corner[0]
		x_max += W
	else:
		x_min,x_max = corner[idx],corner[idx+1]
	dx = 0 #int((x_max-x_min)/20.0)
	ceil_rays = []
	floor_rays = []
	for x in np.arange(x_min+dx,x_max-dx,1):
		if x >= W:
			x -= W
		y_ceil = bondaries[0,x]
		y_floor = bondaries[1,x]
		r_ceil = xy2plucker(x,y_ceil,W,H,Rc)
		r_floor = xy2plucker(x,y_floor,W,H,Rc)
		ceil_rays.append(r_ceil)
		floor_rays.append(r_floor)		
	return ceil_rays, floor_rays

def get_3rays(bondaries,c_list,cor,idx=0,W=1024,H=512,Rc=1.0):
	if idx+1 >= len(c_list):
		x_min,x_max = c_list[idx],c_list[0]
		x_max += W
	else:
		x_min,x_max = c_list[idx],c_list[idx+1]
	p_max,p_min = np.amax(cor[x_min:x_max]),np.amin(cor[x_min:x_max])
	thresh = (p_max+p_min)/2.0
	rays = []
	first = True
	for x in np.arange(x_min,x_max,1):
		if x >= W-1:
			x -= W
		if cor[x]<thresh and first:
			first = False
			x1 = x
			y = bondaries[0,x]
			r1 = xy2plucker(x1,y,W,H,Rc)
		if not first and cor[x+1]>thresh:
			x3 = x
			y = bondaries[0,x]
			r3 = xy2plucker(x3,y,W,H,Rc)
			break
	x2 = int((x1+x3)/2)
	y = bondaries[0,x]
	r2 = xy2plucker(x2,y,W,H,Rc)
	rays.append(r1)		
	rays.append(r2)		
	rays.append(r3)		
	return rays	

def solve_2Eq2(u0,u1,v0,v1):
	A = u0[1]*u1[0]-u0[0]*u1[1]
	B = u0[0]*v1[1]+v0[0]*u1[1]-u0[1]*v1[0]-v0[1]*u1[0]
	C = v0[1]*v1[0]-v0[0]*v1[1]
	inSqrt = (B**2)-(4*A*C)
	if inSqrt < 0:
		return 0,0,0,0
	h1 = (-B+np.sqrt(inSqrt))/(2*A)
	h2 = (-B-np.sqrt(inSqrt))/(2*A)
	lambda1 = (h1*u0-v0)/(v1-h1*u1)
	lambda2 = (h2*u0-v0)/(v1-h2*u1)
	return h1,h2,lambda1[0],lambda2[0]

def solve_2Eq(u0,u1,v0,v1):
	A = u0[1]*u1[0]-u0[0]*u1[1]
	B = u0[0]*v1[1]+v0[0]*u1[1]-u0[1]*v1[0]-v0[1]*u1[0]
	C = v0[1]*v1[0]-v0[0]*v1[1]
	inSqrt = (B**2)-(4*A*C)
	if inSqrt < 0:
		return 0,0,0,0
	h1 = (-B+np.sqrt(inSqrt))/(2*A)
	h2 = (-B-np.sqrt(inSqrt))/(2*A)
	a1,b1 = (v1-h1*u1),(h1*u0-v0)
	a2,b2 = (v1-h2*u1),(h2*u0-v0)
	ai1,ai2 = np.linalg.pinv(a1.reshape(2,1)),np.linalg.pinv(a2.reshape(2,1))
	lambda1 = float(np.dot(ai1,b1))
	lambda2 = float(np.dot(ai2,b2))
	# lambda11 = (h1*u0-v0)/(v1-h1*u1)
	# lambda21 = (h2*u0-v0)/(v1-h2*u1)
	# flag1 = a1*lambda1 - b1
	# flag2 = a2*lambda2 - b2
	return h1,h2,lambda1,lambda2

def review_corners_layout(corner,line,num):
	out = []
	for i in range(num):
		left_C = corner[0,i]
		left_line = line[i-1]
		right_C = corner[2,i]
		rigth_line = line[i]
		center_corner = corner[1,i]
		same_point = np.isclose(left_C,right_C,atol=1.).all()
		same_dir = np.isclose(left_line[:3],rigth_line[:3],atol=0.05).all()
		if same_dir and not same_point:
			out.append(left_C)
			out.append(right_C)
		else:
			out.append(center_corner)
		# else:	
		# 	out.append(left_C)
		# 	out.append(right_C)
	return np.array(out)

def noise(rays,sigma):
	rays = np.array(rays,dtype=np.float64)
	Phi = np.arcsin(rays[:,2])
	varPhi = np.arctan2(rays[:,1],rays[:,0])
	u,v = angles2xy(varPhi,Phi)
	u += np.random.normal(0,sigma,u.shape)
	v += np.random.normal(0,sigma,v.shape)
	rays = xy2plucker(u,v)
	return rays