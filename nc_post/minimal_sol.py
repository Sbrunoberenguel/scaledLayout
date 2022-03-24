import numpy as np
from nc_post.geom import xy2plucker, side, xy2angles, angles2plucker
from nc_post.functions import SVD, solve_2Eq, test_roots, get_rays, get_3rays, ray2A, solve_2Eq2


#MINIMAL SOLUTIONS FOR LINE EXTRACTION

def HorLines_VerPlane(ceil_rays,floor_rays):
	num_ceil = ceil_rays.shape[1]
	num_floor = floor_rays.shape[1]
	A_c = np.concatenate((ceil_rays[3,:].reshape(-1,1),ceil_rays[4,:].reshape(-1,1),ceil_rays[1,:].reshape(-1,1),(-1)*ceil_rays[0,:].reshape(-1,1),
					np.zeros((num_ceil,1)),np.zeros((num_ceil,1)),(-1)*ceil_rays[2,:].reshape(-1,1)),axis=1)
	A_f = np.concatenate((floor_rays[3,:].reshape(-1,1),floor_rays[4,:].reshape(-1,1),np.zeros((num_floor,1)),np.zeros((num_floor,1)),
					floor_rays[1,:].reshape(-1,1),(-1)*floor_rays[0,:].reshape(-1,1),(-1)*floor_rays[2,:].reshape(-1,1)),axis=1)

	A = np.concatenate((A_c,A_f),axis=0)
	_,_,Vt = SVD(A)
	V = np.transpose(Vt)
	X_0,X_1 = V[:,-1],V[:,-2]

	u0,u1 = X_0[0:2],X_1[0:2]
	v0,v1 = X_0[2:4],X_1[2:4]
	w0,w1 = X_0[4:6],X_1[4:6]
#	d0,d1 = X_0[6]  ,X_1[6]

	h_c1,h_c2,lambda11,lambda12 = solve_2Eq(u0,u1,v0,v1)
	h_f1,h_f2,lambda21,lambda22 = solve_2Eq(u0,u1,w0,w1)
	
	# if h_c1==0 and h_c2==0 and lambda11==0 and lambda12 == 0:
	# 	return np.zeros(7),np.zeros(6),np.zeros(6)
	# if h_f1==0 and h_f2==0 and lambda21==0 and lambda22 == 0:
	# 	return np.zeros(7),np.zeros(6),np.zeros(6)
	l_11_21 = abs(lambda11-lambda21)
	l_11_22 = abs(lambda11-lambda22)
	l_21_12 = abs(lambda21-lambda12)
	l_12_22 = abs(lambda12-lambda22)
	l_dir = l_11_21 + l_12_22
	l_cro = l_11_22 + l_21_12 
	if l_dir < l_cro:
		hEstA = np.array([h_c1,h_f1])
		hEstB = np.array([h_c2,h_f2])
		lambda1 = np.mean([lambda11,lambda21])
		lambda2 = np.mean([lambda12,lambda22])
	else:
		hEstA = np.array([h_c1,h_f2])
		hEstB = np.array([h_c2,h_f1])
		lambda1 = np.mean([lambda11,lambda22])
		lambda2 = np.mean([lambda12,lambda21])

	hA = hEstA[0]-hEstA[1]
	hB = hEstB[0]-hEstB[1]

	#Multi option solution
	if hEstA[0]>0 and hEstB[0]>0 and hEstA[1]<0 and hEstB[1]<0:
		if hA > 0.2 and hB > 0.2:
			set1,set2,Xi = [hEstA,hEstB], [lambda1,lambda2],[X_0,X_1]
			X,L_ceil,L_floor,hc,hf = verify(ceil_rays,floor_rays,set1,set2,Xi)
			return X,L_ceil,L_floor,hc,hf

	h_ceil,h_floor = 0,0
	if hA > hB:
		if hEstA[0] > 0:
			h_ceil = hEstA[0]
			h_floor = hEstA[1]
			lambda0 = lambda1
	if hB > hA:
		if hEstB[0] > 0:
			h_ceil = hEstB[0]
			h_floor = hEstB[1]
			lambda0 = lambda2
	if h_ceil == 0 and h_floor == 0:
		h_ceil = hEstA[0] if hEstA[0]>hEstA[1] else hEstB[0]
		h_floor = hEstA[1] if hEstA[0]>hEstA[1] else hEstB[1]
		lambda0 = lambda1 if hEstA[0]>hEstA[1] else lambda2
	X = X_0 + lambda0*X_1
	X/= np.linalg.norm(X[0:2])
	depth = X[6]
	L_ceil = np.array([	X[0],		X[1],		0,
						-X[1]*h_ceil,X[0]*h_ceil,-depth])
	L_floor = np.array([X[0],		X[1],		0,
						-X[1]*h_floor,X[0]*h_floor,-depth])
	L_ceil = np.array(L_ceil,dtype=np.float64)
	L_floor = np.array(L_floor,dtype=np.float64)
	return X,L_ceil,L_floor,h_ceil,h_floor

def verify(c_rays,f_rays,set1,set2,Xi):
	X_0,X_1 = Xi[0],Xi[1]
	#Posibility 1
	X = X_0 + set2[0]*X_1
	X/= np.linalg.norm(X[0:2])
	depth = X[6]
	h_ceil1,h_floor1 = set1[0][0],set1[0][1]
	L_c_1 = np.array([X[0],	X[1],0,
		-X[1]*h_ceil1,	X[0]*h_ceil1,	-depth])
	L_f_1 = np.array([X[0],	X[1],0,
		-X[1]*h_floor1,	X[0]*h_floor1,	-depth])
	count1 = 0
	#Posibility 2
	X = X_0 + set2[1]*X_1
	X/= np.linalg.norm(X[0:2])
	depth = X[6]
	h_ceil2,h_floor2 = set1[1][0],set1[1][1]
	L_c_2 = np.array([X[0],	X[1],0,
		-X[1]*h_ceil2,	X[0]*h_ceil2,	-depth])
	L_f_2 = np.array([X[0],	X[1],0,
		-X[1]*h_floor2,	X[0]*h_floor2,	-depth])
	count2 = 0
	if len(c_rays) > 6:
		aux_crays = c_rays[0].reshape(6,-1)
		aux_frays = f_rays[0].reshape(6,-1)
		for i in np.arange(1,len(c_rays)):
			aux_crays = np.concatenate((aux_crays,c_rays[i].reshape(6,-1)),axis=1)
			aux_frays = np.concatenate((aux_frays,f_rays[i].reshape(6,-1)),axis=1)
		c_rays = aux_crays
		f_rays = aux_frays

	axis = min(c_rays.shape[1],f_rays.shape[1])
	for a in range(axis):
		count1 += abs(side(c_rays[:,a],L_c_1)) + abs(side(f_rays[:,a],L_f_1))
		count2 += abs(side(c_rays[:,a],L_c_2)) + abs(side(f_rays[:,a],L_f_2))
	if count1 < count2:
		L_ceil,L_floor = L_c_1,L_f_1
		X = X_0 + set2[0]*X_1
		hc,hf = h_ceil1,h_floor1
	else:
		L_ceil,L_floor = L_c_2,L_f_2
		X = X_0 + set2[1]*X_1
		hc,hf = h_ceil2,h_floor2
	return X,L_ceil,L_floor,hc,hf

def VerWall_CentralAprox(ceil_rays,floor_rays):
	h_c_generic = 1
	h_f_generic = -1.5
	depth_generic = 1
	A_ceil = ceil_rays[:3,:].T
	A_floor = floor_rays[:3,:].T
	_,_,Vt_ceil = SVD(A_ceil)
	_,_,Vt_floor = SVD(A_floor)
	ceil_plane_n = np.transpose(Vt_ceil)[:,-1]
	floor_plane_n = np.transpose(Vt_floor)[:,-1]
	wall_direction = np.cross(ceil_plane_n,floor_plane_n)
	wall_direction /= np.linalg.norm(wall_direction)
	L_ceil = np.array([ wall_direction[0],              wall_direction[1],              0,
					-wall_direction[1]*h_c_generic,  wall_direction[0]*h_c_generic, -depth_generic])
	L_floor = np.array([wall_direction[0],              wall_direction[1],              0,
					-wall_direction[1]*h_f_generic,  wall_direction[0]*h_f_generic, -depth_generic])
	return L_ceil,L_floor

#def HorLines_VerPlane(bon,sample_ceil,sample_floor,num_ceil,num_floor,Rc=1.0):
# def HorLines_VerPlane(bon,ceil_rays,floor_rays):
# 	num_ceil = ceil_rays.shape[0]
# 	num_floor = floor_rays.shape[0]
# 	A = np.zeros((int(num_ceil),7))
# 	for j in range(num_ceil):
# 		xi = ceil_rays[j]
# 		A[j,:] = np.array([[xi[3],xi[4],xi[1],-xi[0],
# 							0,0,-xi[2]]])
# 	for j in range(num_floor):
# 		chi = floor_rays[j]
# 		A = np.concatenate((A,np.array([[chi[3],chi[4],0,0,
# 										 chi[1],-chi[0],-chi[2]]])),axis=0)
# 	A = A.reshape(-1,7)
# 	_,_,Vt = SVD(A)
# 	V = np.transpose(Vt)
# 	X_0,X_1 = V[:,-1],V[:,-2]
# 	#X_2,X_3 = V[:,-3], V[:,-4]
# 	#X_0,X_1 = Vt[-1,:],Vt[-2,:]
	
# 	#X = (ux,uy, vx,vy, wx,wy, d)
# 	u0,u1 = X_0[0:2],X_1[0:2]
# 	v0,v1 = X_0[2:4],X_1[2:4]
# 	w0,w1 = X_0[4:6],X_1[4:6]
# 	d0,d1 = X_0[6]  ,X_1[6]

# 	h_c1,h_c2,lambda11,lambda12 = solve_2Eq(u0,u1,v0,v1)
# 	h_f1,h_f2,lambda21,lambda22 = solve_2Eq(u0,u1,w0,w1)
	
# 	if h_c1==0 and h_c2==0 and lambda11==0 and lambda12 == 0:
# 		return np.zeros(7),np.zeros(6),np.zeros(6)
# 	if h_f1==0 and h_f2==0 and lambda21==0 and lambda22 == 0:
# 		return np.zeros(7),np.zeros(6),np.zeros(6)
# 	l_11_21 = abs(lambda11-lambda21)
# 	l_11_22 = abs(lambda11-lambda22)
# 	l_21_12 = abs(lambda21-lambda12)
# 	l_12_22 = abs(lambda12-lambda22)
# 	l_dir = l_11_21 + l_12_22
# 	l_cro = l_11_22 + l_21_12 
# 	if l_dir < l_cro:
# 		hEstA = np.array([h_c1,h_f1])
# 		hEstB = np.array([h_c2,h_f2])
# 		lambda1 = np.mean([lambda11,lambda21])
# 		lambda2 = np.mean([lambda12,lambda22])       
# 	else:
# 		hEstA = np.array([h_c1,h_f2])
# 		hEstB = np.array([h_c2,h_f1])
# 		lambda1 = np.mean([lambda11,lambda22])
# 		lambda2 = np.mean([lambda12,lambda21])   

# 	lambda0 = lambda1 if hEstA[0]>hEstB[0] else lambda2
# 	X = X_0 + lambda0*X_1
# 	X/= np.linalg.norm(X[0:2])

# 	hA = hEstA[0]-hEstA[1]
# 	hB = hEstB[0]-hEstB[1]
# 	h_ceil,h_floor = 0,0
# 	if hA > hB:
# 		if hEstA[0] > 0:
# 			h_ceil = hEstA[0]
# 			h_floor = hEstA[1]
# 	if hB > hA:
# 		if hEstB[0] > 0:
# 			h_ceil = hEstB[0]
# 			h_floor = hEstB[1]
# 	if h_ceil == 0 and h_floor == 0:
# 		h_ceil = hEstA[0] if hEstA[0]>hEstA[1] else hEstB[0]
# 		h_floor = hEstA[1] if hEstA[0]>hEstA[1] else hEstB[1]
# 	depth = X[6]
# 	L_ceil = np.array([-X[0],-X[1],0,
# 					-X[1]*h_ceil,X[0]*h_ceil,-depth])
# 	L_floor = np.array([-X[0],-X[1],0,
# 					-X[1]*h_floor,X[0]*h_floor,-depth])
# 	return X,L_ceil,L_floor

def parallel_plane_line(Vt,u=[0,0,1],up=True):
	V = np.transpose(Vt)
	u = np.array(u).reshape(3,)
	L0,L1,L2 = V[:,-1],V[:,-2],V[:,-3]
	l0 = L0[:3]
	l1 = L1[:3]
	l2 = L2[:3]
	L00,L01,L02 = side(L0,L0),side(L0,L1),side(L0,L2)
	L11,L12,L22 = side(L1,L1),side(L1,L2),side(L2,L2)
	ul0 = float(np.dot(u.reshape(1,3),l0))
	ul1 = float(np.dot(u.reshape(1,3),l1))
	ul2 = float(np.dot(u.reshape(1,3),l2))
	p = []
	p.append(L22*np.square(ul1)-2*L12*ul1*ul2+L11*np.square(ul2))
	p.append(2*(L01*np.square(ul2)-L02*ul1*ul2-L12*ul0*ul2+L22*ul0*ul1))
	p.append(L22*np.square(ul0)-2*L02*ul0*ul2+L00*np.square(ul2))
	roots = np.roots(p)
	if roots[0].imag != 0:
		return L0,roots,p
	L = test_roots(roots,u,L0,L1,L2,up)
	L /= np.linalg.norm(L[:3])
	L *= np.sign(L[-1])
	return L,roots,p
		
def second_try(bon,c_list,cor,i,Rc=1.0):
	c_ray, f_ray = get_rays(bon,c_list,i,Rc=Rc)
	u = np.array([0,0,1]).reshape(3,)
	#Get ceiling lines and closest point
	A_ceil = ray2A(c_ray)
	_,_,Vt = SVD(A_ceil)
	L_ceil,r,_ = parallel_plane_line(Vt,u,up=True)
	if r[0].imag != 0:
		#print('Imaginary roots in line %.0f: Computing line with only 3 good rays' %i)
		c_ray = get_3rays(bon,c_list,cor,i)
		A_ceil = ray2A(c_ray)
		_,_,Vt = SVD(A_ceil)
		L_ceil,_,_ = parallel_plane_line(Vt,u,up=True)

	#Get floor lines and closest point
	A_floor = ray2A(f_ray)
	_,_,Vt = SVD(A_floor)
	L_floor,r,_ = parallel_plane_line(Vt,u,up=False)
	if r[0].imag != 0:
		#print('Imaginary roots in line %.0f: Computing line with only 3 good rays' %i)
		f_ray = get_3rays(bon,c_list,cor,i)
		A_floor = ray2A(f_ray)
		_,_,Vt = SVD(A_floor)
		L_floor,_,_ = parallel_plane_line(Vt,u,up=False)

	return L_ceil,L_floor
