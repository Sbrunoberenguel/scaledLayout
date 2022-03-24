import numpy as np
import copy
from nc_post.functions import SVD, get_ray, get_ray_angle, solve_2Eq
from nc_post.minimal_sol import verify
#PER-WALL DLT
'''
WallDLT takes as imput a wall direction (known) and the boundaries
    - For each pair of lines computes the wall: two lines that belong to a vertical plane
        · X = (1, h_c, h_f, d)
    - In the main program, the correct wall direction is chosen from the 2 posibilities

GlobalDLT takes as input all the lines and the boundaries
    - The direction of each line is known and assumend correct
    - An SVD solution is proposed, setting each projecting ray in the reference of its wall
    - A global solution for the layout is given
        · X = (1, h_c, h_f, [d_i])

'''
def wall_DLT(bon,c_list,idx,L,Rc=1.0):
	num_walls = c_list.shape[0]
	l1,l2 = L[0],L[1]
	R_a_L = np.array([	[l1,l2,0],
						[-l2,l1,0],
						[0,0,1]])
	G_a_L = np.block([	[R_a_L,np.zeros((3,3))],
						[np.zeros((3,3)),R_a_L]])
	once = True
	if idx+1>=num_walls:
		low,high = c_list[idx],c_list[0]
	else:
		low,high = c_list[idx],c_list[idx+1]
	if high < low:
		axis1 = np.arange(low,1024)
		axis2 = np.arange(0,high)
		axis = np.concatenate((axis1,axis2),axis=0)
	else:
		axis = np.arange(low,high,1)
	for i in axis:

		c_ray, f_ray = get_ray_angle(bon,i,Rc=Rc)
		xi = np.dot(G_a_L,c_ray)
		chi = np.dot(G_a_L,f_ray)
		if once:
			once=False
			A = np.array([	[xi[3],xi[1],0,-xi[2]],
							[chi[3],0,chi[1],-chi[2]]])
		else:
			B = np.array([	[xi[3],xi[1],0,-xi[2]],
							[chi[3],0,chi[1],-chi[2]]])
			A = np.concatenate((A,B),axis=0)

	# X = (1,h_c,h_f,d)
	# A · X = 0

	_,_,Vt = SVD(A)
	sol = np.transpose(Vt)[:,-1]
	sol /= sol[0]
	h_c,h_f,depth = sol[1],sol[2],sol[3]
	ceil_lines_out = np.array([l1,l2,0,-l2*h_c,l1*h_c,-depth])
	floor_lines_out = np.array([l1,l2,0,-l2*h_f,l1*h_f,-depth])
	ceil_lines_out *= np.sign(ceil_lines_out[-1])
	floor_lines_out *= np.sign(floor_lines_out[-1])

	return ceil_lines_out, floor_lines_out, h_c, h_f


def getBlockGeneral(wallc_rays,wallf_rays,num_walls,idx):
	if wallc_rays.size < 10:
		depth_c = np.zeros((1,num_walls))
		depth_f = np.zeros((1,num_walls))
		depth_c[0,idx] = (-1)*wallc_rays[2]
		depth_f[0,idx] = (-1)*wallf_rays[2]
		A_c = np.array([[wallc_rays[3],wallc_rays[1],0.]])
		A_f = np.array([[wallf_rays[3],0.,wallf_rays[1]]])
		Ac = np.concatenate((A_c,depth_c),axis=1)
		Af = np.concatenate((A_f,depth_f),axis=1)
	else:
		num_ceil = wallc_rays.shape[1]
		num_floor = wallf_rays.shape[1]
		depth_c = np.zeros((num_ceil,num_walls))
		depth_f = np.zeros((num_ceil,num_walls))
		depth_c[:,idx] = (-1)*wallc_rays[2,:]
		depth_f[:,idx] = (-1)*wallf_rays[2,:]
		A_c = np.concatenate((wallc_rays[3,:].reshape(-1,1),wallc_rays[1,:].reshape(-1,1),np.zeros((num_ceil,1))),axis=1)
		A_f = np.concatenate((wallf_rays[3,:].reshape(-1,1),np.zeros((num_floor,1)),wallf_rays[1,:].reshape(-1,1)),axis=1)
		Ac = np.concatenate((A_c,depth_c),axis=1)
		Af = np.concatenate((A_f,depth_f),axis=1)
	return np.concatenate((Ac,Af),axis=0)




#GLOBAL DLT
def global_DLT(c_rays,f_rays,num_walls,c_list,ceil_lines,floor_lines):
	once = True

	for idx in range(num_walls):
		L = ceil_lines[idx]
		l1,l2 = L[0],L[1]
		R_a_L = np.array([	[l1,l2,0],
							[-l2,l1,0],
							[0,0,1]])
		G_a_L = np.block([	[R_a_L,np.zeros((3,3))],
							[np.zeros((3,3)),R_a_L]])

		wallc_rays = np.dot(G_a_L,c_rays[idx])
		wallf_rays = np.dot(G_a_L,f_rays[idx])
		if once:
			once = False
			A = getBlockGeneral(wallc_rays,wallf_rays,num_walls,idx)
		else:
			Ab = getBlockGeneral(wallc_rays,wallf_rays,num_walls,idx)
			A = np.concatenate((A,Ab),axis=0)

	
	# X = (1,h_c,h_f,[d_i])
	# A · X = 0
	_,_,Vt = SVD(A)
	sol = np.transpose(Vt)[:,-1]
	sol /= sol[0]
	if sol[1] < sol[2]:
		sol *= -1
	h_c,h_f = sol[1],sol[2]
	depths = sol[3:]
	ceil_lines_out = copy.copy(ceil_lines)
	floor_lines_out = copy.copy(floor_lines)
	for j in range(num_walls):

		ceil_lines_out[j][3:] = np.array([[-ceil_lines[j][1]*h_c,
											ceil_lines[j][0]*h_c,
											-depths[j]]])
		floor_lines_out[j][3:] = np.array([[-floor_lines[j][1]*h_f,
											 floor_lines[j][0]*h_f,
											-depths[j]]])
		ceil_lines_out[j] *= np.sign(ceil_lines_out[j][-1])
		floor_lines_out[j] *= np.sign(floor_lines_out[j][-1])
	return ceil_lines_out, floor_lines_out, h_c, h_f


def get_c_row(ray,label,num,idx):
	#ray = [0,1,2|3,4,[5]]
	depth = np.zeros(num).reshape(1,-1)
	depth[0,idx] = -ray[2]
	if label==1:
		row = np.array([ray[3],ray[4],ray[1],-ray[0],0,0]).reshape(1,-1)
	else:
		row = np.array([ray[4],-ray[3],-ray[0],-ray[1],0,0]).reshape(1,-1)
	row = np.concatenate((row,depth),axis=1)
	return row.reshape(1,-1)

def get_f_row(ray,label,num,idx):
	#ray = [0,1,2|3,4,5]
	depth = np.zeros(num).reshape(1,-1)
	depth[0,idx] = -ray[2]
	if label==1:
		row = np.array([ray[3],ray[4],0,0,ray[1],-ray[0]]).reshape(1,-1)
	else:
		row = np.array([ray[4],-ray[3],0,0,-ray[0],-ray[1]]).reshape(1,-1)
	row = np.concatenate((row,depth),axis=1)
	return row.reshape(1,-1)

def getBlockManhattan(i,ceil_rays,floor_rays,label_dir,num_walls):
	if ceil_rays.size<10:
		A_c = get_c_row(ceil_rays,label_dir,num_walls,i)
		A_f = get_f_row(floor_rays,label_dir,num_walls,i)
		return np.concatenate((A_c,A_f),axis=0)

	depth_c = np.zeros((ceil_rays.shape[1],num_walls))
	depth_c[:,i] = -ceil_rays[2,:]
	depth_f = np.zeros((floor_rays.shape[1],num_walls))
	depth_f[:,i] = -floor_rays[2,:]
	if label_dir==1:
		A_c = np.concatenate((	ceil_rays[3,:].reshape(-1,1), ceil_rays[4,:].reshape(-1,1), 
								ceil_rays[1,:].reshape(-1,1), -ceil_rays[0,:].reshape(-1,1),
								np.zeros((ceil_rays.shape[1],1)),np.zeros((ceil_rays.shape[1],1)),
								depth_c),axis=1)
		A_f = np.concatenate((	floor_rays[3,:].reshape(-1,1), floor_rays[4,:].reshape(-1,1), 
								np.zeros((floor_rays.shape[1],1)),np.zeros((floor_rays.shape[1],1)),
								floor_rays[1,:].reshape(-1,1), -floor_rays[0,:].reshape(-1,1),
								depth_f),axis=1)
	else:
		A_c = np.concatenate((	ceil_rays[4,:].reshape(-1,1), -ceil_rays[3,:].reshape(-1,1), 
								-ceil_rays[0,:].reshape(-1,1),-ceil_rays[1,:].reshape(-1,1), 
								np.zeros((ceil_rays.shape[1],1)),np.zeros((ceil_rays.shape[1],1)),
								depth_c),axis=1)
		A_f = np.concatenate((	floor_rays[4,:].reshape(-1,1), -floor_rays[3,:].reshape(-1,1), 
								np.zeros((floor_rays.shape[1],1)),np.zeros((floor_rays.shape[1],1)),
								-floor_rays[0,:].reshape(-1,1),-floor_rays[1,:].reshape(-1,1), 
								depth_f),axis=1)
	return np.concatenate((A_c,A_f),axis=0)


#MANHATTAN GLOBAL DLT
def Manhattan_global_DLT(ceil_rays,floor_rays,ceil_lines,floor_lines,dir_label,c_list):
	once = True
	num_walls = ceil_lines.shape[0]

	for i in range(num_walls):
		if once:
			once = False
			A = getBlockManhattan(i,ceil_rays[i],floor_rays[i],dir_label[i],num_walls)
		else:
			Ab = getBlockManhattan(i,ceil_rays[i],floor_rays[i],dir_label[i],num_walls)
			A = np.concatenate((A,Ab),axis=0)

	_,_,Vt = SVD(A)
	V = np.transpose(Vt)
	_,X = solution_extractor(V,ceil_rays,floor_rays)
	X /= np.linalg.norm(X[:2])
	L = X[0:2]
	dir_norm = np.linalg.norm(L)
	L /= dir_norm
	h_c = np.linalg.norm(X[2:4])/dir_norm
	h_f = (-1)*np.linalg.norm(X[4:6])/dir_norm
	depths = X[6:]
	ceil_lines_out = copy.copy(ceil_lines)
	floor_lines_out = copy.copy(floor_lines)
	for i in range(num_walls):
		if dir_label[i] == 1:
			l_dir = np.array([L[0],L[1],0])
		else:
			l_dir = np.array([-L[1],L[0],0])
		ceil_lines_out[i][:3] = l_dir
		floor_lines_out[i][:3] = l_dir
		ceil_lines_out[i][3:] = np.array([[-ceil_lines_out[i][1]*h_c,
											ceil_lines_out[i][0]*h_c,
											-depths[i]]])
		floor_lines_out[i][3:] = np.array([[-floor_lines_out[i][1]*h_f,
											floor_lines_out[i][0]*h_f,
											-depths[i]]])
		ceil_lines_out[i] *= np.sign(ceil_lines_out[i][-1])
		floor_lines_out[i] *= np.sign(floor_lines_out[i][-1])
	return ceil_lines_out, floor_lines_out, h_c, h_f


def solution_extractor(V,ceil_rays,floor_rays):
	X_0,X_1 = V[:,-1],V[:,-2]

	u0,u1 = X_0[0:2],X_1[0:2]
	v0,v1 = X_0[2:4],X_1[2:4]
	w0,w1 = X_0[4:6],X_1[4:6]
#	d0,d1 = X_0[6]  ,X_1[6]

	h_c1,h_c2,lambda11,lambda12 = solve_2Eq(u0,u1,v0,v1)
	h_f1,h_f2,lambda21,lambda22 = solve_2Eq(u0,u1,w0,w1)
	
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
			X,L_ceil,L_floor,_,_ = verify(ceil_rays,floor_rays,set1,set2,Xi)
			return 0,X

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
	return lambda0,X