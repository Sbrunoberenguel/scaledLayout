import numpy as np
from scipy.optimize import least_squares,minimize
import copy
from nc_post.geom import xyz2uv
from nc_post.functions import get_ray, vertical_line_corners


def f_M_vanishing_points(theta,line_list,error):
	L1 = np.array([np.cos(theta),np.sin(theta),0.0],dtype=np.float32)
	L2 = np.array([-np.sin(theta),np.cos(theta),0.0],dtype=np.float32)
	#error = 0
	for i in range(line_list.shape[0]):
		e1 = np.linalg.norm(L1-line_list[i])
		e2 = np.linalg.norm(L2-line_list[i])
		error += min(e1,e2,key=abs)
	return error

def f_A_vanishing_points(theta,line_list,error):
	theta1 = theta[0]
	theta2 = theta[1]
	L1 = np.array([np.cos(theta1),np.sin(theta1),0.0],dtype=np.float32)
	L2 = np.array([np.cos(theta2),np.sin(theta2),0.0],dtype=np.float32)
	#error = 0
	for i in range(line_list.shape[0]):
		e1 = np.linalg.norm(L1-line_list[i])
		e2 = np.linalg.norm(L2-line_list[i])
		error += min(e1,e2,key=abs)
	#error += 1/(abs(np.cos(theta1)-np.cos(theta2))+0.1)
	return error

def f_A_wind_rose(theta,line_list,a):
	error = np.zeros_like(theta)
	theta1 = theta[0]
	theta2 = theta[1]
	theta3 = theta[2]
	theta4 = theta[3]
	theta5 = theta[4]
	theta6 = theta[5]
	L1 = np.array([np.cos(theta1),np.sin(theta1),0.0],dtype=np.float32)
	L2 = np.array([np.cos(theta2),np.sin(theta2),0.0],dtype=np.float32)
	L3 = np.array([np.cos(theta3),np.sin(theta3),0.0],dtype=np.float32)
	L4 = np.array([np.cos(theta4),np.sin(theta4),0.0],dtype=np.float32)
	L5 = np.array([np.cos(theta5),np.sin(theta5),0.0],dtype=np.float32)
	L6 = np.array([np.cos(theta6),np.sin(theta6),0.0],dtype=np.float32)
	#error = 0
	for i in range(line_list.shape[0]):
		e1 = alpha_diff(L1,line_list[i])		#np.linalg.norm(L1-line_list[i])
		e2 = alpha_diff(L2,line_list[i])
		e3 = alpha_diff(L3,line_list[i])
		e4 = alpha_diff(L4,line_list[i])
		e5 = alpha_diff(L5,line_list[i])
		e6 = alpha_diff(L6,line_list[i])
		err  = np.array([e1,e2,e3,e4,e5,e6])
		idx = int(np.where(err == min(error,key=abs))[0])
		error[idx] = err[idx]
	return error


def Manhattan_vanishing_points(ceil_lines,floor_lines,line_length):
	num = ceil_lines.shape[0]
	l_thres = 150
	line_list = []
	for i in range(num):
		if line_length[i] < l_thres:
			continue
		else:
			line_list.append(ceil_lines[i][:3])	
			line_list.append(floor_lines[i][:3])
	
	line_list = np.array(line_list)
	x0_init = 0.0
	try:
		theta = least_squares(f_M_vanishing_points, x0=x0_init, args=(line_list,0), method = 'lm')
		VP = theta.x
	except:
		print('Manhatan direction fails \n  Default theta set \n')
		VP = x0_init
	L1 = np.array([np.cos(VP),np.sin(VP),0.0],dtype=np.float32)
	L2 = np.array([-np.sin(VP),np.cos(VP),0.0],dtype=np.float32)
	L = np.array([L1,L2])
	return L

def Manhattan_wind_rose(ceil_lines,line_length):
	num = ceil_lines.shape[0]
	l_thres = 5
	line_list = []
	for i in range(num):
		if line_length[i] < l_thres:
			continue
		else:
			line_list.append(ceil_lines[i][:3])
	
	line_list = np.array(line_list)
	num_directions = 2
	#gap = np.pi/12.0
	x1_init = 0

	Variables = cluster2DirectionsinPlaneOrthogonal(line_list,num_directions,x1_init)

	VP = Variables[:num_directions]
	# sigma = Variables[num_directions:2*num_directions]
	# P_c = Variables[2*num_directions:]

	P_xc = gaussian_2dir(Variables,ceil_lines[:,:3])

	ang = lambda a: a - 0 # np.pi/2.0
	c_vp = lambda a: np.cos(ang(a))
	s_vp = lambda a: np.sin(ang(a))
 
	# L = np.array([[c_vp(VP[0]),s_vp(VP[0]),0.0],
				#  [-s_vp(VP[0]),c_vp(VP[0]),0.0]])
	L = np.array([[1.0,0.0,0.0],
				 [ 0.0,1.0,0.0]])
	return L, VP, P_xc

def Atlanta_wind_rose(ceil_lines,line_length,num_dir):
	num = ceil_lines.shape[0]
	l_thres = 5
	line_list = []
	for i in range(num):
		if line_length[i] < l_thres:
			continue
		else:
			line_list.append(ceil_lines[i][:3])
	
	line_list = np.array(line_list)
	num_directions = num_dir
	#gap = np.pi/12.0
	x1_init = 0

	Variables = clusterDirectionsinPlaneOrthogonal(line_list,num_directions,x1_init)

	VP = Variables[:num_directions]
	# sigma = Variables[num_directions:2*num_directions]
	# P_c = Variables[2*num_directions:]

	P_xc = gaussian_dir(Variables,ceil_lines[:,:3])

	ang = lambda a: a - 0 # np.pi/2.0
	c_vp = lambda a: np.cos(ang(a))
	s_vp = lambda a: np.sin(ang(a))

	L = [[c_vp(VP[i]),s_vp(VP[i]),0.0] for i in range(num_directions)]
	L = np.array(L)
	return L, VP, P_xc

def re_order_list(c_list):
	longer = []
	for i in range(c_list.shape[0]):
		if i+1 >= c_list.shape[0]:
			end = c_list[0]+1024
		else:
			end = c_list[i+1]
		ini = c_list[i]
		longer.append(end-ini)
	longer = np.array(longer)
	idx = longer.argmax()
	re_order = np.roll(c_list,-idx)
	return re_order,idx

def swap_line_dir(line,L_VP):
	mask = np.ones_like(line[:,:3])
	num = line.shape[0]
	permute = []
	for i in range(num):
		mask[i] = [1,0,0] if abs(line[i,0])>abs(line[i,1]) else [0,1,0]
	for i in range(num):
		prev = abs(mask[i-1][:3])
		curr = abs(mask[i][:3])
		next_ = abs(mask[i+1][:3]) if i+1 < num else abs(mask[0][:3])
		if (prev==curr).all() and (curr==next_).all():
			permute.append(i)
	if len(permute)!=0:
		for i in range(len(permute)):
			if (line[permute[i]][:3] == L_VP[0]).all():
				line[permute[i]][:3] = L_VP[1]
			else:
				line[permute[i]][:3] = L_VP[0]
			# line[permute[i]][0:2] = [line[permute[i]][1],line[permute[i]][0]]
			# line[permute[i]][3:5] = [line[permute[i]][4],line[permute[i]][3]]
	return line

def filter_lines_direction(line):
	num = line.shape[0]
	for i in range(num):
		prev = abs(line[i-1][:3])
		curr = abs(line[i][:3])
		next_ = abs(line[i+1][:3]) if i+1 < num else abs(line[0][:3])
		if (prev==curr).all() and (curr==next_).all():
			l = line[i][:3]
			line[i][:3] = np.array([-l[1],l[0],l[2]])
	return line

def alpha_diff(u,v):
	norm = np.linalg.norm(u)*np.linalg.norm(v)
	try:
		dot_dir = np.dot(u,v)/norm
		dot_inv = np.dot(-u,v)/norm
		alpha1 = np.arccos(dot_dir)
		alpha2 = np.arccos(dot_inv)
		return min(alpha1,alpha2)
	except:
		dot_dir = np.dot(u,v)
		dot_inv = np.dot(-u,v)
		print(dot_dir,dot_inv)
		return 5.0

def Angle2Vector(muAngle):
	muVector = [[np.sin(muAngle[i]),np.cos(muAngle[i])] for i in range(muAngle.shape[0])]
	muVector = np.array(muVector)
	return muVector

def gauss_prob(x,mu,sigma):
	exp_up = np.arcsin(np.dot(mu,x))**2
	exp_down = 2*(sigma**2)
	down = np.sqrt(2*np.pi)*sigma
	expon = np.exp(-1*float(exp_up/exp_down))
	return float(expon/down)

def f_minimize(Variables,x):
	nG = int(len(Variables)/3)
	muAngle = Variables[0:nG]
	sigma = Variables[nG:2*nG]
	P_c = Variables[2*nG:3*nG]
	P_c_norm = P_c.sum()
	P_c = P_c/P_c_norm
	num_mu = muAngle.shape[0]
	num_x = x.shape[0]
	muVector = Angle2Vector(muAngle)
	P_x_given_c = np.zeros((num_mu,num_x))
	P_xc = np.zeros((num_mu,num_x))
	for i in range(num_mu):
		for j in range(num_x):
			P_x_given_c[i,j] = gauss_prob(x[j],muVector[i],sigma[i])
			P_xc[i,j] = P_c[i] * P_x_given_c[i,j]
	res = 0
	for i in range(num_x):
		res += np.log(P_xc.sum(axis=0)[i])
	return -1*res

def f_minimize_2dir(Variables,x):
	nG = int(len(Variables)/3)
	muAngle = Variables[0:nG]
	sigma = Variables[nG:2*nG]
	P_c = Variables[2*nG:3*nG]
	P_c_norm = P_c.sum()
	P_c = P_c/P_c_norm
	num_mu = 2
	num_x = x.shape[0]
	muVector = np.array([[np.sin(muAngle[0]),np.cos(muAngle[0])],
						 [-np.cos(muAngle[0]),np.sin(muAngle[0])]])
	P_x_given_c = np.zeros((num_mu,num_x))
	P_xc = np.zeros((num_mu,num_x))
	for i in range(num_mu):
		for j in range(num_x):
			P_x_given_c[i,j] = gauss_prob(x[j],muVector[i],sigma[i])
			P_xc[i,j] = P_c[i] * P_x_given_c[i,j]
	res = 0
	for i in range(num_x):
		res += np.log(P_xc.sum(axis=0)[i])
	return -1*res

def gaussian_dir(Variables,x):
	x = x[:,:2]
	nG = int(len(Variables)/3)
	muAngle = Variables[0:nG]
	sigma = Variables[nG:2*nG]
	P_c = Variables[2*nG:3*nG]
	P_c_norm = P_c.sum()
	P_c = P_c/P_c_norm
	num_mu = muAngle.shape[0]
	num_x = x.shape[0]
	muVector = Angle2Vector(muAngle)
	P_x_given_c = np.zeros((num_mu,num_x))
	P_xc = np.zeros((num_mu,num_x))
	for i in range(num_mu):
		for j in range(num_x):
			P_x_given_c[i,j] = gauss_prob(x[j],muVector[i],sigma[i])
			P_xc[i,j] = P_c[i] * P_x_given_c[i,j]
	return P_xc

def gaussian_2dir(Variables,x):
	x = x[:,:2]
	nG = int(len(Variables)/3)
	muAngle = Variables[0:nG]
	sigma = Variables[nG:2*nG]
	P_c = Variables[2*nG:3*nG]
	P_c_norm = P_c.sum()
	P_c = P_c/P_c_norm
	num_mu = muAngle.shape[0]
	num_x = x.shape[0]
	muVector = np.array([[np.sin(muAngle[0]),np.cos(muAngle[0])],
						 [-np.cos(muAngle[0]),np.sin(muAngle[0])]])
	P_x_given_c = np.zeros((num_mu,num_x))
	P_xc = np.zeros((num_mu,num_x))
	for i in range(num_mu):
		for j in range(num_x):
			P_x_given_c[i,j] = gauss_prob(x[j],muVector[i],sigma[i])
			P_xc[i,j] = P_c[i] * P_x_given_c[i,j]
	return P_xc

def clusterDirectionsinPlaneOrthogonal(v,nGaussians,thetaIn = 0):
	'''
	IN: Intro of line direction vectors and parameters
	OUT: Angles of main directions as a probability distribution
	METHOD: Optim algorithm (transcript from Jesús Bermudez MatLab implementation)
	'''
	#Variable inicialization
	sigmaIn = 0.5
	eps = 0.05

	muAngle = [thetaIn + i*np.pi/nGaussians for i in range(nGaussians)]
	#muAngle = np.array(muAngle)
	mu_bounds = [(0,np.pi+thetaIn) for i in range(nGaussians)]

	sigmaInit = [sigmaIn for i in range(nGaussians)]
	#sigma = sigmaInit
	sigma_bounds = [(eps,2*sigmaIn) for i in range(nGaussians)]

	P_c = [1/nGaussians for i in range(nGaussians)]
	P_c_bounds = [(0,1) for i in range(nGaussians)]

	bnds = mu_bounds + sigma_bounds + P_c_bounds

	x = v[:,:2]
	Optimizable = muAngle + sigmaInit + P_c
	#Optimizable = np.array([muAngle,sigma,P_c])

	Optim = minimize(f_minimize,Optimizable,args=x, bounds=tuple(bnds))

	sol = Optim.x
	# mu_out = sol[0:nGaussians]
	# sigma_out = sol[nGaussians:2*nGaussians]
	# P_c_out = sol[2*nGaussians:]
	return sol #mu_out,sigma_out,P_c_out

def cluster2DirectionsinPlaneOrthogonal(v,nGaussians,thetaIn = 0):
	'''
	IN: Intro of line direction vectors and parameters
	OUT: Angles of main directions as a probability distribution
	METHOD: Optim algorithm (transcript from Jesús Bermudez MatLab implementation)
	'''
	#Variable inicialization
	sigmaIn = 0.5
	eps = 0.05

	muAngle = [thetaIn for i in range(nGaussians)]
	#muAngle = np.array(muAngle)
	mu_bounds = [(-eps,np.pi) for i in range(nGaussians)]

	sigmaInit = [sigmaIn for i in range(nGaussians)]
	#sigma = sigmaInit
	sigma_bounds = [(eps,2*sigmaIn) for i in range(nGaussians)]

	P_c = [1/nGaussians for i in range(nGaussians)]
	P_c_bounds = [(0,1) for i in range(nGaussians)]

	bnds = mu_bounds + sigma_bounds + P_c_bounds

	x = v[:,:2]
	Optimizable = muAngle + sigmaInit + P_c
	#Optimizable = np.array([muAngle,sigma,P_c])

	Optim = minimize(f_minimize_2dir,Optimizable,args=x, bounds=tuple(bnds))

	sol = Optim.x
	# mu_out = sol[0:nGaussians]
	# sigma_out = sol[nGaussians:2*nGaussians]
	# P_c_out = sol[2*nGaussians:]
	return sol #mu_out,sigma_out,P_c_out