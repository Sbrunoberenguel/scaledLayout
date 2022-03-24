## Scaled 360 layouts: Revisiting non-central panoramas
## Berenguel-Baeta, Bermudez-Cameo, Guerrero

'''
Post-processing program for non-central circular panoramas
At date: 12-Apr-2021
	- Handle Manhattan with occlusions
	- Pipeline: RANSAC-VP-Occlusions-GlobalDLT-3DCorners-FinalAdj
	- VP sets the direction of the walls (only fit fot Manhattan)
	- RANSAC: lines = f(bondaries)
	- VP: MW 2 perpendicular
	- Occlusions: Only for MW
	- GlobalDLT: 2 solutions, one for MW and other for AW
		- MW-sol: adjust the layout with wall label dir known
		- AW-sol: adjust the layout with wall direction known
	- Bundle Adjustment for AW
	- Vectorized
'''

import numpy as np
from cv2 import cv2
import open3d as op
import argparse
import os
import copy
import warnings
from tqdm import trange

from functions import  cluster_peaks, cluster, vertical_line_corners, vertical_line_corners_2, get_ray, review_corners_layout, get_ray_angle, noise
from RANSAC import  RANSAC_2LinesVerPlane
from rendering import  draw_corner_probability, draw_corners, layout_3D_lines, layout_3D_corners, layout_reconstruction, test
from direction_optim import  swap_line_dir, Manhattan_wind_rose, Atlanta_wind_rose
from DLT import  wall_DLT, global_DLT, Manhattan_global_DLT
from occlusion import  occlusion_manhattan, occlusion_atlanta
from minimal_sol import  HorLines_VerPlane, VerWall_CentralAprox
from geom import  xy2angles,angles2xy,side,closest_point
from bundleAdjustment import bundle_adjustment

warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
#MAIN PROGRAM

def pipe(args,sigma_noise,thres):
#Input parameters
	#Image name
	name = args.data.split('/')[-1]
	#World assumption: Manhattan OR Atlanta (not both)
	manhattan = args.Manhattan
	atlanta = not manhattan
	if manhattan:
		adjustment='manhattan'
	else:
		adjustment='general'
	#Temporal fix to avoid direction computation in atlanta cases (PTE know num_dir)
#	args.dir = args.dir * manhattan
	#Non-central panorama radius
	Rc = args.Rc

	path = args.data
	if not os.path.isdir(path):
		print('Output directory %s didn\'t exist. One created' % path)
		os.makedirs(path)
	
	np.set_printoptions(precision=5,suppress=True)
	#Director vector of known plane: assumption horizontal plane	
	#u = np.array([[0,0,1]])
	#Output of the network
	px_bon = np.load(path+'_edges.npy') ##Should start in -PI and go to +PI, else flip
	# px_bon = np.load(path+'.npy')
	px_bon = np.flip(px_bon,axis=1)
	# cor = np.load(args.img)
	cor = np.load(path+'_corners.npy')
	cor = np.flip(cor)
	c_list,dCor = cluster(cor)
	num_walls = len(c_list)

	#Case of HorizonNet-like output (in pixels)
	if px_bon.shape[0]==2:
		if np.amin(px_bon)<0 or np.amax(px_bon)>511:
			for k in range(1024):
				px_bon[0,k] = max(0,px_bon[0,k])
				px_bon[1,k] = min(511,px_bon[1,k])
		theta,vp1 = xy2angles(np.arange(1024),px_bon[0])
		_,vp2 = xy2angles(np.arange(1024),px_bon[1])
		bon = np.concatenate((theta,vp1,vp2),axis=0).reshape(3,-1)
	#Case of angle-data
	else:
		bon = copy.copy(px_bon)
		px_bon = np.zeros((2,1024))
		for k in range(1024):
			vP,_ = xy2angles(k,256)
			aux = min(bon[0,:],key = lambda x:abs(x-vP))
			idx = np.where(bon[0,:]==aux)[0]
			_,px_bon[0,k] = angles2xy(aux,bon[1,int(idx[0])])
			px_bon[0,k] = max(px_bon[0,k],0)
			_,px_bon[1,k] = angles2xy(aux,bon[2,int(idx[0])])
			px_bon[1,k] = min(px_bon[1,k],511)


	bon = np.array(bon,dtype=np.float64)
	cor = np.array(cor,dtype=np.float64)

	c_rays = []
	f_rays = []
	# angles = np.zeros((2,1024))
	for a in range(1024):
		cr,fr,yc,yf = get_ray_angle(bon,a,Rc=Rc)
		c_rays.append(cr)
		f_rays.append(fr)
		# angles[0,a] = yc
		# angles[1,a] = yf

	# ang_name = path.split('/')[-1] + '.npy'
	# np.save('ManhattanTest/img_angles/'+ang_name,angles)
	c_rays = np.array(c_rays,dtype=np.float64)
	f_rays = np.array(f_rays,dtype=np.float64)

	#Here introduce noise for evaluation purposes
	if sigma_noise != 0:
		c_rays = noise(c_rays,sigma_noise)
		f_rays = noise(f_rays,sigma_noise)

	wall_crays = []
	wall_frays = [] 
	wall_bon = []
	for i in range(num_walls):
		if i+1>=num_walls:
			low,high = c_list[i],c_list[0]
			axis1 = np.arange(low+1,1024)
			axis2 = np.arange(0,high)
			axis = np.concatenate((axis1,axis2),axis=0)
		else:
			low,high = c_list[i],c_list[i+1]
			axis = np.arange(low+1,high,1)
		wall_crays.append(c_rays[axis].T)
		wall_frays.append(f_rays[axis].T)
		wall_bon.append(bon[:,axis])

	draw_corner_probability(cor,dCor,c_list,path)
	
	#IMPORTANT THRESHOLD #
	if Rc != 0:
		if manhattan:
			pixel_thres = 50
		else:
			pixel_thres = 25
	else:
		pixel_thres = 1024
	#&&&&&&&&&&&&&&&&&&&&&&&&&&&#

	if args.ran:
		#RANSAC parameters
		P = 0.999
		eps = 0.5		#ratio of inliers
		m = 3			#samples
		thres1 = 0.1	#meters
		thres2 = 1		#pixels
		#Get lines from network output: RANSAC tipe algorithm		
		ceil_lines,floor_lines,line_length = RANSAC_2LinesVerPlane(wall_bon,wall_crays,wall_frays,c_list,pixel_thres,P,eps,m,thres1,thres2,Rc=Rc)
		ceil_lines = np.array(ceil_lines)
		floor_lines = np.array(floor_lines)	
		line_length = np.array(line_length)

		#Correction in case any wall could not be recovered
		num_walls = ceil_lines.shape[0]

		#Checkpoint1
		if args.save:
			np.save('{}/ceil_lines_afterRANSAC.npy'.format(path),ceil_lines)
			np.save('{}/floor_lines_afterRANSAC.npy'.format(path),floor_lines)

			# _,_,_ = layout_3D_lines(name+'_ransac',path,ceil_lines,floor_lines)
	else:
		#Line extraction without the RANSAC (same implementation)
		gaps = np.roll(c_list - np.roll(c_list,1),-1)
		gaps[-1] += 1024
		# first = int(np.where(gaps==np.amax(gaps))[0])
		walls_idx = np.arange(num_walls)
		# walls_idx = np.roll(walls_idx,-first)
		ceil_lines = []
		floor_lines = []
		line_length = []
		for i in walls_idx:
			ceil_rays = wall_crays[i]
			floor_rays = wall_frays[i]
			l_length = ceil_rays.shape[1]
			if l_length < pixel_thres:
				L_ceil,L_floor = VerWall_CentralAprox(ceil_rays,floor_rays)
			else:
				_,L_ceil,L_floor,h_c,h_f = HorLines_VerPlane(ceil_rays,floor_rays)

			ceil_lines.append(L_ceil)
			floor_lines.append(L_floor)
			line_length.append(ceil_rays.shape[1])
		ceil_lines = np.array(ceil_lines)
		floor_lines = np.array(floor_lines)	
		line_length = np.array(line_length)
		if args.save:
			np.save('{}/ceil_lines_NoRANSAC.npy'.format(path),ceil_lines)
			np.save('{}/floor_lines_NoRANSAC.npy'.format(path),floor_lines)

#Get Main directions	
	if args.dir:
		if manhattan:
			#print('Manhattan world assumption \n')
			#Two perpendicular main directions
			#L_VP = Manhattan_vanishing_points(ceil_lines, floor_lines, line_length)
			#Statistic method assuming 2 orthogonal directions
			L_VP,_,prob_vp = Manhattan_wind_rose(ceil_lines,line_length)

		elif atlanta:
			#print('Atlanta world assumption \n')
			#Stadistic method assuming num_dir arbitrary main directions
			L_VP,_,prob_vp = Atlanta_wind_rose(ceil_lines,line_length,args.num_dir)
		if args.save:
			np.save('{}/VP_dir.npy'.format(path),L_VP)
		
		ceil_lines_op1 = copy.copy(ceil_lines)
		floor_lines_op1 = copy.copy(floor_lines)
		for i in range(num_walls):
			idx = np.where(prob_vp[:,i] == np.amax(prob_vp[:,i]))[0]
			ceil_lines_op1[i][:3] = L_VP[int(idx)]
			floor_lines_op1[i][:3] = L_VP[int(idx)]

		if manhattan:
			ceil_lines_op1 = swap_line_dir(ceil_lines_op1,L_VP)
			floor_lines_op1 = swap_line_dir(floor_lines_op1,L_VP)
	else:
		ceil_lines_op1 = copy.copy(ceil_lines)
		floor_lines_op1 = copy.copy(floor_lines)
		L_VP = [ceil_lines_op1[0,:3]]


#Manage occlusions
	#Only for manhattan world assumption
	if args.occ:
		if manhattan:
			ceil_lines_op2, floor_lines_op2,c_list,wall_bon,wall_crays,wall_frays,occ_found = occlusion_manhattan(L_VP,ceil_lines_op1,floor_lines_op1,
																						c_list,wall_bon,wall_crays,wall_frays,bon,c_rays,f_rays)
			#Save new corner list after occlusion management
		
			ceil_lines_op2 = swap_line_dir(ceil_lines_op2,L_VP)
			floor_lines_op2 = swap_line_dir(floor_lines_op2,L_VP)
			if occ_found:
				args.fa = True
			if args.save:
				np.save('{}/corner_list_op.npy'.format(path),c_list)
		else:
			ceil_lines_op2, floor_lines_op2 = copy.copy(ceil_lines_op1), copy.copy(floor_lines_op1)

		#Checkpoint3
		if args.save:
			np.save('{}/ceil_lines_afterOcclusion.npy'.format(path),ceil_lines_op2)
			np.save('{}/floor_lines_afterOcclusion.npy'.format(path),floor_lines_op2)
	else:
		#PTE correct definition
		ceil_lines_op2, floor_lines_op2 = ceil_lines_op1, floor_lines_op1
	

#Global DLT
	if args.DLT:
		num_walls = ceil_lines_op2.shape[0]
		if adjustment=='general':
			ceil_lines_op3, floor_lines_op3, h_c, h_f = global_DLT(wall_crays,wall_frays,num_walls,c_list,
																ceil_lines_op2,floor_lines_op2)

		elif adjustment=='manhattan':
			dir_label = []
			if args.dir:
				for i in range(ceil_lines_op2.shape[0]):
					if (ceil_lines_op2[i][:3] == L_VP[0]).all():
						dir_label.append(1)
					else:
						dir_label.append(2)
			else:
				dir_label = [i%2 for i in range(ceil_lines_op2.shape[0])]				
			ceil_lines_op3, floor_lines_op3, h_c, h_f = Manhattan_global_DLT(wall_crays,wall_frays,
															ceil_lines_op2,floor_lines_op2,dir_label,c_list)
		else:
			print('Adjusment ERROR')
		num_walls = ceil_lines_op3.shape[0]
	else:
		ceil_lines_op3 = ceil_lines_op2 
		floor_lines_op3 = floor_lines_op2 

#Save final lines
	if args.save:
		np.save('{}/ceil_lines_afterDLT.npy'.format(path),ceil_lines_op3)
		np.save('{}/floor_lines_afterDLT.npy'.format(path),floor_lines_op3)

#Get intersection points between lines - corners
	ceil_C = []
	floor_C = []
	if manhattan:
		for i in range(num_walls):
			C1,C2 = ceil_lines_op3[i],ceil_lines_op3[i-1]
			F1,F2 = floor_lines_op3[i],floor_lines_op3[i-1]
			_,pc,_ = vertical_line_corners_2(C1,C2)
			_,pf,_ = vertical_line_corners_2(F1,F2)
			c_point = pc[:3]/pc[3]
			f_point = pf[:3]/pf[3]
			ceil_C.append(c_point)
			floor_C.append(f_point)
	else:
		new_c_list = copy.copy(c_list)
		for i in range(num_walls):
			ray_c_c,ray_f_c = c_rays[c_list[i],:],f_rays[c_list[i],:]
			c_line_l, c_line_r = ceil_lines_op3[i-1],  ceil_lines_op3[i]
			f_line_l, f_line_r = floor_lines_op3[i-1], floor_lines_op3[i]
			_,p1,_ = vertical_line_corners_2(c_line_l,c_line_r)
			p1 = p1[:3]/p1[3]
			_,x_L_l,_ = vertical_line_corners_2(c_line_l,ray_c_c)
			pl = x_L_l[:3]/x_L_l[3]
			_,x_L_r,_ = vertical_line_corners_2(c_line_r,ray_c_c)
			pr = x_L_r[:3]/x_L_r[3]
			#Check occlusions
			occluded = (np.linalg.norm(pr-p1) > 4) and (np.linalg.norm(pl-p1) > 4)
			if occluded and args.occ:
				#Add occluded data
				new_c_list = np.insert(new_c_list,i,c_list[i])
				wall_bon.insert(i,bon[:,c_list[i]].reshape(3,1))
				wall_crays.insert(i,ray_c_c)
				wall_frays.insert(i,ray_f_c)
				#Ceil corners
				ceil_C.append(pl)
				ceil_C.append(pr)
				#Floor corners
				_,x_L_l,_ = vertical_line_corners_2(f_line_l,ray_f_c)
				pl = x_L_l[:3]/x_L_l[3]
				_,x_L_r,_ = vertical_line_corners_2(f_line_r,ray_f_c)
				pr = x_L_r[:3]/x_L_r[3]
				floor_C.append(pl)
				floor_C.append(pr)
				num_walls += 1
			else:
				#Ceil corner
				ceil_C.append(pl)
				#Floor corner
				_,x_L_center,_ = vertical_line_corners_2(f_line_l,ray_f_c)
				pc = x_L_center[:3]/x_L_center[3]
				floor_C.append(pc)

		c_list = new_c_list
	ceil_layout = np.array(ceil_C).reshape(-1,3)
	floor_layout = np.array(floor_C).reshape(-1,3)

	if args.test:
		_,p_ceiling,p_floor,p_walls = layout_3D_corners(path,ceil_layout,floor_layout)
		draw_corners(args.img,os.path.join(path,name),p_ceiling,p_floor,p_walls)
		return ceil_layout, floor_layout

	#Final adjustment
	if args.fa:
		ceil_layout,floor_layout,h_c,h_f = bundle_adjustment(manhattan,ceil_layout,floor_layout,wall_bon,c_list,wall_crays,wall_frays)
		ceil_lines_Final = np.zeros((num_walls,6))
		floor_lines_Final = np.zeros((num_walls,6))
		for i in range(num_walls):
			ceil_lines_Final[i,:3] = ceil_layout[i-1] - ceil_layout[i]
			ceil_lines_Final[i,3:] = np.cross(ceil_layout[i],ceil_layout[i-1])
			floor_lines_Final[i,:3] = floor_layout[i-1] - floor_layout[i]
			floor_lines_Final[i,3:] = np.cross(floor_layout[i],floor_layout[i-1])
		ceil_lines_Final = np.roll(ceil_lines_Final,-1,axis=0)
		floor_lines_Final = np.roll(floor_lines_Final,-1,axis=0)
		ceil_lines_Final = ceil_lines_Final/np.linalg.norm(ceil_lines_Final[:,:3],axis=1,keepdims=True)
		floor_lines_Final = floor_lines_Final/np.linalg.norm(floor_lines_Final[:,:3],axis=1,keepdims=True)
	else:
		ceil_lines_Final = ceil_lines_op3
		floor_lines_Final = floor_lines_op3

	#Save room corners

	np.save('{}/ceil_corners.npy'.format(path),ceil_layout)
	np.save('{}/floor_corners.npy'.format(path),floor_layout)
	np.save('{}/ceil_lines_Final.npy'.format(path),ceil_lines_Final)
	np.save('{}/floor_lines_Final.npy'.format(path),floor_lines_Final)

	#Output: Images and point clouds
	if args.pcd:
		layout_corners,p_ceiling,p_floor,p_walls = layout_3D_corners(path,ceil_layout,floor_layout)	
		
		_,_,_ = layout_3D_lines(name+'_optim',path,ceil_lines_Final,floor_lines_Final)
		draw_corners(args.img,os.path.join(path,name),p_ceiling,p_floor,p_walls)
		walls,ceiling,floor = layout_reconstruction(ceil_layout,floor_layout,px_bon,c_list,h_c,h_f)

		std = np.array([512,1024,3])
		img_name = args.img.split('/')[-1]
		img = cv2.imread(args.img)
		img = cv2.flip(img,1)
		if (img.shape != std).any():
			img = cv2.resize(img,(1024,512))
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_rgb = img_rgb/255.0
		room = floor+walls
		if args.ceiling:
			room += ceiling
		room_3D = op.geometry.PointCloud()
		room_3D.points = op.utility.Vector3dVector(room.reshape(-1,3))
		room_3D.colors = op.utility.Vector3dVector(img_rgb.reshape(-1,3))
		op.io.write_point_cloud('{}/{}.pcd'.format(path,img_name[:-4]),room_3D)
		op.io.write_point_cloud('{}/{}_wire.pcd'.format(path,img_name[:-4]),layout_corners)
		if args.visualize:
			print('\007')
			op.visualization.draw_geometries([layout_corners])
			op.visualization.draw_geometries([room_3D])

	return ceil_layout,floor_layout


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', required=True, help='Input data to post-process')
	parser.add_argument('--img', required=True, help='Input image')
	parser.add_argument('--visualize', action='store_true', default=False,help='Visualize 3D layout lines after processing')
	parser.add_argument('--Manhattan', action='store_true', default=False,help='Force Manhattan world assumption')
	parser.add_argument('--num_dir', default='6', type=int, help='Set numer of wall directions on Atlanta world assumption')
	parser.add_argument('--ceiling', action='store_true', default=False,help='Includes ceiling in 3D room reconstruction')
	parser.add_argument('--Rc', default='1.0', type=float)
	parser.add_argument('--save', action='store_true',default=False,help='Store intermediate solutions')
	#---------------------------------------------------------------------------
	parser.add_argument('--ran', action='store_false',default=True,help='Do not implement RANSAC for direction estimation')
	parser.add_argument('--dir', action='store_true',default=False,help='Direction optimization. Manages Manhattan. For Atlanta world, MUST provide number of directions.')
	parser.add_argument('--occ', action='store_true',default=False,help='Handles occlusions')
	parser.add_argument('--DLT', action='store_false',default=True,help='Jumps global DLT')
	parser.add_argument('--pcd', action='store_false',default=True,help='Jumps room 3D visual reconstruction')
	parser.add_argument('--test',action='store_true',default=False,help='Exits on test')
	parser.add_argument('--fa', action='store_true', default=False,help='Uses Final adjustment')	
	args = parser.parse_args()

	lines,corners = pipe(args,0,50)
