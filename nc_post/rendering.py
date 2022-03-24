import numpy as np
import math as m
from cv2 import cv2
import open3d as op
import matplotlib.pyplot as plt
from nc_post.geom import xyz2uv, xy2plucker, xy2angles, closest_point
from nc_post.functions import vertical_line_corners


#IMAGE DRAWING AND SCENE RECONSTRUCTION


def test(bon,c_list):
	x = np.arange(1024)
	fig,ax = plt.subplots()
	ax.plot(x,512-bon[0],'b')
	ax.plot(x,512-bon[1],'g')
	for i in range(c_list.size):
		ax.axvline(c_list[i],0,512,linestyle='solid',color='r')
	ax.set_xlim([0,1024])
	ax.set_ylim([0,512])
	fig.savefig('test.png')
	plt.close()

#Draws corner probability from the output of the Network
def draw_corner_probability(cor,dcor,c_list,path):
	x1 = np.arange(1024)
	x2 = np.arange(len(dcor))
	fig,(ax1,ax2) = plt.subplots(2,1)
	cor = np.flip(cor).reshape(-1,)
	dcor = np.flip(dcor)
	c_list = 1024 - c_list
	c_list.sort()
	ax1.plot(x1,cor,'b')
	ax2.plot(x2,dcor,'g')
	ax1.set(ylabel='Corner Probability')
	ax2.set(ylabel='First derivate')
	ax1.grid()
	for i in range(c_list.size):
		y = c_list[i]
		ax1.axvline(y,-1,1,linestyle='solid',color='r')
		ax2.axvline(y,-1,1,linestyle='solid',color='r')
	ax1.set_xlim([0,1024])
	ax2.set_xlim([0,1024])
	ax1.set_ylim([0,1])
	ax2.set_ylim([-1,1])
	fig.savefig('{}/corner_probability.png'.format(path))
	plt.close()

#Draws the lines that connect the intersection points of the corners
#Color domain in BGR=[0,255]
#PTE CORRECTING THE FUNCTION
def draw_corners(img_name,path,c_lines,f_lines,w_lines,Rc=1.0):
	img = cv2.imread(img_name)
	img = cv2.flip(img,1)
	#img = np.roll(img,512,axis=1)
	ceil_color = np.array([255,0,0])
	floor_color = np.array([0,0,255])
	wall_color = np.array([255,255,255])
	for i in range(c_lines.shape[0]):
		u,v = xyz2uv(c_lines[i],Rc=Rc)
		img[m.floor(v),m.floor(u)] = ceil_color
	for i in range(f_lines.shape[0]):
		u,v = xyz2uv(f_lines[i],Rc=Rc)
		v = v if v<512 else 511
		img[m.floor(v),m.floor(u)] = floor_color
	for i in range(w_lines.shape[0]):
		u,v = xyz2uv(w_lines[i],Rc=Rc)
		v = v if v<512 else 511
		img[m.floor(v),m.floor(u)] = wall_color
	img = cv2.flip(img,1)
	cv2.imwrite(path+'_layout.png',img)

#Draws the lines from the director vector and closest point to origin
#Color domain in BGR=[0,255]
#PTE CORRECTING THE FUNCTION
def draw_lines(bon,cor,img_name,name,path,c_lines,f_lines, Rc=1.0):
	img = cv2.imread(img_name)
	img = cv2.flip(img,1)
	ceil_color = np.array([255,0,0])
	floor_color = np.array([0,0,255])
	idx = -1
	for i in range(bon.shape[1]):
		if i in cor:
			idx+=1
		c_ray = xy2plucker(i,bon[0,i],Rc)
		f_ray = xy2plucker(i,bon[1,i],Rc)
		_,_,cp = vertical_line_corners(c_ray,c_lines[idx])
		_,_,fp = vertical_line_corners(f_ray,f_lines[idx])
		uc,vc = xyz2uv(cp[:3]/cp[3],Rc=Rc)
		uf,vf = xyz2uv(fp[:3]/fp[3],Rc=Rc)
		img[vc,uc] = ceil_color
		img[vf,uf] = floor_color
	img = cv2.flip(img,1)
	cv2.imwrite(path+name+'_boundaries.png',img)

#Reconstrucs 3D layout from the director vector and closest point to origin
#Color domain in RGB=[0,1]
def layout_3D_lines(img_name,path,c_lines,f_lines):
	c_point = closest_point(c_lines[0])
	f_point = closest_point(f_lines[0])
	ceil = extrap(c_point,c_lines[0])
	floor = extrap(f_point,f_lines[0])
	origin = op.geometry.PointCloud()
	origin.points = op.utility.Vector3dVector(np.array([0,0,0]).reshape(1,3))
	for i in np.arange(1,c_lines.shape[0]):
		c_point = closest_point(c_lines[i])
		f_point = closest_point(f_lines[i])
		ceil_list = extrap(c_point,c_lines[i])
		floor_list = extrap(f_point,f_lines[i])
		ceil = np.concatenate((ceil,ceil_list),axis=0)
		floor = np.concatenate((floor,floor_list),axis=0)
	pc_ceil = op.geometry.PointCloud()
	pc_floor= op.geometry.PointCloud()
	pc_ceil.points = op.utility.Vector3dVector(ceil)
	pc_floor.points= op.utility.Vector3dVector(floor)
	pc_ceil.paint_uniform_color([0,0,1])
	pc_floor.paint_uniform_color([1,0,0])
	origin.paint_uniform_color([0,0,0])
	layout = pc_ceil+pc_floor+origin
	op.io.write_point_cloud(path+'/'+img_name+'_3Dlayout.pcd',layout)
	return layout,ceil,floor

#Reconstrucs 3D layout from intersection points of corners
#Color domain in RGB=[0,1]
def layout_3D_corners(ceil_C,floor_C):
	p_ceiling = np.array([ceil_C[0][:3]])
	p_walls = np.array([ceil_C[0][:3]])
	p_floor = np.array([floor_C[0][:3]])
	ceiling = op.geometry.PointCloud()
	walls = op.geometry.PointCloud()
	floor = op.geometry.PointCloud()
	num_corners = ceil_C.shape[0]
	for i in range(num_corners):
		idx = i
		idx_next = i+1 if i+1<num_corners else i+1-num_corners
	#Build ceiling
		pc_ini = ceil_C[idx][:3]
		pc_end = ceil_C[idx_next][:3]
		l_ceil = np.linspace(pc_ini,pc_end,num=200)
		p_ceiling = np.concatenate((p_ceiling,l_ceil),axis=0)	
	#Build floor
		pf_ini = floor_C[idx][:3]
		pf_end = floor_C[idx_next][:3]
		l_floor = np.linspace(pf_ini,pf_end,num=200)
		p_floor = np.concatenate((p_floor,l_floor),axis=0)	
	#Build walls		
		l_wall = np.linspace(pc_ini,pf_ini,num=100)
		p_walls = np.concatenate((p_walls,l_wall),axis=0)	

	ceiling.points = op.utility.Vector3dVector(p_ceiling)
	ceiling.paint_uniform_color([0,0,1])	
	floor.points = op.utility.Vector3dVector(p_floor)
	floor.paint_uniform_color([1,0,0])		
	walls.points = op.utility.Vector3dVector(p_walls)
	walls.paint_uniform_color([0,0,0])	
	
	origin = op.geometry.PointCloud()
	origin.points = op.utility.Vector3dVector(np.array([0,0,0]).reshape(1,3))
	origin.paint_uniform_color([0,0,0])
	
	PointCloud = ceiling+walls+floor+origin
	return PointCloud,p_ceiling,p_floor,p_walls

#Extrapolates a point with a given vector up to 100 points
def extrap(point,vector):
	p_list = np.zeros((200,3))
	v = normalize_vector(vector[:3])
	for i in range(100):
		p_list[i+100] = point + v*(i/20.0)
		p_list[99-i] = point - v*(i/20.0)
	return p_list

def normalize_vector(v):
	norm = np.linalg.norm(v)
	return v/norm
	
def layout_reconstruction(ceil_C,floor_C,bon,c_list,h_ceil,h_floor,Rc=1.0):
	walls = np.zeros((512,1024,3))
	ceiling = np.zeros((512,1024,3))
	floor = np.zeros((512,1024,3))
	W = 1024
	x = np.arange(W)
	varPhi_init = -np.pi
	varPhi_end = np.pi
	varPhi = x*(varPhi_end-varPhi_init)/W + varPhi_init
	cir = np.array([np.cos(varPhi),np.sin(varPhi),np.zeros(W)]).T
	for i in range(c_list.shape[0]):
		corner = c_list[i]
		ceil_corner = int(bon[0][corner])
		floor_corner = int(bon[1][corner])
		ceil_c = ceil_C[i]
		floor_c = floor_C[i]
		walls[ceil_corner:floor_corner,corner,:] = np.linspace(ceil_c,
															floor_c,
															floor_corner-ceil_corner)
		i_next = i+1
		if i_next >= c_list.shape[0]:
			i_next -= c_list.shape[0]
		next_corner = c_list[i_next]
		ceil_c_next = ceil_C[i_next]
		floor_c_next = floor_C[i_next]

		if next_corner < corner:
			next_corner += 1024		
		ceil_line = np.linspace(ceil_c,ceil_c_next,np.absolute(next_corner-corner))
		floor_line = np.linspace(floor_c,floor_c_next,np.absolute(next_corner-corner))
		k = 0
		for j in np.arange(corner,next_corner,1):
			if j >= 1024:
				j-= 1024
			origin = cir[j]*Rc
			up_origin = (origin+np.array([0,0,h_ceil]))
			down_origin = (origin+np.array([0,0,h_floor]))

			column = np.linspace(ceil_line[k],floor_line[k],int(bon[1][j])-int(bon[0][j]))
			img_floor = np.linspace(floor_line[k],down_origin,512-int(bon[1][j])-1)
			img_ceil = np.linspace(up_origin,ceil_line[k],abs(int(bon[0][j]-1)))
			walls[int(bon[0][j]),j,:] = ceil_line[k]
			walls[int(bon[1][j]),j,:] = floor_line[k]
			walls[int(bon[0][j]):int(bon[1][j]),j,:] = column
			ceiling[:int(bon[0][j])-1,j,:] = img_ceil
			floor[int(bon[1][j])+1:,j,:] = img_floor
			k += 1
	return walls,ceiling,floor
