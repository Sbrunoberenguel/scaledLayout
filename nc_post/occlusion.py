import numpy as np
import copy
from nc_post.geom import closest_point,side, angles2plucker
from nc_post.functions import get_ray, are_parallel, vertical_line_corners

def occlusion_manhattan(L_VP,ceil_lines,floor_lines,c_list,wall_bon,wall_crays,wall_frays,bon,c_rays,f_rays):
	occlusion_found = False
	num_walls = len(wall_bon)
	ceil_lines_out = copy.copy(ceil_lines)
	floor_lines_out = copy.copy(floor_lines)
	new_c_list = copy.copy(c_list)
	if num_walls%2 != 0:
		for i in range(num_walls):
			cline = ceil_lines[i]
			cline *= np.sign(cline[0])
			fline = floor_lines[i]
			
			prev_line = ceil_lines[i-1]
			prev_line*= np.sign(prev_line[0])
			# 
			# _,pline,_ = vertical_line_corners(cline,cray)
			# _,prev,_ = vertical_line_corners(prev_line,cray)
			# p1 = pline[:3]/pline[3]
			# p2 = prev[:3]/prev[3]
			# dist = np.linalg.norm(p1-p2)
			if np.isclose(cline[:3],prev_line[:3],atol=0.1).all():
				#print('Manhattan occlusion managed')
				occlusion_found = True
				#Occlusion found
				cray,fray = c_rays[c_list[i]],f_rays[c_list[i]]
				_,pc,_ = vertical_line_corners(cline,cray)
				pc = pc[:3]/pc[3]
				_,pf,_ = vertical_line_corners(fline,fray)
				pf = pf[:3]/pf[3]
				l_dir = L_VP[0] if np.isclose(L_VP[1],cline[:3],atol=0.1).all() else L_VP[1]
				c_mom = np.cross(pc,l_dir)
				f_mom = np.cross(pf,l_dir)
				new_ceil_line = np.concatenate((l_dir,c_mom)).reshape(-1,)
				new_floor_line = np.concatenate((l_dir,f_mom)).reshape(-1,)
				wall_bon.insert(i,bon[:,c_list[i]])
				wall_crays.insert(i,cray)
				wall_frays.insert(i,fray)
				ceil_lines_out = np.insert(ceil_lines_out,i,new_ceil_line,axis=0)
				floor_lines_out = np.insert(floor_lines_out,i,new_floor_line,axis=0)
				new_c_list = np.insert(new_c_list,i,c_list[i])
				
	return ceil_lines_out,floor_lines_out,new_c_list,wall_bon,wall_crays,wall_frays,occlusion_found

def occlusion_atlanta(ceil_lines,floor_lines,c_list,wall_bon,wall_crays,wall_frays,bon,c_rays,f_rays):
	num_walls = len(wall_bon)
	ceil_lines_out = copy.copy(ceil_lines)
	floor_lines_out = copy.copy(floor_lines)
	new_c_list = copy.copy(c_list)
	d=[]
	print('Atlanta occlusion finder')
	for i in range(num_walls):
		cray = angles2plucker(bon[0,c_list[i]],bon[1,c_list[i]])
		cline = ceil_lines[i]
		prev_line = ceil_lines[i-1]
		_,pline,_ = vertical_line_corners(cline,cray)
		_,prev,_ = vertical_line_corners(prev_line,cray)
		p1 = pline[:3]/pline[3]
		p2 = prev[:3]/prev[3]
		dist = np.linalg.norm(p1-p2)
		d.append(dist)
		
		# if np.isclose(cline[:3],prev_line[:3],atol=0.1).all():
		# 	#Occlusion found
		# 	cray,fray = c_rays[c_list[i]],f_rays[c_list[i]]
		# 	_,pc,_ = vertical_line_corners(cline,cray)
		# 	pc = pc[:3]/pc[3]
		# 	_,pf,_ = vertical_line_corners(fline,fray)
		# 	pf = pf[:3]/pf[3]
		# 	l_dir = L_VP[0] if np.isclose(L_VP[1],cline[:3],atol=0.1).all() else L_VP[1]
		# 	c_mom = np.cross(pc,l_dir)
		# 	f_mom = np.cross(pf,l_dir)
		# 	new_ceil_line = np.concatenate((l_dir,c_mom)).reshape(-1,)
		# 	new_floor_line = np.concatenate((l_dir,f_mom)).reshape(-1,)
		# 	wall_bon.insert(i,bon[:,c_list[i]])
		# 	wall_crays.insert(i,cray)
		# 	wall_frays.insert(i,fray)
		# 	ceil_lines_out = np.insert(ceil_lines_out,i,new_ceil_line,axis=0)
		# 	floor_lines_out = np.insert(floor_lines_out,i,new_floor_line,axis=0)
		# 	new_c_list = np.insert(new_c_list,i,c_list[i]-1)
				
	return ceil_lines_out,floor_lines_out,new_c_list,wall_bon,wall_crays,wall_frays

#OCCLUSION MANAGEMENT
def occlusion_management_old(bon,num_walls,c_list,ceil_lines,floor_lines,Rc=1.0):
	new_walls = 0
	ceil_lines_out = copy.copy(ceil_lines)
	floor_lines_out = copy.copy(floor_lines)
	new_c_list = copy.copy(c_list)
	if (num_walls%2 != 0):
		for i in range(num_walls): #,desc='Manage occlusions'):
			idx = i
			idx_next = 0 if i+1 >= num_walls else i+1
			ceil_L = ceil_lines[idx]
			ceil_M = ceil_lines[idx_next]
			if are_parallel(ceil_L,ceil_M):
				new_walls += 1
				depth_L = np.linalg.norm(ceil_L[3:])
				depth_M = np.linalg.norm(ceil_M[3:])
				occluder = idx_next if depth_M < depth_L else idx
				direction = ceil_lines[occluder][:3]
				direction = direction*(-1) if occluder==idx_next else direction
				l1,l2,_ = direction
				ceil_chi_a,_ = get_ray(bon, idx = c_list[occluder],Rc=Rc)
				_,ceil_C,_ = vertical_line_corners(ceil_lines[occluder],ceil_chi_a)
				ceil_closest = closest_point(ceil_lines[occluder])
				floor_closest = closest_point(floor_lines[occluder])
				h_c = ceil_closest[2]
				h_f = floor_closest[2]
				ceil_corner = ceil_C[:3]/ceil_C[3]
				depth = np.linalg.norm(ceil_corner-ceil_closest)
				
				new_line_l = np.array([l2,-l1,0])
				
				new_ceil_l_b = np.array([l1*h_c,
										 l2*h_c,
										 -depth])
				new_floor_l_b= np.array([l1*h_f,
										 l2*h_f,
										 -depth])

				new_ceil_line = np.concatenate((new_line_l,new_ceil_l_b),axis=0)
				new_floor_line = np.concatenate((new_line_l,new_floor_l_b),axis=0)
				
				ceil_lines_out = np.insert(ceil_lines_out,idx_next,new_ceil_line,axis=0)
				floor_lines_out = np.insert(floor_lines_out,idx_next,new_floor_line,axis=0)
				new_c_list = np.insert(new_c_list,idx_next,c_list[idx_next]-1)
		if new_walls == 0:
			print('No occlusions found')
	#else:
	#	print('No occlusion management')

	return ceil_lines_out, floor_lines_out, new_c_list, new_walls