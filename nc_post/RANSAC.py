import numpy as np
from tqdm import tqdm,trange
from nc_post.geom import xy2plucker, xyz2uv, side, xyz2angles
from nc_post.functions import vertical_line_corners, get_ray_angle
from nc_post.minimal_sol import HorLines_VerPlane, VerWall_CentralAprox
from nc_post.line2RaysClosestDistancePoint import * 

#RANSAC IMPLEMETATION
'''
Takes as imput the boundaries and corners from the network.
We get a first estimation of each wall separately as:
    - Ramdom selection of 3 samples on a wall
    - With the samples we define a wall: 2 horizontal lines that belong to a vertical plane
        · X = (ux,uy, vx,vy, wx,wy, d)
    - Each wall is independent of the rest

'''

def RANSAC_2LinesVerPlane(wall_bon,c_rays,f_rays,c_list,pixel_thres,P,eps,m,thres1,thres2,Rc=1.0):
	num_walls = c_list.shape[0]
	ceil_lines, floor_lines = [],[]
	central_idx = []
	RANSAC_iter_max = np.log(1-P)/np.log(1-eps**m)
	num_samples = m
	line_length = []
	walls_idx = np.arange(num_walls)
	hc_list,hf_list = [],[]
	for i in walls_idx: #,desc='RANSAC: Line Extraction'):
		best_vote = 0
		l_length = c_rays[i].shape[1]
		RANSAC_iter = int(2.5*RANSAC_iter_max)
		if l_length < pixel_thres:
			ceil_rays = c_rays[i]
			floor_rays = f_rays[i]
			central_idx.append(i)
			best_L_ceil,best_L_floor = VerWall_CentralAprox(ceil_rays,floor_rays)
		else:
			ceil_points,floor_points = [],[]
			for _ in range(RANSAC_iter):
				idx_ceil = np.random.choice(np.arange(l_length),num_samples,replace=False)
				sample_ceil = c_rays[i][:,idx_ceil]
				idx_floor = np.random.choice(l_length,num_samples,replace=False)
				sample_floor = f_rays[i][:,idx_floor]
				X,L_ceil,L_floor,h_c,h_f = HorLines_VerPlane(sample_ceil,sample_floor)
				if (X == 0).all():
					continue
				L_ceil *= np.sign(L_ceil[-1])
				L_floor *= np.sign(L_floor[-1])
				#ceil_dist,floor_dist = np.zeros(l_length),np.zeros(l_length)
				#Metric distance as first threshold
				ceil_dist1 = np.array(list(side(L_ceil,c_rays[i])))
				floor_dist1 = np.array(list(side(L_floor,f_rays[i])))
				ceil_vote1 = np.full_like(np.arange(l_length),ceil_dist1<thres1)
				floor_vote1 = np.full_like(np.arange(l_length),floor_dist1<thres1)
				if ceil_vote1.sum() == 0 or floor_vote1.sum() == 0:
					continue

				#Pixel distance as second threshold
				x_line = line2RaysClosestDistancePoint(L_ceil,c_rays[i])
				varphi1,phi1 = xyz2angles(x_line)
				varphi2,phi2 = wall_bon[i][0],wall_bon[i][1]
				ceil_dist2 = np.array(list(np.sqrt((varphi1-varphi2)**2 + (phi1-phi2)**2)))
				x_line = line2RaysClosestDistancePoint(L_floor,f_rays[i])
				varphi1,phi1 = xyz2angles(x_line)
				varphi2,phi2 = wall_bon[i][0],wall_bon[i][2]
				floor_dist2 = np.array(list(np.sqrt((varphi1-varphi2)**2 + (phi1-phi2)**2)))

				ceil_vote2 = np.full_like(ceil_vote1,ceil_dist2<=thres2)
				floor_vote2= np.full_like(floor_vote1,floor_dist2<=thres2)
				c_votes = ceil_vote1*ceil_vote2
				f_votes = floor_vote1*floor_vote2
				vote = c_votes.sum() + f_votes.sum()
				if vote > best_vote: # or floor_vote2.sum() > best_floor_vote:
					b_L_ceil = L_ceil
					b_L_floor = L_floor
					best_vote = int(vote)
					vote_points_ceil = np.where(c_votes!=0)[0]
					vote_points_floor = np.where(f_votes!=0)[0]
					ceil_points = c_rays[i][:,vote_points_ceil]
					floor_points = f_rays[i][:,vote_points_floor]

			ceil_points = list(ceil_points)
			floor_points = list(floor_points)
			hc_list += [h_c]*l_length
			hf_list += [h_f]*l_length
			if len(ceil_points) == 0 or len(floor_points)==0:
				ceil_rays = c_rays[i]
				floor_rays = f_rays[i]
				central_idx.append(i)
				best_L_ceil,best_L_floor = VerWall_CentralAprox(ceil_rays,floor_rays)
			else:
				best_L_ceil = b_L_ceil
				best_L_floor = b_L_floor
				# ceil_rays = c_rays[ceil_points]
				# floor_rays = f_rays[floor_points]

				# _,best_L_ceil,best_L_floor = HorLines_VerPlane(bon,ceil_rays,floor_rays)
		ceil_lines.append(best_L_ceil)
		floor_lines.append(best_L_floor)
		line_length.append(l_length)
	hc_median = np.median(hc_list)
	hf_median = np.median(hf_list)
	for k in central_idx:
		L_ceil = ceil_lines[k]
		L_floor = floor_lines[k]
		ceil_lines[k] = np.array([	L_ceil[0],		L_ceil[1],		0,
							-L_ceil[1]*hc_median,L_ceil[0]*hc_median,-1.0])
		floor_lines[k] = np.array([L_floor[0],		L_floor[1],		0,
							-L_floor[1]*hf_median,L_floor[0]*hf_median,-1.0])

	return ceil_lines, floor_lines, line_length
