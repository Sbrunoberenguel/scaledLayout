import numpy as np
from scipy.optimize import least_squares
import copy

from nc_post.geom import side,xyz2angles, ray2angles, angles2plucker, xy2angles
from nc_post.functions import get_ray, vertical_line_corners
from nc_post.line2RaysClosestDistancePoint import *

def optim2points(optimizable):
    h_c,h_f,points2D = optimizable[0],optimizable[1],np.array(optimizable[2:])
    points2D = points2D.reshape(-1,2)
    ceil_corners,floor_corners = np.zeros((points2D.shape[0],3)),np.zeros((points2D.shape[0],3))
    ceil_corners[:,:2] = points2D
    ceil_corners[:,2] = h_c
    floor_corners[:,:2] = points2D
    floor_corners[:,2] = h_f
    return ceil_corners,floor_corners

def points2lines(points):
    lines_dir = [points[i-1]-points[i] for i in range(points.shape[0])]
    lines_mom = [np.cross(points[i],points[i-1]) for i in range(points.shape[0])]
    lines = np.concatenate((np.array(lines_dir).reshape(-1,3),np.array(lines_mom).reshape(-1,3)),axis=1)
    lines /= np.linalg.norm(lines[:,:3],axis=1,keepdims=True)
#    lines = np.flip(lines,axis=0)
    return np.roll(lines,-1,axis=0)

def distance_func(optimizable,wall_bon,wall_crays,wall_frays,corner_angles,dist,manhattan):
    d = []
    num_walls = len(wall_crays)
    ceil_corners,floor_corners = optim2points(optimizable)
    cang,fang = np.arctan2(ceil_corners[:,1],ceil_corners[:,0]), np.arctan2(floor_corners[:,1],floor_corners[:,0])
    ceil_lines = points2lines(ceil_corners)
    floor_lines = points2lines(floor_corners)

    for i in range(num_walls):
        if manhattan:
        ##Force perpendicularity of line in manhattan environments
            d += [1000*np.dot(ceil_lines[i][:3],ceil_lines[i-1][:3])]
        else:
        ##Force corners in corner-ray direction
            d += [1000*(cang[i]-corner_angles[i])]
            d += [1000*(fang[i]-corner_angles[i])]

        c_ray = wall_crays[i].reshape(6,-1)
        f_ray = wall_frays[i].reshape(6,-1)
        if dist=='side':
            d += list(side(ceil_lines[i],c_ray))
            d += list(side(floor_lines[i],f_ray))
        else:
            x_line = line2RaysClosestDistancePoint(ceil_lines[i],c_ray)
            varphi1,phi1 = xyz2angles(x_line)
            varphi2,phi2 = wall_bon[i][0],wall_bon[i][1]
            d += list(np.sqrt((varphi1-varphi2)**2 + (phi1-phi2)**2))
            x_line = line2RaysClosestDistancePoint(floor_lines[i],f_ray)
            varphi1,phi1 = xyz2angles(x_line)
            varphi2,phi2 = wall_bon[i][0],wall_bon[i][2]
            d += list(np.sqrt((varphi1-varphi2)**2 + (phi1-phi2)**2))
    return d


def bundle_adjustment(manhattan,ceil_corners,floor_corners,wall_bon,c_list,wall_crays,wall_frays):
    points2D_0 = ceil_corners[:,:2]
    h_c_0 = ceil_corners[0,-1]
    h_f_0 = floor_corners[0,-1]
    corner_angles,_ = xy2angles(c_list,np.zeros_like(c_list))
 
    optimizable = np.zeros((points2D_0.shape[0]+1,2))
    optimizable[0,0] = h_c_0 if h_c_0 < 5 else 5
    optimizable[0,1] = h_f_0 if h_f_0 > -5 else -5
    optimizable[1:,:] = points2D_0
    op1 = np.array([optimizable.reshape(-1,)[i] for i in range(optimizable.size)])

    if np.amax(op1)>50 or np.amin(op1)<-50:
        for i in range(op1.shape[0]):
            if op1[i]>50:
                op1[i] = 49.9 - i 
            if op1[i]<-50:
                op1[i] = -49.9 + i
    
    min_bound = [0,-5] + [-50 for i in range(points2D_0.size)]
    max_bound = [5,0]+ [50 for i in range(points2D_0.size)]
    
    #Step-1: coarse optimization
    optim_var1 = least_squares(distance_func, x0=op1, args=(wall_bon,wall_crays,wall_frays,corner_angles,'side',manhattan),max_nfev = 100,
                                        bounds=(min_bound,max_bound))
    op2 = optim_var1.x

    #Step-2: fine optimization
    optim_var2 = least_squares(distance_func, x0=op2, args=(wall_bon,wall_crays,wall_frays,corner_angles,'px2px',manhattan),max_nfev=50,
                                        bounds=(min_bound,max_bound))
    var = optim_var2.x

    h_c,h_f,points2D = var[0],var[1],var[2:]
    points2D = points2D.reshape(-1,2)
    ceil_corners_out = copy.copy(ceil_corners)
    floor_corners_out = copy.copy(floor_corners)
    ceil_corners_out[:,:2] = points2D
    ceil_corners_out[:,2] = h_c

    floor_corners_out[:,:2] = points2D
    floor_corners_out[:,2] = h_f
    return ceil_corners_out, floor_corners_out, h_c,h_f
