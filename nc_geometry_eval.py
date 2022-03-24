import numpy as np
import os
from cv2 import cv2
import nc_post.PipeLineOr as pp
import matplotlib.pyplot as plt
import open3d as op
import argparse
from tqdm import tqdm,trange
from shapely.geometry import Polygon
import warnings
import copy
import scipy.signal as ss

# warnings.simplefilter("ignore")
def cluster_peaks(cor):
	delete = True
	cor = cor.reshape(-1,)
	dCor = [cor[i-1]-cor[i] for i in range(cor.size)]
	th = 0.8
	min_gap = 3
	peaks = ss.find_peaks(cor,height=th)
	c_list = peaks[0]
	while delete:
		gaps = c_list-np.roll(c_list,1)
		if min(gaps,key=abs)<min_gap:
			idx = int(np.where(gaps==min(gaps,key=abs))[0])
			pk1 = peaks[1]['peak_heights'][idx]
			pk2 = peaks[1]['peak_heights'][idx-1]
			if pk1 > pk2:
				c_list = np.delete(c_list,idx-1)
			else:
				c_list = np.delete(c_list,idx)
		else:
			delete=False
	return np.array(c_list),np.array(dCor)*10

def bon_nc_edge(edge, H, W):
	#edge = cv2.dilate(edge, np.array([[1,1]]))
	bon = np.zeros((2,W))
	for i in range(W):
		for j in range(H):
			if (edge[j,i] >= 250).all():
				bon[0,i] = j
				break
		for j in np.arange(H-1,0,-1):
			if (edge[j,i] >= 250).all():
				bon[1,i] = j
				break
	#bon = ((bon + 0.5) / H - 0.5) * np.pi
	return bon

def cor_nc_txt(f):
	cor = np.zeros((1,1024))
	x_prev = 0
	for i in range(len(f)-1):
		x,y = f[i].split(' ')
		if int(x)==x_prev:
			continue
		elif int(x) >=1024:
			x = 1023
		cor[0,int(x)] = 1
	cor = cor_gt_generation(cor)
	return cor

def cor_gt_generation(cor):
	p_base = 0.96
	# Prepare 1d wall-wall probability
	candidates = np.where(cor[0]==1)[0]
	candidates = np.concatenate((candidates,[candidates[0]+1024]))
	dist = np.full_like(cor,0)
	for i in range(dist.shape[1]):
		dist[0,i] = min(abs(candidates-i))
	y_cor = (p_base ** dist)
	return y_cor

def metrics(gt_corners,pred_corners,camera,pov,up2scale):
	#gt_corners *= [-1,-1,1]
	num_pred_corners = int(pred_corners.shape[0]/2)
	num_gt_corners = int(gt_corners.shape[0]/2)
	camera = camera.reshape(1,-1)
	if up2scale:
		#Normalizar con la altura
		hf_pred = abs(pred_corners[-1,2])
		hf_gt = abs(gt_corners[-1,2])#camera[0,2]
		pred_corners += camera
		pred_corners = pred_corners/hf_pred
		gt_corners = gt_corners/hf_gt
		#normalizar con la diagonal
		xpm,xpMa = np.amin(pred_corners[:,0]),np.amax(pred_corners[:,0])
		ypm,ypMa = np.amin(pred_corners[:,1]),np.amax(pred_corners[:,1])
		zpm,zpMa = np.amin(pred_corners[:,2]),np.amax(pred_corners[:,2])
		pred_norm = np.sqrt((xpMa-xpm)**2 + (ypMa-ypm)**2 + (zpMa-zpm)**2)
		xgm,xgMa = np.amin(gt_corners[:,0]),np.amax(gt_corners[:,0])
		ygm,ygMa = np.amin(gt_corners[:,1]),np.amax(gt_corners[:,1])
		zgm,zgMa = np.amin(gt_corners[:,2]),np.amax(gt_corners[:,2])
		gt_norm = np.sqrt((xgMa-xgm)**2 + (ygMa-ygm)**2 + (zgMa-zgm)**2)
		pred_corners = pred_corners/pred_norm
		gt_corners = gt_corners/gt_norm
	else:
		pred_corners += camera
	pred_ceil,pred_floor = pred_corners[:num_pred_corners],pred_corners[num_pred_corners:]
	gt_ceil,gt_floor = gt_corners[:num_gt_corners],gt_corners[num_gt_corners:]
	#height error
	pred_height = pred_ceil[0,2]-pred_floor[0,2]
	gt_height = gt_ceil[0,2]-gt_floor[0,2]
	height_error = abs(pred_height-gt_height)
	#3D IoU
	try:
		pred_floorplan = Polygon(pred_floor[:,:2])
		pred_area = pred_floorplan.area
		gt_floorplan = Polygon(gt_floor[:,:2])
		gt_area = gt_floorplan.area
		inter_floorplan = pred_floorplan.intersection(gt_floorplan)
		inter_area = inter_floorplan.area
		pred_volume = pred_area * pred_height
		gt_volume = gt_area * gt_height
		inter_height = min(gt_ceil[0,-1],pred_ceil[0,-1]) - max(gt_floor[0,-1],pred_floor[0,-1])
		inter_volume = inter_area * inter_height 
		IoU3D = inter_volume / (pred_volume+gt_volume-inter_volume)
		# if pred_area <2.0 and not up2scale:
		# 	return 0,0,0
	except:
		# return 0,0,0
		IoU3D = 0
	#Corner error metric
	dc,df = 0,0
	for i in range(num_pred_corners):
		dc += min(np.linalg.norm(pred_ceil[i]-gt_ceil,axis=1,keepdims=True))
		df += min(np.linalg.norm(pred_floor[i]-gt_floor,axis=1,keepdims=True))
		#norm = np.linalg.norm(pred_ceil[i]-pred_floor,axis=1)
	ceil_distance = dc/num_pred_corners
	floor_distance = df/num_pred_corners
	distance = float(ceil_distance+floor_distance)/2.0
	if distance > 10:
		distance = 10
	return distance, height_error, IoU3D


if __name__ == '__main__':
	#Auxiliar hard-coded parameters
	label=False
	process = True
	up2scale = False
	see_each = 700
	d_max = 1
	#Directories
	#test_root = 'Test_sets/test_AW/no_central'
	# test_root = 'Test_sets/test_MW'
	test_root = 'caspe'
	edges_info = 'EM_gt'
	corners_info = 'label_cor'
	img_info = 'img'
	network = 'net_pred'

	D3_gt_info = '3D_gt'
	img_path = os.path.join(test_root,img_info)
	img_list = os.listdir(img_path)
	img_list.sort()
	# edges_path = os.path.join(test_root,edges_info)
	# edges_list = os.listdir(edges_path)
	# edges_list.sort()
	# corners_path = os.path.join(test_root,corners_info)
	# corners_list = os.listdir(corners_path)
	# corners_list.sort()
	result_path = os.path.join(test_root,network)
	# D3_path = os.path.join(test_root,D3_gt_info)
	# D3_list = os.listdir(D3_path)
	# D3_list.sort()

	# WL_path = os.path.join(test_root,'WallLenght')

	if not os.path.isdir(result_path):
		os.makedirs(result_path)
	
	if label:
		for i in trange(len(edges_list),desc='Computing network output as GT'):
			img_gt = cv2.imread(os.path.join(edges_path,edges_list[i]))
			if 'M22' in edges_list[i]:
				img_gt = cv2.dilate(img_gt,np.array([[1,1]]))
			bon = bon_nc_edge(img_gt,512,1024)
			f = open(os.path.join(corners_path,corners_list[i])).read().split('\n')
			cor = cor_nc_txt(f)
			np.save(os.path.join(result_path,corners_list[i][:-4]+'_corners.npy'),cor.reshape(-1,))

	CE = []
	HE = []
	IoU3D = []
	count = 0
	out = 0
	# quitar = ['AFimg1056','AFimg1006','AFimg0908','AFimg0774','AFimg0789','AFimg0728','AFimg0824']
	for i in trange(len(img_list),desc='Computing geometrical processing and metrics'):
		# i+=228
		data_default = os.path.join(result_path,img_list[i][:-4])
		img_default = os.path.join(img_path,img_list[i])
		# print(img_default)
		# if img_list[i][:-4] in quitar:
			# continue

		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument('--data', default = data_default, help='Input data to post-process')
		parser.add_argument('--img', default = img_default, help='Input image')
		parser.add_argument('--visualize', action='store_true', default=False,help='Visualize 3D layout lines after processing')
		parser.add_argument('--Manhattan', action='store_true', default=True,help='Force Manhattan world assumption')
		parser.add_argument('--num_dir', default='6', type=int, help='Set numer of wall directions on Atlanta world assumption')
		parser.add_argument('--ceiling', action='store_true', default=False,help='Includes ceiling in 3D room reconstruction')
		parser.add_argument('--Rc', default='1.0', type=float)
		parser.add_argument('--save', action='store_true',default=True,help='Store intermediate solutions')
		#---------------------------------------------------------------------------
		parser.add_argument('--ran', action='store_false',default=True,help='RANSAC approach')
		parser.add_argument('--dir', action='store_true', default=True,help='Manhattan direction optimization')
		parser.add_argument('--occ', action='store_true', default=False,help='Occlusion management')
		parser.add_argument('--DLT', action='store_false',default=True,help='Global DLT')
		parser.add_argument('--pcd', action='store_false',default=True,help='Room 3D visual reconstruction')
		parser.add_argument('--test',action='store_true',default=False,help='Exits on test')
		parser.add_argument('--fa',  action='store_true',default=False,help='Uses Final adjustment at the end')	
		args = parser.parse_args()
		args.data = data_default
		args.img = img_default
		success = 5
		#gt_corners = np.load(os.path.join(D3_path,img_list[i][:-3]+'npy'))
		# num_cor = gt_corners.shape[0]//2
		
		# cor = np.load(data_default+'_corners.npy')
		# cor = np.flip(cor)
		# c_list,dCor = cluster_peaks(cor)

		# if c_list.shape[0] != num_cor:
		# 	continue

		if process:
			pred_ceil,pred_floor = pp.pipe(args,0,50)
			# It should be in the first attempt but.... shit happens
			try:
				pred_floorplan = Polygon(pred_floor[:,:2])
			except:
				print('Holy S**t: '+img_list[i])
				pred_ceil,pred_floor = pp.pipe(args,0,50)
		else:
			pred_ceil = np.load(os.path.join(data_default,'ceil_corners.npy'))
			pred_floor = np.load(os.path.join(data_default,'floor_corners.npy'))
	'''
		pred_corners = np.concatenate((pred_ceil,pred_floor),axis=0)

		# if pred_corners.shape[0] != gt_corners.shape[0]:
		# 	continue

		camera = np.load(os.path.join(test_root,'cam_pose',img_list[i][:-3]+'npy'))
		pov = True if img_list[i][:3]=='img' else False
		distance,height,iou = metrics(gt_corners,pred_corners,camera,pov,up2scale)
		if iou == 0:
			print(img_default)
		#metrics
		if up2scale:
			CE.append(distance*100)
			HE.append(height*100)
			IoU3D.append(iou)
		else:
			CE.append(distance)
			HE.append(height)
			IoU3D.append(iou)
		if (i+1)%see_each == 0:
			corner = np.array(CE)
			room = np.array(HE)
			inter = np.array(IoU3D)
			print('\nCE mean after %.1u interations: %.4f' %(i+1,corner.mean()))
			print('HE mean after %.1u interations: %.4f' %(i+1,room.mean()))
			print('3D IoU mean after %.1u interations: %.3f \n' %(i+1,inter.mean()))

	CE = np.array(CE)
	HE = np.array(HE)
	IoU = np.array(IoU3D)

	np.save('CE_errors.npy',CE)
	np.save('IoU_errors.npy',IoU)

	CE_mean = CE.mean()
	HE_mean = HE.mean()
	IoU_mean = IoU.mean()
	CE_median = np.median(CE)
	HE_median = np.median(HE)
	IoU_median = np.median(IoU)
	CE_std = np.std(CE)
	HE_std = np.std(HE)
	IoU_std = np.std(IoU)

	print('\n')
	print(np.amax(CE))
	print(np.amin(IoU))
#	print('Counted: ',count,'\n Out: ',out)
	print('#####')

	print('Results from: '+network)
	print('Computed %.1u images for metrics' %len(CE))
	if up2scale:
		print('UP TO SCALE RESULTS \n')
	print('Mean: Corner Error, Euclidean distance in meters: %.4f \u00B1 %.4f' %(CE_mean,CE_std))
	print('Mean: Room Height Error in meters: %.4f \u00B1 %.4f' %(HE_mean,HE_std))
	print('Mean: 3D Intersection over Union: %.4f \u00B1 %.4f' %(IoU_mean*100,IoU_std*100))
	print('-------------------------------------------------------')
	print('Median: Corner Error, Euclidean distance in meters: %.4f' %CE_median)
	print('Median: Room Height Error in meters: %.4f' %HE_median)
	print('Median: 3D Intersection over Union: %.4f' %(IoU_median*100))
	print('-------------------------------------------------------')
	'''