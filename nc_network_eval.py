import os
import numpy as np
import cv2
from torch import torch
import argparse
from tqdm import tqdm,trange
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon
import glob
import json

from network.model import HorizonNet
from network.inference import augment,augment_undo,find_N_peaks
from network.dataset import visualize_a_data
from network.misc import post_proc, panostretch, utils

import warnings

warnings.simplefilter("ignore")

def inference(net, x, device, flip=False, rotate=[], visualize=False,
              force_cuboid=True, min_v=None, r=0.05):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward (with testing augmentation)
    x, aug_type = augment(x, flip, rotate)
    y_bon_, y_cor_ = net(x.to(device))
    y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
    y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0],
                                   torch.FloatTensor(y_bon_[0]),
                                   torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None

    y_bon_ = np.floor((y_bon_[0] / np.pi + 0.5) * H - 0.5)
    y_cor_ = y_cor_[0, 0]

#Own programing
    edges = y_bon_
    corners = y_cor_
    if np.amax(edges)>511:
        print('Boundary error')
        print(np.amax(edges))
        
   # return edges, corners
    
     # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    # Detech wall-wall peaks
    if min_v is None:
        min_v = 0 if force_cuboid else 0.05
    r = int(round(W * r / 2))
    N = 4 if force_cuboid else None
    xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
    if not force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
        if not Polygon(xy2d).is_valid:
            #print(
            #    'Fail to generate valid general layout!! '
            #    'Generate cuboid as fallback.',
            #    file=sys.stderr)
            xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
            cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    # Normalized to [0, 1]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H

    return cor_id, z0, z1, vis_out, edges, corners
'''	
def inference(net, x, device, flip=False, rotate=[]):
	# net   : the trained HorizonNet
	# x     : tensor in shape [1, 3, 512, 1024]
	# flip  : fliping testing augmentation
	# rotate: horizontal rotation testing augmentation

	H, W = tuple(x.shape[2:])

	# Network feedforward (with testing augmentation)
	x, aug_type = augment(x, flip, rotate)
	y_bon_, y_cor_ = net(x.to(device))
	y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
	y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

	vis_out = visualize_a_data(x[0],
							torch.FloatTensor(y_bon_[0]),
							torch.FloatTensor(y_cor_[0]))
	y_bon_ = np.floor((y_bon_[0] / np.pi + 0.5) * H - 0.5)
	y_cor_ = y_cor_[0, 0]
	#Own programing
	edges = y_bon_
	corners = y_cor_
	if np.amax(edges)>511:
		for k in range(edges.shape[1]):
			edges[0,k] = edges[0,k] if edges[0,k]<512 else 511
			edges[1,k] = edges[1,k] if edges[1,k]>=0 else 0
		print('Boundary error')
#		print(np.amax(edges))
#		input(' ')
	return vis_out,edges,corners
'''

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
		else:
			cor[0,int(x)] = 1
	cor = cor_gt_generation(cor)
	return cor

def cluster(cor):
	min_gap = 2
	min_list = [0]
	corner = []
	th = 0.7
	dCor = [cor[i-1]-cor[i] for i in range(cor.size)]
	for i in range(len(dCor)-1):
		prev = dCor[i]
		next = dCor[i+1]
		if prev<0 and next>0 and cor[i]>th:
			corner.append(i)
	return np.array(corner),np.array(dCor)

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

def inrange(l):
	for i in range(1024):
		if l[0,i] < 0:
			l[0,i]=0
			# print('Lower boundary error')
		if l[1,i] > 511:
			l[1,i] = 511
			# print('Upper boundary error')
	return l


def metrics(prediction_path,gt_path,edge_metric=True):
	th=0.8
	prediction_list = os.listdir(prediction_path)
	prediction_list.sort()
	gt_list = os.listdir(gt_path)
	gt_list.sort()
	if len(gt_list)!=len(prediction_list):
		print('ERROR: Different number of predictions an labels.')
		return 0,0,0,0
	P,Acc,R,IoU,f1 = [],[],[],[],[]
	for i in trange(len(prediction_list),desc='Computing metrics'):
		if edge_metric:
			edges = np.load(os.path.join(prediction_path,prediction_list[i]))
			edges = inrange(edges)
			img_gt = cv2.imread(os.path.join(gt_path,gt_list[i]))
			bon = bon_nc_edge(img_gt,512,1024)
			prediction = np.zeros((512,1024))
			gt = np.zeros((512,1024))
			for i in range(1024):
				gt[0:int(bon[0,i]),i] = 1
				gt[int(bon[1,i]):,i] = 1
				prediction[0:int(edges[0,i]),i] = 1
				prediction[int(edges[1,i]):,i] = 1
		else:
			prediction = np.load(os.path.join(prediction_path,prediction_list[i]))
			f = open(os.path.join(gt_path,gt_list[i])).read().split('\n')
			gt = cor_nc_txt(f)
			
		#TruePositive, TrueNegative, FalsePositive, FalseNegative
		tp = np.sum(np.logical_and(gt>th,prediction>th))
		tn = np.sum(np.logical_and(gt<=th,prediction<=th))
		fp = np.sum(np.logical_and(gt<=th,prediction<=th))
		fn = np.sum(np.logical_and(gt>th,prediction<=th))
		# How accurate the positive predictions are
		P.append(tp / (tp + fp))
		# Coverage of actual positive sample
		R.append(tp / (tp + fn))
		# Overall performance of model
		Acc.append((tp + tn) / (tp + tn + fp + fn))
		# Hybrid metric useful for unbalanced classes 
		f1.append(2 * (tp / (tp + fp))*(tp / (tp + fn))/((tp / (tp + fp))+(tp / (tp + fn))))
		# Intersection over Union
		IoU.append(tp / (tp + fp + fn))
	return np.mean(P), np.mean(R), np.mean(Acc), np.mean(IoU) #, np.mean(f1)



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--test_root', default='Caspe/',
					help='Test images and labels root directory')
	parser.add_argument('--weight','-w', default='weights/best_valid.pth',
					help='Weights to use in the network')
	parser.add_argument('--infer','-i',action='store_true',default=True,
					help='Infer the images if not have the results already')
					
	parser.add_argument('--flip', action='store_true',
					help='whether to perfome left-right flip. '
					     '# of input x2.')
	parser.add_argument('--rotate', nargs='*', default=[], type=float,
					help='whether to perfome horizontal rotate. '
					     'each elements indicate fraction of image width. '
					     '# of input xlen(rotate).')
	# Post-processing realted
	parser.add_argument('--r', default=0.05, type=float)
	parser.add_argument('--min_v', default=None, type=float)
	parser.add_argument('--force_cuboid', action='store_true')
	parser.add_argument('--visualize', action='store_true', default = True)
	args = parser.parse_args()

	img_path = os.path.join(args.test_root,'img/')
	# edge_gt_path = os.path.join(args.test_root,'EM_gt/')
	# corner_gt_path = os.path.join(args.test_root,'label_cor/')
	
	net_result_path = os.path.join(args.test_root,'results',args.weight.split('/')[-2])
	edge_result_path = os.path.join(net_result_path,'EM_pred')
	corner_result_path = os.path.join(net_result_path,'CM_pred')
	json_result_path = os.path.join(net_result_path,'json')
	raw_result_path = os.path.join(net_result_path,'raw')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net = utils.load_trained_model(HorizonNet, args.weight).to(device)
	net.eval()
	
	img_name = os.listdir(img_path)
	img_name.sort()
	
	if not os.path.isdir(edge_result_path):
		os.makedirs(edge_result_path)
	if not os.path.isdir(corner_result_path):
		os.makedirs(corner_result_path)
	if not os.path.isdir(json_result_path):
		os.makedirs(json_result_path)
	if not os.path.isdir(raw_result_path):
		os.makedirs(raw_result_path)
	# if args.infer:
		# with torch.no_grad():	
		# 	for i in trange(len(img_name), desc='Image test inferencing'):
		# 		# Load image
		# 		img_pil = Image.open(os.path.join(img_path,img_name[i]))
	if args.infer:
		edge_mask = np.zeros((2,1024))
		edge_mask[:,256:256+512] = 1
		corner_mask = np.zeros((1,1024))
		corner_mask[:,256:256+512] = 1
		# Inferencing
		with torch.no_grad():
			for i in trange(len(img_name), desc='Image test inferencing'):
				# k = os.path.split(i_path)[-1][:-4]

				# Load image
				# img_pil = Image.open(i_path)
				# if img_pil.size != (1024, 512):
				#     img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
				img_or = cv2.imread(os.path.join(img_path,img_name[i]))
				#img_rot = np.roll(img_or,512,axis=1)
				img_or = cv2.resize(img_or,(1024,512))

				img_ori = np.array(img_or)[..., :3].transpose([2, 0, 1]).copy()
				x_or = torch.FloatTensor([img_ori / 255])

				# img_r = np.array(img_rot)[..., :3].transpose([2, 0, 1]).copy()
				# x_rot = torch.FloatTensor([img_r / 255])

				# Inferenceing corners
				# 
				_, _, _, vis_out, edges, corners = inference(net, x_or, device,
													args.flip, args.rotate,
													args.visualize,
													args.force_cuboid,
													args.min_v, args.r)
				# _, _, _, _, edges_rot, corners_rot = inference(net, x_rot, device,
				# 									args.flip, args.rotate,
				# 									args.visualize,
				# 									args.force_cuboid,
				# 									args.min_v, args.r)

				# # Output result
				'''
				with open(os.path.join(args.output_dir, k + '.json'), 'w') as f:
					json.dump({
						'z0': float(z0),
						'z1': float(z1),
						'uv': [[float(u), float(v)] for u, v in cor_id],
					}, f)
				'''

				# edge_orM = edge_mask * edges_or
				# edge_rotM =edge_mask * edges_rot
				# edges = edge_orM + np.roll(edge_rotM,512,axis=1)

				# corner_orM = corner_mask * corners_or
				# corner_rotM =corner_mask * corners_rot
				# corners = corner_orM + np.roll(corner_rotM,512,axis=1)
				if args.visualize:
					vis_path = os.path.join(raw_result_path,img_name[i][:-4] + '.raw.png')
					vh, vw = vis_out.shape[:2]
					Image.fromarray(vis_out)\
							.resize((vw//2, vh//2), Image.LANCZOS)\
							.save(vis_path)
				np.save(os.path.join(edge_result_path,img_name[i][:-4]+'_edges.npy'), edges)
				np.save(os.path.join(corner_result_path,img_name[i][:-4]+'_corners.npy'),corners)
'''
	P_e, R_e, Acc_e, IoU_e = metrics(edge_result_path,edge_gt_path,edge_metric=True)
	P_c, R_c, Acc_c, IoU_c = metrics(corner_result_path,corner_gt_path,edge_metric=False)
	
	try:
		print('EDGES: Precision: ' + str('%.3f' % P_e) + '; Accuracy: ' + str('%.3f' % Acc_e) + '; Recall: ' + str('%.3f' % R_e) + '; IoU: ' + str('%.3f' % IoU_e)) # + '; F1 Score: ' + str('%.3f' %f1_e))
		
		print('CORNERS: Precision: ' + str('%.3f' % P_c) + '; Accuracy: ' + str('%.3f' % Acc_c) + '; Recall: ' + str('%.3f' % R_c) + '; IoU: ' + str('%.3f' % IoU_c)) # + '; F1 Score: ' + str('%.3f' %f1_c))
	
	except:
		print('Metrics Error')
'''
	

