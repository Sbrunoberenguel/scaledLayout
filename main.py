'''
Main program to obtain the 3D scaled layout from a single
    non-central panorama.

'''

import os
import cv2
import copy
import torch
import argparse
import numpy as np
import open3d as op
from tqdm import tqdm
import scipy.io as sio

import network
from network import model
from network import inference
from network.misc import utils

import nc_geometry

def get_input_image(path):
    img = cv2.imread(path)
    x = np.array(img)[..., :3].transpose([2, 0, 1]).copy()
    x = torch.FloatTensor([x / 255])
    x = torch.nn.functional.interpolate(x,(512,1024),align_corners=True, mode='bilinear')
    return x,img

def set_arguments(args):
    #Default setings for Manhattan environments
    if args.Manhattan:
        args.ran = True
        args.dir = True
        args.num_dir = 6
        args.occ = True
        args.DLT = True
        args.fa = False
    #Default setings for Atlanta environments
    if not args.Manhattan:
        args.ran = True
        args.dir = False
        args.num_dir = 6
        args.occ = False
        args.DLT = True
        args.fa = True
    return args

def main(args):
    network = utils.load_trained_model(model.HorizonNet, args.pth)
    network.eval()
    network.to(args.device)

    input_list = os.listdir(args.input_path)

    for name in tqdm(input_list):
        if name[:2] != 'AF':
            Rc = int(name.split('R')[1][:5])
            args.Rc = Rc/10000.0
        args = set_arguments(args)
        path = os.path.join(args.input_path,name)
        try:
            x,img = get_input_image(path)
        except:
            print('No image found or cannot be read')
            continue
        os.makedirs(os.path.join(args.out_dir,name[:-4]),exist_ok=True)
        with torch.no_grad():
            edges,corners = inference.inference(network, x, args.device)
        if args.save_raw:
            np.save(os.path.join(args.out_dir,name[:-4],name[:-4]+'net_edg.npy'),edges)
            np.save(os.path.join(args.out_dir,name[:-4],name[:-4]+'net_cor.npy'),corners)
        
        Corners3D, Lines3D, PC, PC_c = nc_geometry.pipe(args,edges,corners,name,img)
        data = {'CeilCorners': Corners3D[0],
                'FloorCorners':Corners3D[1],
                'CeilLines': Lines3D[0],
                'FloorLines':Lines3D[1],
                'RoomReconstruction':
                    {'points': np.asarray(PC_c.points),
                    'Colors': np.asarray(PC_c.colors)}}
        sio.savemat(os.path.join(args.out_dir,name[:-4],name[:-3]+'mat'), data)

        if args.visualize:
            op.visualization.draw_geometries([PC],window_name=name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', required=True, help='Input data')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--pth', required=True, help='Network weigths')
    parser.add_argument('--visualize', action='store_true', help='Visualize 3D layout lines after processing')
    parser.add_argument('--Manhattan', action='store_true', help='Force Manhattan world assumption')
    parser.add_argument('--Rc', default='1.0', type=float, help='Radious of non-central acquisition system (aka. calibration)')
    parser.add_argument('--save_raw', action='store_true',default=False,help='Store intermediate solutions')
    parser.add_argument('--no_cuda', action='store_true', help='Disable use of GPU')
    args = parser.parse_args()

    #Device
    args.device = torch.device('cpu') if args.no_cuda else torch.device('cuda')

    main(args)