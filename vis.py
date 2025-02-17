"""Evaluate CPPFNet. Also contains functionality to compute evaluation metrics given transforms

Example Usages:
    1. Visualize CPPFNet
        python vis.py --noise_type crop --resume [path-to-model.pth] --dataset_path [your_path]/modelnet40_ply_hdf5_2048
"""
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import open3d as o3d
import random
from tqdm import tqdm
import torch

from arguments import cppfnet_eval_arguments
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, to_numpy
from common.math_torch import se3
from data_loader.datasets import get_test_datasets
import models.cppfnet
import time


def vis(npys):
    pcds = []
    colors = [[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]]
    for ind, npy in enumerate(npys):
        color = colors[ind] if ind < 3 else [random.random() for _ in range(3)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(npy)
        pcd.paint_uniform_color(color)
        pcds.append(pcd)
    return pcds


def inference_vis(data_loader, model: torch.nn.Module):
    _logger.info('Starting inference...')
    model.eval()
    start = time.time()
    n = 0
    with torch.no_grad():
        for data in tqdm(data_loader):

            #gt_transforms = data['transform_gt']
            # points_src = data['points_src'][..., :3].cuda()
            # points_ref = data['points_ref'][..., :3].cuda()
            points_src = data['points_src'][..., :3]
            points_ref = data['points_ref'][..., :3]
            #points_raw = data['points_raw'][..., :3]
            dict_all_to_device(data, _device)
            pred_transforms, endpoints = model(data, _args.num_reg_iter)
            src_transformed = se3.transform(pred_transforms[-1], points_src)

            src_np = torch.squeeze(points_src).cpu().detach()
            src_transformed_np = torch.squeeze(src_transformed).cpu().detach()
            ref_np = torch.squeeze(points_ref).cpu().detach()
            src_transformed_np = src_transformed_np.numpy()

            #计算推理用时
            end = time.time()
            print("循环运行时间:%.2f秒" % (end - start))
            #保存点云
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(src_transformed_np)
            # o3d.io.write_point_cloud(r'E:/' + 'src_transformed_np' + ".pcd", pcd)
            pcds = vis([src_np, src_transformed_np, ref_np])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(src_transformed_np)
            o3d.io.write_point_cloud(r'E:\test\2/' + str(n) + ".pcd", pcd, True)
            n = n + 1
            # pcds_1 = vis([src_np, src_transformed_np])
            # pcds = vis([ref_np])
            # o3d.visualization.draw_geometries(pcds)
            # o3d.visualization.draw_geometries(pcds_1)


def get_model():
    _logger.info('Computing transforms using {}'.format(_args.method))
    assert _args.resume is not None
    model = models.cppfnet.get_model(_args)
    model.to(_device)
    if _device == torch.device('cpu'):
        model.load_state_dict(
            torch.load(_args.resume, map_location=torch.device('cpu'))['state_dict'])
    else:
        model.load_state_dict(torch.load(_args.resume)['state_dict'])
    return model


def main():
    # Load data_loader
    test_dataset = get_test_datasets(_args)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1, shuffle=False)
    model = get_model()
    inference_vis(test_loader, model)  # Feedforward transforms

    _logger.info('Finished')


if __name__ == '__main__':
    # Arguments and logging
    parser = cppfnet_eval_arguments()
    _args = parser.parse_args()
    _logger, _log_path = prepare_logger(_args, log_path=_args.eval_save_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    if _args.gpu >= 0 and (_args.method == 'cppf' or _args.method == 'cppfnet'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
        _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        _device = torch.device('cpu')

    main()
