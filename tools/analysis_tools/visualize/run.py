# uniad's visualization
import cv2
import torch
import argparse
import os
import re
import glob
import sys
from tqdm import tqdm
import imageio
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import numpy as np
import mmcv
import matplotlib
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils import splits
from pyquaternion import Quaternion
#from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import obtain_map_info
from mmcv.datasets.nuscenes_e2e_dataset import obtain_map_info
#from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
from mmcv.datasets.eval_utils.map_api import NuScenesMap
from PIL import Image
from tools.analysis_tools.visualize.utils import color_mapping, AgentPredictionData
from tools.analysis_tools.visualize.render.bev_render import BEVRender
from tools.analysis_tools.visualize.render.cam_render import CameraRender


class Visualizer:
    """
    BaseRender class
    """

    def __init__(
            self,
            dataroot='/mnt/petrelfs/yangjiazhi/e2e_proj/data/nus_mini',
            version='v1.0-mini',
            predroot=None,
            with_occ_map=False,
            with_map=False,
            with_planning=False,
            with_pred_box=True,
            with_pred_traj=False,
            show_gt_boxes=False,
            show_lidar=False,
            show_command=False,
            show_hd_map=False,
            show_sdc_car=False,
            show_sdc_traj=False,
            show_legend=False):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.predict_helper = PredictHelper(self.nusc)
        self.with_occ_map = with_occ_map
        self.with_map = with_map
        self.with_planning = with_planning
        self.show_lidar = show_lidar
        self.show_command = show_command
        self.show_hd_map = show_hd_map
        self.show_sdc_car = show_sdc_car
        self.show_sdc_traj = show_sdc_traj
        self.show_legend = show_legend
        self.with_pred_traj = with_pred_traj
        self.with_pred_box = with_pred_box
        self.veh_id_list = [0, 1, 2, 3, 4, 6, 7]
        self.use_json = '.json' in predroot
        self.token_set = set()
        self.predictions = self._parse_predictions_multitask_pkl(predroot)
        self.bev_render = BEVRender(show_gt_boxes=show_gt_boxes)
        self.cam_render = CameraRender(show_gt_boxes=show_gt_boxes)

        if self.show_hd_map:
            self.nusc_maps = {location: NuScenesMap(dataroot=dataroot, map_name=location) for location in 
                            ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']}

    def _parse_predictions_multitask_pkl(self, predroot):
        outputs = mmcv.fileio.io.load(predroot)
        outputs = outputs['bbox_results']
        prediction_dict = dict()
        for k in range(len(outputs)):
            token = outputs[k]['token']
            self.token_set.add(token)
            if self.show_sdc_traj:
                outputs[k]['boxes_3d'].tensor = torch.cat(
                    [outputs[k]['boxes_3d'].tensor, outputs[k]['sdc_boxes_3d'].tensor], dim=0)
                outputs[k]['scores_3d'] = torch.cat(
                    [outputs[k]['scores_3d'], outputs[k]['sdc_scores_3d']], dim=0)
                outputs[k]['labels_3d'] = torch.cat([outputs[k]['labels_3d'], torch.zeros(
                    (1,), device=outputs[k]['labels_3d'].device)], dim=0)
            # detection
            bboxes = outputs[k]['boxes_3d']
            scores = outputs[k]['scores_3d']
            labels = outputs[k]['labels_3d']

            track_scores = scores.cpu().detach().numpy()
            track_labels = labels.cpu().detach().numpy()
            track_boxes = bboxes.tensor.cpu().detach().numpy()

            track_centers = bboxes.gravity_center.cpu().detach().numpy()
            track_dims = bboxes.dims.cpu().detach().numpy()
            track_yaw = bboxes.yaw.cpu().detach().numpy()

            if 'track_ids' in outputs[k]:
                track_ids = outputs[k]['track_ids'].cpu().detach().numpy()
            else:
                track_ids = None

            # speed
            track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

            # trajectories
            if 'traj' in outputs[k]:
                trajs = outputs[k]['traj'].numpy()
                traj_scores = outputs[k]['traj_scores'].numpy()
            else:

                trajs = None
                traj_scores = None

            predicted_agent_list = []

            # occflow
            if self.with_occ_map:
                if 'topk_query_ins_segs' in outputs[k]['occ']:
                    occ_map = outputs[k]['occ']['topk_query_ins_segs'][0].cpu(
                    ).numpy()
                else:
                    occ_map = np.zeros((1, 5, 200, 200))
            else:
                occ_map = None

            occ_idx = 0
            for i in range(track_scores.shape[0]):
                if track_scores[i] < 0.25:
                    continue
                if occ_map is not None and track_labels[i] in self.veh_id_list:
                    occ_map_cur = occ_map[occ_idx, :, ::-1]
                    occ_idx += 1
                else:
                    occ_map_cur = None
                if track_ids is not None:
                    if i < len(track_ids):
                        track_id = track_ids[i]
                    else:
                        track_id = 0
                else:
                    track_id = None


                if trajs is not None and traj_scores is not None:
                    traj = trajs[i]
                    traj_score = traj_scores[i]
                else:
                    traj = None
                    traj_score = None

                predicted_agent_list.append(
                    AgentPredictionData(
                        track_scores[i],
                        track_labels[i],
                        track_centers[i],
                        track_dims[i],
                        track_yaw[i],
                        track_velocity[i],
                        traj,
                        traj_score,
                        pred_track_id=track_id,
                        pred_occ_map=occ_map_cur,
                        past_pred_traj=None
                    )
                )

            if self.with_map:
                map_thres = 0.7
                score_list = outputs[k]['pts_bbox']['score_list'].cpu().numpy().transpose([
                    1, 2, 0])
                predicted_map_seg = outputs[k]['pts_bbox']['lane_score'].cpu().numpy().transpose([
                    1, 2, 0])  # H, W, C
                predicted_map_seg[..., -1] = score_list[..., -1]
                predicted_map_seg = (predicted_map_seg > map_thres) * 1.0
                predicted_map_seg = predicted_map_seg[::-1, :, :]
            else:
                predicted_map_seg = None

            if self.with_planning:
                # detection
                bboxes = outputs[k]['sdc_boxes_3d']
                scores = outputs[k]['sdc_scores_3d']
                labels = 0

                track_scores = scores.cpu().detach().numpy()
                track_labels = labels
                track_boxes = bboxes.tensor.cpu().detach().numpy()

                track_centers = bboxes.gravity_center.cpu().detach().numpy()
                track_dims = bboxes.dims.cpu().detach().numpy()
                track_yaw = bboxes.yaw.cpu().detach().numpy()
                track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

                if self.show_command and 'command' in outputs[k]:
                    command = outputs[k]['command'][0].cpu().detach().numpy()
                else:
                    command = None

                if 'planning_traj' in outputs[k]:
                    planning_traj = outputs[k]['planning_traj'][0].cpu().detach().numpy()
                else:
                    planning_traj = None

                planning_agent = AgentPredictionData(
                    track_scores[0],
                    track_labels,
                    track_centers[0],
                    track_dims[0],
                    track_yaw[0],
                    track_velocity[0],
                    planning_traj,
                    1,
                    pred_track_id=-1,
                    pred_occ_map=None,
                    past_pred_traj=None,
                    is_sdc=True,
                    command=command,
                )
                predicted_agent_list.append(planning_agent)
            else:
                planning_agent = None

            prediction_dict[token] = dict(predicted_agent_list=predicted_agent_list,
                                        predicted_map_seg=predicted_map_seg,
                                        predicted_planning=planning_agent)
        return prediction_dict

    def visualize_bev(self, sample_token, out_filename, t=None):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()

        if self.show_lidar:
            self.bev_render.show_lidar_data(sample_token, self.nusc)
        if self.bev_render.show_gt_boxes:
            self.bev_render.render_anno_data(
                sample_token, self.nusc, self.predict_helper)
        if self.with_pred_box:
            self.bev_render.render_pred_box_data(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_pred_traj:
            self.bev_render.render_pred_traj(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_map:
            self.bev_render.render_pred_map_data(
                self.predictions[sample_token]['predicted_map_seg'])
        if self.with_occ_map:
            self.bev_render.render_occ_map_data(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_planning:
            self.bev_render.render_pred_box_data(
                [self.predictions[sample_token]['predicted_planning']])
            self.bev_render.render_planning_data(
                self.predictions[sample_token]['predicted_planning'], show_command=self.show_command)
        if self.show_hd_map:
            self.bev_render.render_hd_map(
                self.nusc, self.nusc_maps, sample_token)
        if self.show_sdc_car:
            self.bev_render.render_sdc_car()
        if self.show_legend:
            self.bev_render.render_legend()
        self.bev_render.save_fig(out_filename + '.jpg')

    def visualize_cam(self, sample_token, out_filename):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)
        self.cam_render.render_pred_track_bbox(
            self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc)
        if self.with_pred_traj:
            self.cam_render.render_pred_traj(
                self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc, render_sdc=self.with_planning)
        self.cam_render.save_fig(out_filename + '_cam.jpg')

    def combine(self, out_filename):
        # pass
        bev_image = cv2.imread(out_filename + '.jpg')
        cam_image = cv2.imread(out_filename + '_cam.jpg')
        merge_image = cv2.hconcat([cam_image, bev_image])
        cv2.imwrite(out_filename + '.jpg', merge_image)
        os.remove(out_filename + '_cam.jpg')

    def to_video(self, folder_path, output_path, fps=4, downsample=1):
        def extract_number(file_name):
            match = re.match(r'(\d+)', file_name)
            return int(match.group()) if match else float('inf')

        files = os.listdir(folder_path)
        imgs_path = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
        imgs_path.sort(key=extract_number)

        if not imgs_path:
            print(f"No images found in {folder_path}. Cannot create video.")
            return

        first_img = cv2.imread(os.path.join(folder_path, imgs_path[0]))
        if first_img is None:
            print("Failed to read the first image. Cannot proceed.")
            return

        h, w = first_img.shape[:2]
        h //= downsample
        w //= downsample
        h = h if h % 2 == 0 else h - 1
        w = w if w % 2 == 0 else w - 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for img_name in tqdm(imgs_path, desc="Writing video"):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)

        writer.release()
        print(f"Video saved successfully: {output_path}")


def main(args):
    render_cfg = dict(
        with_occ_map=False, # Not working
        with_map=False, # mapformer output, make sure comment the pop function in uniad_e2e.py
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_gt_boxes=True,
        show_lidar=False, 
        show_command=True, # ['TURN RIGHT', 'TURN LEFT', 'KEEP FORWARD']
        show_hd_map=True, # 
        show_sdc_car=True, # showing the car picture (png from source/)
        show_legend=False, # showing the legend (png from source/)
        show_sdc_traj=False # showing sdc bboxes 3d
    )

    viser = Visualizer(version=args.version, predroot=args.predroot, dataroot=args.dataroot, **render_cfg)

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    val_splits = splits.val

    scene_token_to_name = dict()
    for i in range(len(viser.nusc.scene)):
        scene_token_to_name[viser.nusc.scene[i]['token']] = viser.nusc.scene[i]['name']

    for i in range(len(viser.nusc.sample)):
        sample_token = viser.nusc.sample[i]['token']
        scene_token = viser.nusc.sample[i]['scene_token']

        if scene_token_to_name[scene_token] not in val_splits:
            continue

        if sample_token not in viser.token_set:
            print(i, sample_token, 'not in prediction pkl!')
            continue
        

        viser.visualize_bev(sample_token, os.path.join(args.out_folder, str(i).zfill(3)))

        if args.project_to_cam:
            viser.visualize_cam(sample_token, os.path.join(args.out_folder, str(i).zfill(3)))
            viser.combine(os.path.join(args.out_folder, str(i).zfill(3)))

    viser.to_video(args.out_folder, args.demo_video, fps=30, downsample=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data/m3cad_carla_ue5', help='Path to the dataset root directory')
    parser.add_argument('--version', default='v1.0-mini', help='Dataset version (e.g., v1.0-mini, v1.0-trainval, v1.0-test)')
    parser.add_argument('--predroot', default='output/results.pkl', help='Path to results.pkl')
    parser.add_argument('--out_folder', default='output_vis', help='Output folder path')
    parser.add_argument('--demo_video', default='mini_val_final.avi', help='Demo video name')
    parser.add_argument('--project_to_cam', action='store_true', help='Project to cam (default: False)')
    args = parser.parse_args()
    main(args)


# # has a note: which the image belongs to which frame
# #!/usr/bin/env python3
# import cv2
# import torch
# import argparse
# import os
# import re
# import glob
# import sys
# import json
# import errno
# from tqdm import tqdm
# import imageio
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
# import numpy as np
# import mmcv
# import matplotlib
# import matplotlib.pyplot as plt
# from nuscenes import NuScenes
# from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
# from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
# from nuscenes.utils.data_classes import LidarPointCloud, Box
# from nuscenes.utils import splits
# from pyquaternion import Quaternion
# # from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import obtain_map_info
# from mmcv.datasets.nuscenes_e2e_dataset import obtain_map_info
# # from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
# from mmcv.datasets.eval_utils.map_api import NuScenesMap
# from PIL import Image
# from tools.analysis_tools.visualize.utils import color_mapping, AgentPredictionData
# from tools.analysis_tools.visualize.render.bev_render import BEVRender
# from tools.analysis_tools.visualize.render.cam_render import CameraRender


# class Visualizer:
#     """
#     BaseRender class
#     """

#     def __init__(
#             self,
#             dataroot='/mnt/petrelfs/yangjiazhi/e2e_proj/data/nus_mini',
#             version='v1.0-mini',
#             predroot=None,
#             with_occ_map=False,
#             with_map=False,
#             with_planning=False,
#             with_pred_box=True,
#             with_pred_traj=False,
#             show_gt_boxes=False,
#             show_lidar=False,
#             show_command=False,
#             show_hd_map=False,
#             show_sdc_car=False,
#             show_sdc_traj=False,
#             show_legend=False):
#         self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
#         self.predict_helper = PredictHelper(self.nusc)
#         self.with_occ_map = with_occ_map
#         self.with_map = with_map
#         self.with_planning = with_planning
#         self.show_lidar = show_lidar
#         self.show_command = show_command
#         self.show_hd_map = show_hd_map
#         self.show_sdc_car = show_sdc_car
#         self.show_sdc_traj = show_sdc_traj
#         self.show_legend = show_legend
#         self.with_pred_traj = with_pred_traj
#         self.with_pred_box = with_pred_box
#         self.veh_id_list = [0, 1, 2, 3, 4, 6, 7]
#         self.use_json = '.json' in predroot if predroot is not None else False
#         self.token_set = set()
#         self.predictions = self._parse_predictions_multitask_pkl(predroot) if predroot is not None else {}
#         self.bev_render = BEVRender(show_gt_boxes=show_gt_boxes)
#         self.cam_render = CameraRender(show_gt_boxes=show_gt_boxes)

#         if self.show_hd_map:
#             self.nusc_maps = {location: NuScenesMap(dataroot=dataroot, map_name=location) for location in
#                               ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']}

#     def _parse_predictions_multitask_pkl(self, predroot):
#         outputs = mmcv.fileio.io.load(predroot)
#         # expected structure: outputs['bbox_results']
#         outputs = outputs.get('bbox_results', outputs) if isinstance(outputs, dict) else outputs
#         prediction_dict = dict()
#         for k in range(len(outputs)):
#             token = outputs[k].get('token') if isinstance(outputs[k], dict) else outputs[k]['token']
#             if token is None:
#                 continue
#             self.token_set.add(token)
#             # keep original code behavior where possible
#             if self.show_sdc_traj and 'sdc_boxes_3d' in outputs[k] and 'boxes_3d' in outputs[k]:
#                 outputs[k]['boxes_3d'].tensor = torch.cat(
#                     [outputs[k]['boxes_3d'].tensor, outputs[k]['sdc_boxes_3d'].tensor], dim=0)
#                 outputs[k]['scores_3d'] = torch.cat(
#                     [outputs[k]['scores_3d'], outputs[k]['sdc_scores_3d']], dim=0)
#                 outputs[k]['labels_3d'] = torch.cat([outputs[k]['labels_3d'], torch.zeros(
#                     (1,), device=outputs[k]['labels_3d'].device)], dim=0)

#             # detection
#             bboxes = outputs[k]['boxes_3d']
#             scores = outputs[k]['scores_3d']
#             labels = outputs[k]['labels_3d']

#             track_scores = scores.cpu().detach().numpy()
#             track_labels = labels.cpu().detach().numpy()
#             track_boxes = bboxes.tensor.cpu().detach().numpy()

#             track_centers = bboxes.gravity_center.cpu().detach().numpy()
#             track_dims = bboxes.dims.cpu().detach().numpy()
#             track_yaw = bboxes.yaw.cpu().detach().numpy()

#             if 'track_ids' in outputs[k]:
#                 track_ids = outputs[k]['track_ids'].cpu().detach().numpy()
#             else:
#                 track_ids = None

#             # speed
#             track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

#             # trajectories
#             if 'traj' in outputs[k]:
#                 trajs = outputs[k]['traj'].numpy()
#                 traj_scores = outputs[k]['traj_scores'].numpy()
#             else:
#                 trajs = None
#                 traj_scores = None

#             predicted_agent_list = []

#             # occflow
#             if self.with_occ_map:
#                 if 'occ' in outputs[k] and 'topk_query_ins_segs' in outputs[k]['occ']:
#                     occ_map = outputs[k]['occ']['topk_query_ins_segs'][0].cpu().numpy()
#                 else:
#                     occ_map = np.zeros((1, 5, 200, 200))
#             else:
#                 occ_map = None

#             occ_idx = 0
#             for i in range(track_scores.shape[0]):
#                 if track_scores[i] < 0.25:
#                     continue
#                 if occ_map is not None and track_labels[i] in self.veh_id_list:
#                     occ_map_cur = occ_map[occ_idx, :, ::-1]
#                     occ_idx += 1
#                 else:
#                     occ_map_cur = None
#                 if track_ids is not None:
#                     if i < len(track_ids):
#                         track_id = track_ids[i]
#                     else:
#                         track_id = 0
#                 else:
#                     track_id = None

#                 if trajs is not None and traj_scores is not None:
#                     traj = trajs[i]
#                     traj_score = traj_scores[i]
#                 else:
#                     traj = None
#                     traj_score = None

#                 predicted_agent_list.append(
#                     AgentPredictionData(
#                         track_scores[i],
#                         track_labels[i],
#                         track_centers[i],
#                         track_dims[i],
#                         track_yaw[i],
#                         track_velocity[i],
#                         traj,
#                         traj_score,
#                         pred_track_id=track_id,
#                         pred_occ_map=occ_map_cur,
#                         past_pred_traj=None
#                     )
#                 )

#             if self.with_map:
#                 map_thres = 0.7
#                 score_list = outputs[k]['pts_bbox']['score_list'].cpu().numpy().transpose([1, 2, 0])
#                 predicted_map_seg = outputs[k]['pts_bbox']['lane_score'].cpu().numpy().transpose([1, 2, 0])  # H, W, C
#                 predicted_map_seg[..., -1] = score_list[..., -1]
#                 predicted_map_seg = (predicted_map_seg > map_thres) * 1.0
#                 predicted_map_seg = predicted_map_seg[::-1, :, :]
#             else:
#                 predicted_map_seg = None

#             if self.with_planning:
#                 # detection
#                 bboxes = outputs[k]['sdc_boxes_3d']
#                 scores = outputs[k]['sdc_scores_3d']
#                 labels = 0

#                 track_scores = scores.cpu().detach().numpy()
#                 track_labels = labels
#                 track_boxes = bboxes.tensor.cpu().detach().numpy()

#                 track_centers = bboxes.gravity_center.cpu().detach().numpy()
#                 track_dims = bboxes.dims.cpu().detach().numpy()
#                 track_yaw = bboxes.yaw.cpu().detach().numpy()
#                 track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

#                 if self.show_command and 'command' in outputs[k]:
#                     command = outputs[k]['command'][0].cpu().detach().numpy()
#                 else:
#                     command = None

#                 if 'planning_traj' in outputs[k]:
#                     planning_traj = outputs[k]['planning_traj'][0].cpu().detach().numpy()
#                 else:
#                     planning_traj = None

#                 planning_agent = AgentPredictionData(
#                     track_scores[0],
#                     track_labels,
#                     track_centers[0],
#                     track_dims[0],
#                     track_yaw[0],
#                     track_velocity[0],
#                     planning_traj,
#                     1,
#                     pred_track_id=-1,
#                     pred_occ_map=None,
#                     past_pred_traj=None,
#                     is_sdc=True,
#                     command=command,
#                 )
#                 predicted_agent_list.append(planning_agent)
#             else:
#                 planning_agent = None

#             prediction_dict[token] = dict(
#                 predicted_agent_list=predicted_agent_list,
#                 predicted_map_seg=predicted_map_seg,
#                 predicted_planning=planning_agent
#             )
#         return prediction_dict

#     def visualize_bev(self, sample_token, out_filename, t=None):
#         self.bev_render.reset_canvas(dx=1, dy=1)
#         self.bev_render.set_plot_cfg()

#         if self.show_lidar:
#             self.bev_render.show_lidar_data(sample_token, self.nusc)
#         if self.bev_render.show_gt_boxes:
#             self.bev_render.render_anno_data(sample_token, self.nusc, self.predict_helper)
#         if self.with_pred_box:
#             self.bev_render.render_pred_box_data(self.predictions[sample_token]['predicted_agent_list'])
#         if self.with_pred_traj:
#             self.bev_render.render_pred_traj(self.predictions[sample_token]['predicted_agent_list'])
#         if self.with_map:
#             self.bev_render.render_pred_map_data(self.predictions[sample_token]['predicted_map_seg'])
#         if self.with_occ_map:
#             self.bev_render.render_occ_map_data(self.predictions[sample_token]['predicted_agent_list'])
#         if self.with_planning:
#             self.bev_render.render_pred_box_data([self.predictions[sample_token]['predicted_planning']])
#             self.bev_render.render_planning_data(
#                 self.predictions[sample_token]['predicted_planning'], show_command=self.show_command)
#         if self.show_hd_map:
#             self.bev_render.render_hd_map(self.nusc, self.nusc_maps, sample_token)
#         if self.show_sdc_car:
#             self.bev_render.render_sdc_car()
#         if self.show_legend:
#             self.bev_render.render_legend()
#         self.bev_render.save_fig(out_filename + '.jpg')

#     def visualize_cam(self, sample_token, out_filename):
#         self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
#         self.cam_render.render_image_data(sample_token, self.nusc)
#         self.cam_render.render_pred_track_bbox(
#             self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc)
#         if self.with_pred_traj:
#             self.cam_render.render_pred_traj(
#                 self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc, render_sdc=self.with_planning)
#         self.cam_render.save_fig(out_filename + '_cam.jpg')

#     def combine(self, out_filename):
#         # pass
#         bev_image = cv2.imread(out_filename + '.jpg')
#         cam_image = cv2.imread(out_filename + '_cam.jpg')
#         merge_image = cv2.hconcat([cam_image, bev_image])
#         cv2.imwrite(out_filename + '.jpg', merge_image)
#         os.remove(out_filename + '_cam.jpg')


#     def annotate_image(self, img_path, text, margin=16):
#         img = cv2.imread(img_path)
#         if img is None:
#             return

#         h_img, w_img = img.shape[:2]


#         lines = text.split('\n')

#         font = cv2.FONT_HERSHEY_SIMPLEX

#         thickness = 3




#         sizes = []
#         max_w = 0
#         max_h_line = 0
#         for line in lines:
#             (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
#             sizes.append((tw, th))
#             if tw > max_w: max_w = tw
#             if th > max_h_line: max_h_line = th


#         max_text_width = w_img - 2 * margin - 2 * pad
#         if max_text_width <= 0:
#             cv2.putText(img, lines[-1], (margin, h_img - margin), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
#             cv2.imwrite(img_path, img)
#             return


#         attempts = 0
#         while max_w > max_text_width and attempts < 12:
#             scale *= 0.85
#             sizes = []
#             max_w = 0
#             max_h_line = 0
#             for line in lines:
#                 (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
#                 sizes.append((tw, th))
#                 if tw > max_w: max_w = tw
#                 if th > max_h_line: max_h_line = th
#             attempts += 1


#         total_height = sum([h for (_, h) in sizes]) + line_spacing * (len(lines) - 1)


#         y_bottom = h_img - margin  # baseline of last line
#         rect_height = total_height + 2 * pad
#         rect_tl_y = max(h_img - margin - rect_height, 0)
#         rect_br_y = min(h_img - margin, h_img - 1)

#         max_line_w = max([w for (w, _) in sizes])
#         rect_tl_x = max((w_img - max_line_w) // 2 - pad, 0)
#         rect_br_x = min((w_img + max_line_w) // 2 + pad, w_img - 1)

#         overlay = img.copy()
#         cv2.rectangle(overlay, (rect_tl_x, rect_tl_y), (rect_br_x, rect_br_y), (0, 0, 0), -1)
#         alpha = 0.45
#         img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


#         current_y = rect_tl_y + pad + sizes[0][1]  # baseline for first line
#         for idx, line in enumerate(lines):
#             tw, th = sizes[idx]
#             x = max((w_img - tw) // 2, margin)
#             cv2.putText(img, line, (x, current_y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
#             cv2.putText(img, line, (x, current_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
#             current_y += th + line_spacing

#         cv2.imwrite(img_path, img)

#     def to_video(self, folder_path, output_path, fps=4, downsample=1):
#         def extract_number(file_name):
#             match = re.match(r'(\d+)', file_name)
#             return int(match.group()) if match else float('inf')

#         files = os.listdir(folder_path)
#         imgs_path = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
#         imgs_path.sort(key=extract_number)

#         if not imgs_path:
#             print(f"No images found in {folder_path}. Cannot create video.")
#             return

#         first_img = cv2.imread(os.path.join(folder_path, imgs_path[0]))
#         if first_img is None:
#             print("Failed to read the first image. Cannot proceed.")
#             return

#         h, w = first_img.shape[:2]
#         h //= downsample
#         w //= downsample
#         h = h if h % 2 == 0 else h - 1
#         w = w if w % 2 == 0 else w - 1

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

#         for img_name in tqdm(imgs_path, desc="Writing video"):
#             img_path = os.path.join(folder_path, img_name)
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue
#             img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
#             writer.write(img)

#         writer.release()
#         print(f"Video saved successfully: {output_path}")


# def _get_token_from_agent(agent):
#     """


#     """
#     if agent is None:
#         return 'N/A'
#     for attr in ['sample_token', 'token', 'instance_token']:
#         try:
#             val = getattr(agent, attr)
#             if val is not None:
#                 return str(val)
#         except Exception:
#             continue

#     try:
#         val = getattr(agent, 'pred_track_id', None)
#         return str(val) if val is not None else 'N/A'
#     except Exception:
#         return 'N/A'


# def main(args):
#     render_cfg = dict(
#         with_occ_map=False,  # Not working
#         with_map=False,  # mapformer output, make sure comment the pop function in uniad_e2e.py
#         with_planning=True,
#         with_pred_box=True,
#         with_pred_traj=True,
#         show_gt_boxes=True,
#         show_lidar=False,
#         show_command=True,  # ['TURN RIGHT', 'TURN LEFT', 'KEEP FORWARD']
#         show_hd_map=True,  #
#         show_sdc_car=True,  # showing the car picture (png from source/)
#         show_legend=False,  # showing the legend (png from source/)
#         show_sdc_traj=False  # showing sdc bboxes 3d
#     )

#     viser = Visualizer(version=args.version, predroot=args.predroot, dataroot=args.dataroot, **render_cfg)

#     if not os.path.exists(args.out_folder):
#         os.makedirs(args.out_folder, exist_ok=True)

#     val_splits = splits.val

#     scene_token_to_name = dict()
#     for i in range(len(viser.nusc.scene)):
#         scene_token_to_name[viser.nusc.scene[i]['token']] = viser.nusc.scene[i]['name']


#     ndjson_path = os.path.join(args.out_folder, "frame_mapping.ndjson")

#     try:
#         nd_f = open(ndjson_path, "a", buffering=1, encoding="utf-8")
#     except Exception as e:
#         print(f"Failed to open ndjson file for append: {e}")
#         nd_f = None


#     for i in range(len(viser.nusc.sample)):
#         sample_token = viser.nusc.sample[i]['token']
#         scene_token = viser.nusc.sample[i]['scene_token']

#         if scene_token_to_name.get(scene_token) not in val_splits:
#             continue

#         if sample_token not in viser.token_set:

#             print(i, sample_token, 'not in prediction pkl!')
#             continue


#         out_base = os.path.join(args.out_folder, str(i).zfill(3))


#         viser.visualize_bev(sample_token, out_base)
#         if args.project_to_cam:
#             viser.visualize_cam(sample_token, out_base)



#         pred = viser.predictions.get(sample_token, None)
#         if pred is None:
#             ego_token = 'N/A'
#             sender_tokens = []
#         else:
#             planning_agent = pred.get('predicted_planning', None)
#             ego_token = _get_token_from_agent(planning_agent)

#             sender_tokens = []
#             pal = pred.get('predicted_agent_list', [])
#             for ag in pal:
#                 tok = _get_token_from_agent(ag)
#                 if tok is None or tok == 'N/A':
#                     continue
#                 if ego_token != 'N/A' and tok == ego_token:
#                     continue
#                 if tok not in sender_tokens:
#                     sender_tokens.append(tok)


#         display_senders = sender_tokens[:10]
#         sender_str = ','.join(display_senders) if display_senders else 'None'
#         print(f"[Frame {i:04d}] scene: {scene_token_to_name.get(scene_token, 'N/A')}  ego_token: {ego_token}  senders_tokens: {sender_str}  sample_token: {sample_token}")


#         if args.stamp_meta:
#             line1 = f"idx:{i:04d}  scene:{scene_token_to_name.get(scene_token, 'N/A')}"
#             line2 = f"ego_token:{ego_token}  senders:{sender_str}"
#             line3 = f"sample:{sample_token}"
#             stamp = '\n'.join([line1, line2, line3])
#             viser.annotate_image(out_base + '.jpg', stamp)


#         mapping_entry = {
#             "frame_index": i,
#             "image_name": f"{str(i).zfill(3)}.jpg",
#             "sample_token": sample_token,
#             "scene_name": scene_token_to_name.get(scene_token, 'N/A'),
#             "ego_token": ego_token,

#         }
#         if nd_f is not None:
#             try:
#                 nd_f.write(json.dumps(mapping_entry, ensure_ascii=False) + "\n")
#                 nd_f.flush()
#                 try:
#                     os.fsync(nd_f.fileno())
#                 except OSError:

#                     pass
#             except Exception as e:
#                 print(f"Failed to append mapping entry to ndjson: {e}")


#     if 'nd_f' in locals() and nd_f is not None:
#         try:
#             nd_f.close()
#         except Exception:
#             pass


#     mapping_path = os.path.join(args.out_folder, "frame_mapping.json")
#     try:
#         entries = []
#         if os.path.exists(ndjson_path):
#             with open(ndjson_path, "r", encoding="utf-8") as fh:
#                 for line in fh:
#                     line = line.strip()
#                     if not line:
#                         continue
#                     try:
#                         entries.append(json.loads(line))
#                     except Exception:

#                         continue

#         with open(mapping_path, "w", encoding="utf-8") as wf:
#             json.dump(entries, wf, indent=2, ensure_ascii=False)
#         print(f"Frame mapping NDJSON saved to: {os.path.abspath(ndjson_path)}")
#         print(f"Frame mapping JSON saved to: {os.path.abspath(mapping_path)}")
#     except Exception as e:
#         print(f"Failed to merge/write mapping JSON: {e}")


#     viser.to_video(args.out_folder, args.demo_video, fps=30, downsample=2)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataroot', default='data/m3cad_carla_ue5', help='Path to the dataset root directory')
#     parser.add_argument('--version', default='v1.0-mini', help='Dataset version (e.g., v1.0-mini, v1.0-trainval, v1.0-test)')
#     parser.add_argument('--predroot', default='output/results.pkl', help='Path to results.pkl')
#     parser.add_argument('--out_folder', default='output_vis', help='Output folder path')
#     parser.add_argument('--demo_video', default='mini_val_final.avi', help='Demo video name')
#     parser.add_argument('--project_to_cam', action='store_true', help='Project to cam (default: False)')

#     parser.add_argument('--stamp_meta', action='store_true',
#                         help='Stamp frame index / scene / sample token onto output images')
#     args = parser.parse_args()
#     main(args)

