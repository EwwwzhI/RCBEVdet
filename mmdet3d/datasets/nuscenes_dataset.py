# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from random import random, randrange, sample

import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from ..core.visualizer import show_result, show_seg_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        img_info_prototype (str, optional): Type of img information.
            Based on 'img_info_prototype', the dataset will prepare the image
            data info in the type of 'mmcv' for official image infos,
            'bevdet' for BEVDet, and 'bevdet4d' for BEVDet4D.
            Defaults to 'mmcv'.
        multi_adj_frame_id_cfg (tuple[int]): Define the selected index of
            reference adjcacent frames.
        ego_cam (str): Specify the ego coordinate relative to a specified
            camera by its name defined in NuScenes.
            Defaults to None, which use the mean of all cameras.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 multi_adj_frame_id_cfg_longterm=None,
                 ego_cam='CAM_FRONT',
                 stereo=False,
                 map_classes=None,
                 include_location=True,
                 temporal_augment_rate=0):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.multi_adj_frame_id_cfg_longterm = multi_adj_frame_id_cfg_longterm
        self.ego_cam = ego_cam
        self.stereo = stereo
        self.map_classes = map_classes
        self.include_location = include_location
        self.temporal_augment_rate = temporal_augment_rate
        if self.temporal_augment_rate > 0:
            print('USE TEMPORAL AUGMENTATION RATE =', self.temporal_augment_rate)

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        if self.include_location:
            input_dict = dict(
                sample_idx=info['token'],
                pts_filename=info['lidar_path'],
                sweeps=info['sweeps'],
                timestamp=info['timestamp'] / 1e6,
                location=info["location"],
                scene_token=info['scene_token'],
            )
        else:
            input_dict = dict(
                sample_idx=info['token'],
                pts_filename=info['lidar_path'],
                sweeps=info['sweeps'],
                timestamp=info['timestamp'] / 1e6,
                scene_token=info['scene_token'],
            )

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        input_dict["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego

        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype #bev 自己写的里多帧的模型作为baseline，eta会不准）
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
                    if self.multi_adj_frame_id_cfg_longterm is not None:
                        info_adj_list_lt = self.get_adj_info_lt(info, index)
                        input_dict.update(dict(adjacent_lt=info_adj_list_lt))
        else:
            if not self.test_mode:
                annos = self.get_ann_info(index)
                input_dict['ann_info'] = annos
        return input_dict

    def get_adj_info_lt(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg_longterm))
        for select_id in adj_id_list:
            select_id = min(max(index - select_id, 0), len(self.data_infos) - 1)
            if not self.data_infos[select_id]['scene_token'] == info['scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_adj_info(self, info, index):
        info_adj_list = []
        if self.temporal_augment_rate > 0:
            R = random()
            if R < self.temporal_augment_rate:
                adj_id_list = list(
                    range(self.multi_adj_frame_id_cfg[0],
                          self.multi_adj_frame_id_cfg[1] + 1,
                          self.multi_adj_frame_id_cfg[2]))
                adj_id_list.pop(randrange(len(adj_id_list)))
            else:
                adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        else:
            adj_id_list = list(range(*self.multi_adj_frame_id_cfg))

        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        # print("#### multi adj index ####", adj_id_list)
        for select_id in adj_id_list:
            select_id = min(max(index - select_id, 0), len(self.data_infos) - 1)
            if not self.data_infos[select_id]['scene_token'] == info['scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            sample_token = self.data_infos[sample_id]['token']

            trans = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_translation']
            rot = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_rotation']
            rot = pyquaternion.Quaternion(rot)
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval
        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name == 'pts_seg' or 'gt' in name:
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate_map(self, results):

        print("Evaluating for BEV segmentation classes:", self.map_classes)

        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["pts_seg"]
            label = result["gt_masks_bev"]

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred.cpu()
            label = label.cpu()
            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            print("mIoU", name, ":", metrics[f"map/{name}/iou@max"])
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        metrics = {}
        if "pts_seg" in results[0]:
            metrics.update(self.evaluate_map(results))

        if "pts_bbox" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    if name == 'pts_seg' or 'gt' in name:
                        continue
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                    metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files))

            if tmp_dir is not None:
                tmp_dir.cleanup()
        if show or out_dir:
            # self.show(results, out_dir, show=show, pipeline=pipeline)
            # ******* 手动实现可视化结果 *******
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import random
            from datetime import datetime
            mmcv.mkdir_or_exist(out_dir)
            out_dir_pred = osp.join(out_dir, 'pred')
            mmcv.mkdir_or_exist(out_dir_pred)
            if show:
                out_dir_gt = osp.join(out_dir, 'gt')
                mmcv.mkdir_or_exist(out_dir_gt)
            for idx in range(len(results)):
                assert "pts_bbox" in results[idx]
                filename_pred = 'pred/pred--time='+datetime.now().strftime('%H:%M:%S')+'--idx='+str(idx)+'.png'
                result_path_pred = osp.join(out_dir, filename_pred)
                if show:
                    filename_gt = 'gt/gt--time='+datetime.now().strftime('%H:%M:%S')+'--idx='+str(idx)+'.png'
                    result_path_gt = osp.join(out_dir, filename_gt)
                print(f'\n******** BEGIN PRINT RESULT {idx}**********\n')

                print(f'\n******** BEGIN PRINT BBOX**********\n')
                corner = results[idx]['pts_bbox']['boxes_3d'].corners
                fig = plt.figure(figsize=(16, 16))
                plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
                plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
                for i in range(corner.shape[0]):
                    x1 = corner[i][0][0]
                    y1 = corner[i][0][1]
                    x2 = corner[i][2][0]
                    y2 = corner[i][2][1]
                    x3 = corner[i][6][0]
                    y3 = corner[i][6][1]
                    x4 = corner[i][4][0]
                    y4 = corner[i][4][1]
                    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=0.5)

                if "pts_seg" in results[idx]:
                    print('\n******** END PRINT BBOX**********\n')
                    # print seg
                    print('\n******** BEGIN PRINT SEG**********\n')
                    print("BEV:", results[idx]['pts_seg'].shape)
                    for xx in range(200):
                        for yy in range(200):
                            xc = -50 + xx * 0.5
                            yc = -50 + yy * 0.5
                            # 0 vehicle, 1 可行驶区域, 2 车道线
                            if results[idx]['pts_seg'][0, xx, yy] > 0.45:
                                plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                            if results[idx]['pts_seg'][1, xx, yy] > 0.45:
                                plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                            if results[idx]['pts_seg'][2, xx, yy] > 0.40:
                                plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                    plt.savefig(result_path_pred)
                    print('\n******** END PRINT SEG**********\n')
                    # show seg gt
                    if show:
                        print('\n******** BEGIN PRINT GT**********\n')
                        print("BEV:", results[idx]['gt_masks_bev'][0].shape)
                        fig = plt.figure(figsize=(16, 16))
                        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
                        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
                        for xx in range(200):
                            for yy in range(200):
                                xc = -50 + xx * 0.5
                                yc = -50 + yy * 0.5
                                # 0 vehicle, 1 可行驶区域, 2 车道线
                                if results[idx]['gt_masks_bev'][0][0, xx, yy] == 1:
                                    plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                                if results[idx]['gt_masks_bev'][0][1, xx, yy] == 1:
                                    plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                                if results[idx]['gt_masks_bev'][0][2, xx, yy] == 1:
                                    plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))        
                        plt.savefig(result_path_gt)
                        print('\n******** END PRINT GT**********\n')

                else:
                    plt.savefig(result_path_pred)
                    print('\n******** END PRINT BBOX**********\n')

                print(f'\n******** END PRINT RESULT {idx}**********\n')



        return metrics


    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            flag = False
            if 'pts_seg' in result.keys():
                show_gt_seg = result["gt_masks_bev"]
                show_pred_seg = result['pts_seg']
                flag = True
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)
            if flag:
                file_name_seg = file_name + '_seg'
                show_seg_result(points, show_gt_seg, show_pred_seg, out_dir, file_name_seg, show=show)


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


