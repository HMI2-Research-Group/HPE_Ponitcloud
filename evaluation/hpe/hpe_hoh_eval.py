# Human-Object-Human (HOH) Evaluation
# Copyright (C) 2024 HMI^2 Lab Santa Clara University

import os
import sys  
import time
import numpy as np
import pickle
import constant

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tabulate import tabulate
from dex_ycb_toolkit.logging import get_logger
from utils.eval_util import EvalUtil
from eval import align_w_scale, curve, createHTML
from get_hoh_datasets import get_dataset

_AUC_VAL_MIN = 0.0
_AUC_VAL_MAX = 50.0
_AUC_STEPS = 100

class HPEEvaluator():
    def __init__(self, name, handtype):
        """Constructor.

        Args:
          name: Dataset name. E.g., 'hoh_test'.
          handtype: Type of hand. 'giver', 'receiver'.
        """
        self._name = name
        self._handtype = handtype

        self._dataset = get_dataset(self._name, self._handtype)
        self._out_dir = os.path.join(os.path.dirname(__file__), "results")
        self._anno_file = os.path.join(self._out_dir, "anno_hpe_{}_{}.pkl".format(self._name, self._handtype))
        #self._anno_file = os.path.join("evaluation/dexycb_sample_results","anno_hpe_s0_test.pkl")

        if os.path.isfile(self._anno_file):
            print('Found HPE annotation file.')
        else:
            print('Cannot find HPE annotation file.')
            self._generate_anno_file()

        self._anno = self._load_anno_file()

    def _generate_anno_file(self):
        """Generates the annotation file."""
        print('Generating HPE annotation file')
        s = time.time()

        joint_3d_gt = {}
        total_samples = 0  #total number of samples

        for i, sample in enumerate(self._dataset):
            total_samples += len(sample) 
            if (i + 1) in np.floor(np.linspace(0, len(self._dataset), 11))[1:]:
                print('{:3.0f}%  Sample length: {:6d}'.format(100 * i / len(self._dataset), len(sample)))

            # Iterate over each dictionary in the sample list
            for j, entry in enumerate(sample):
                if not isinstance(entry, dict):
                    continue
                
                key = j + 1

                if key is None:
                    continue
                
                for hand_type in [self._handtype]:
                    hand_data = entry.get(hand_type)

                    if hand_data is None:
                        continue

                    label_file = hand_data.get('label_file')
                    
                    if label_file is None or not os.path.isfile(label_file):
                        continue
                    
                    label = np.load(label_file)
                    joint_3d = label['joint_3d'].reshape(21, 3)
                    
                    if np.all(joint_3d == -1) and hand_type == 'receiver':
                        joint_3d = constant.RECEIVER_HAND_POSE
                        joint_3d_gt[key] = joint_3d
                        continue
                    elif np.all(joint_3d == -1) and hand_type == 'giver':
                        joint_3d = constant.GIVER_HAND_POSE
                        joint_3d_gt[key] = joint_3d
                        continue
                    
                    joint_3d *= 10**35
                    joint_3d_gt[key] = joint_3d

        print('# total samples: {:6d}'.format(total_samples))
        # print('# valid {:6s} samples: {:6d}'.format(self._handtype, total_samples))

        anno = {
            'joint_3d': joint_3d_gt,
        }
        with open(self._anno_file, 'wb') as f:
            pickle.dump(anno, f)

        e = time.time()
        print('time: {:7.2f}'.format(e - s))


    def _load_anno_file(self):
        """Loads the annotation file.

        Returns:
          A dictionary holding the loaded annotation.
        """
        with open(self._anno_file, 'rb') as f:
            anno = pickle.load(f)

        anno['joint_3d'] = {
            k: v.astype(np.float64) for k, v in anno['joint_3d'].items()
        }

        return anno

    def _load_results(self, res_file):
        """Loads results from a result file.

        Args:
          res_file: Path to the result file.

        Returns:
          A dictionary holding the loaded results.
        """
        results = {}
        with open(res_file, 'r') as f:
            for line in f:
                elems = line.split(',')
                if len(elems) != 64:
                    raise ValueError(
                        'a line does not have 64 comma-separated elements: {}'.format(line))
                image_id = int(elems[0])
                joint_3d = np.array(elems[1:], dtype=np.float64).reshape(21, 3)
                results[image_id] = joint_3d
        return results

    def evaluates(self, res_file, out_dir=None):
        """Evaluates HPE metrics given a result file.

        Args:
          res_file: Path to the result file.
          out_dir: Path to the output directory.

        Returns:
          A dictionary holding the results.
        """
        if out_dir is None:
            out_dir = self._out_dir

        res_name = os.path.splitext(os.path.basename(res_file))[0]
        log_file = os.path.join(out_dir, "hpe_eval_{}_{}.log".format(self._name, res_name))
        logger = get_logger(log_file)

        res = self._load_results(res_file)
        logger.info('Running evaluation')

        joint_3d_gt = self._anno['joint_3d']

        eval_util_ab = EvalUtil()
        eval_util_rr = EvalUtil()
        eval_util_pa = EvalUtil()

        for idx, (i, kpt_gt) in enumerate(joint_3d_gt.items(), start=1):
            assert idx in res, "missing id in result file: {}".format(idx)
            vis = np.ones_like(kpt_gt[:, 0])
            kpt_pred = res[idx]

            eval_util_ab.feed(kpt_gt, vis, kpt_pred)
            eval_util_rr.feed(kpt_gt - kpt_gt[0], vis, kpt_pred - kpt_pred[0])
            eval_util_pa.feed(kpt_gt, vis, align_w_scale(kpt_gt, kpt_pred))


        mean_ab, _, auc_ab, pck_ab, thresh_ab = eval_util_ab.get_measures(_AUC_VAL_MIN, _AUC_VAL_MAX, _AUC_STEPS)
        mean_rr, _, auc_rr, pck_rr, thresh_rr = eval_util_rr.get_measures(_AUC_VAL_MIN, _AUC_VAL_MAX, _AUC_STEPS)
        mean_pa, _, auc_pa, pck_pa, thresh_pa = eval_util_pa.get_measures(_AUC_VAL_MIN, _AUC_VAL_MAX, _AUC_STEPS)

        tabular_data = [['absolute', mean_ab, auc_ab],
                        ['root-relative', mean_rr, auc_rr],
                        ['procrustes', mean_pa, auc_pa]]
        metrics = ['alignment', 'MPJPE (mm)', 'AUC']
        table = tabulate(tabular_data,
                         headers=metrics,
                         tablefmt='pipe',
                         floatfmt='.4f',
                         numalign='right')
        logger.info('Results: \n' + table)

        hpe_curve_dir = os.path.join(out_dir, "hpe_curve_{}_{}".format(self._name, res_name))
        os.makedirs(hpe_curve_dir, exist_ok=True)

        createHTML(hpe_curve_dir, [
            curve(thresh_ab, pck_ab, 'Distance in mm', 'Percentage of correct keypoints', 'PCK curve for absolute keypoint error'),
            curve(thresh_rr, pck_rr, 'Distance in mm', 'Percentage of correct keypoints', 'PCK curve for root-relative keypoint error'),
            curve(thresh_pa, pck_pa, 'Distance in mm', 'Percentage of correct keypoints', 'PCK curve for Procrustes aligned keypoint error'),
        ])

        results = {
            'absolute': {
                'mpjpe': mean_ab,
                'auc': auc_ab
            },
            'root-relative': {
                'mpjpe': mean_rr,
                'auc': auc_rr
            },
            'procrustes': {
                'mpjpe': mean_pa,
                'auc': auc_pa
            },
        }

        logger.info('Evaluation complete.')

        return results

# Example usage
#evaluator = HPEEvaluator('hoh_test', 'giver')
#results = evaluator.evaluates('evaluation/hoh_eval_results/giver_hpe_predicted_results_test.txt')
#print(results)
