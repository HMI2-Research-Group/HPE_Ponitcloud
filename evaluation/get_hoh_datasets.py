# Human-Object-Human (HOH) Evaluation
# Copyright (C) 2024 HMI^2 Lab Santa Clara University

import os
import json
import re
import numpy as np
import scipy.io

_SUBJECTS = [
    'subject-00',
    'subject-01',
    'subject-02',
    'subject-03',
    'subject-04',
    'subject-05',
    'subject-07',
    'subject-11',
    'subject-12',
    'subject-13',
]

_HOH_CLASSES = {
     1: '00_plate',
     2: '01_short_cup',
     3: '02_unknown',
     4: '03_roller',
     5: '04_bowl',
     6: '05_jug',
     7: '07_spoon',
     8: '011_mug1',
     9: '012_mug2',
    10: '013_dough_roller',
}

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]


class HoHDataset():
    def __init__(self, split, base_dir, handtype):
        """Constructor.

        Args:
            split: Split name. 'train', 'test'.
            base_dir: Base directory of the dataset.
        """
        self._split = split
        self._base_dir = base_dir
        self._handtype = handtype
        self.base_dir_without_filtered = self._base_dir.replace('filtered', '')
        self._intrinsics = self.load_intrinsics()
        self._sequences = self.load_sequences()
        self._mano_betas = self.load_mano_betas()

    def load_intrinsics(self):
        """Load camera intrinsics from calibration files."""
        intrinsics = {}
        calib_dir = os.path.join(self.base_dir_without_filtered, "calibration")
        
        # Iterate over the group directories
        for group in ['group1_calib', 'group2_calib']: 
            group_dir = os.path.join(calib_dir, group)
            if os.path.isdir(group_dir):
                for subfolder in os.listdir(group_dir):
                    subfolder_path = os.path.join(group_dir, subfolder)
                    if os.path.isdir(subfolder_path):
                        calib_results_path = os.path.join(subfolder_path, 'calib_results')
                        if os.path.isdir(calib_results_path):
                            mat_file = os.path.join(calib_results_path, 'scaled_kc_intrinsics.mat')
                            if os.path.isfile(mat_file):
                                mat_data = scipy.io.loadmat(mat_file)
                                K = mat_data.get('scaled_kc_intrinsics')  # Adjust the key as necessary
                                if K is not None:
                                    fx = K[0, 0]
                                    fy = K[1, 1]
                                    ppx = K[2, 0]
                                    ppy = K[2, 1]
                                    intrinsics[subfolder_path] = {
                                        "fx": fx,
                                        "fy": fy,
                                        "ppx": ppx,
                                        "ppy": ppy
                                    }
        return intrinsics

    def load_mano_betas(self):
        """Load MANO betas from calibration files."""
        mano_betas = {}
        calib_dir = os.path.join(self.base_dir_without_filtered, "calibration")
        for hand_type in ['giver_hand', 'receiver_hand']:
            hand_dir = os.path.join(calib_dir, hand_type)
            if os.path.isdir(hand_dir):
                for subfolder in os.listdir(hand_dir):
                    subfolder_path = os.path.join(hand_dir, subfolder)
                    if os.path.isdir(subfolder_path):
                        for filename in os.listdir(subfolder_path):
                            if filename.endswith(".json"):
                                with open(os.path.join(subfolder_path, filename), 'r') as f:
                                    mano_calib = json.load(f)
                                frame_name_1 = filename.split('_')[-2].split('.')[0]
                                frame_name_2 = filename.split('_')[-1].split('.')[0]
                                frame_name = os.path.join(frame_name_1, frame_name_2)
                                # Divide each value in the inner lists by 10
                                mano_betas[frame_name] = [
                                    [value / 10 for value in mano_calib[0]] 
                                    for mano_calib[0] in mano_calib
                                ]
        return mano_betas

    def load_sequences(self):
        """Load sequences based on the split."""
        sequences = []
        data_dir = os.path.join(self._base_dir, self._split)
    
        for subject_dir in os.listdir(data_dir):
            subject_path = os.path.join(data_dir, subject_dir)
            if os.path.isdir(subject_path):
                for seq_dir in os.listdir(subject_path):
                    seq_path = os.path.join(subject_path, seq_dir)
                    if '.DS_Store' not in seq_path:
                        sequences.append(seq_path)            
        return sequences

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, idx):
        seq_path = self._sequences[idx]
        samples = []
        seen_frames = set()  # To keep track of frames that have already been added

        for subfolder in os.listdir(seq_path):
            if os.path.isdir(seq_path):
                # Iterate through the files in the subfolder
                for filename in os.listdir(seq_path):
                    if not filename.endswith('.ply'):  
                        continue
                                     
                    role = self._handtype
                    
                    if filename.startswith(role):
                        match = re.search(r'/(\d+)/', seq_path)
                        seq_subject = match.group(1) if match else 'unknown'
                        frame = filename.split('_')[1].split('.')[0]
                        
                        if frame in seen_frames:
                            continue  
                        
                        sample = {
                            'subject': seq_subject,
                            'frame': frame,
                            'giver': self._get_sample(seq_path, "giver", frame) if role == "giver" else None,
                            'receiver': self._get_sample(seq_path, "receiver", frame) if role == "receiver" else None,
                            'file': filename  
                        }
                        
                        # Only append samples where the relevant hand type is not None
                        if self._handtype == "giver" and sample['giver'] is not None:
                            samples.append(sample)
                            seen_frames.add(frame)  # Mark this frame as seen
                        elif self._handtype == "receiver" and sample['receiver'] is not None:
                            samples.append(sample)
                            seen_frames.add(frame)  # Mark this frame as seen

        return samples




 
    def _get_sample(self, seq_path, role, frame):
        """Get sample details for either giver or receiver."""
        sample = {
            'intrinsics': self._get_intrinsics(seq_path),
            'hoh_ids': self._get_hoh_ids(seq_path),
            'hoh_grasp_ind': self._get_hoh_grasp_ind(seq_path),
            'mano_side': self._get_mano_side(seq_path),
            'mano_betas': self._get_mano_betas(seq_path, role, frame),
            'label_file': self._get_label_file(seq_path, role, frame)  
        }
        return sample

    def _get_label_file(self, seq_path, role, frame):
        """Retrieve label file path for the sequence."""
        match = re.search(r'/(\d+)/', seq_path)
        seq_subject = match.group(1) if match else 'unknown'
        label_folder = os.path.join('data/HOH/calibration/labels', seq_subject)
        label_file = None
        if frame:
            for file in os.listdir(label_folder):
                if file.startswith(f'{role}_{frame}'):
                    label_file = os.path.join(label_folder, file)
                    break
        return label_file

    def _get_intrinsics(self, seq_path):
        """Retrieve intrinsics for the sequence."""
        # Find the closest matching directory in the calibration folder
        intrinsics_path = os.path.join(self.base_dir_without_filtered, "calibration", "group1_calib", "0") # You can change to group2_calib
        return self._intrinsics.get(intrinsics_path, {})

    def _get_hoh_ids(self, seq_path):
        """Retrieve HoH IDs for the sequence."""
        # Extract the number from the seq_path
        match = re.search(r'/(\d+)/', seq_path)
        if match:
            num = int(match.group(1))
            return [num, 0, 0, 0]
        else:
            return [0, 0, 0, 0]

    def _get_hoh_grasp_ind(self, seq_path):
        """Retrieve HoH grasp index for the sequence."""
        # Implement logic to get HoH grasp index for the sequence
        return 0

    def _get_mano_side(self, seq_path):
        """Retrieve MANO side (left or right) for the sequence."""
        # Implement logic to get MANO side for the sequence
        return "right"

    def _get_frame(self, seq_path):
        """Retrieve the frame."""
        files = os.listdir(seq_path)

        for filename in files:
            # Check if the filename has the required pattern
            if any(keyword in filename for keyword in ['giver_frame', 'receiver_frame', 'object_frame']):
                # Extract the frame part from the filename (e.g., frame488)
                frame_part = filename.split('_')[-1].split('.')[0]
                return frame_part

        return None

    def _get_mano_betas(self, seq_path, role, frame):
        """Retrieve MANO betas for the sequence."""
        # Form the role-specific transformed path
        if role == "giver":
            transformed_path = f"giver/{frame}"
        elif role == "receiver":
            transformed_path = f"receiver/{frame}"
        else:
            return [] 
        
        return self._mano_betas.get(transformed_path, [])

# Factory method to create dataset instances
_sets = {}
split = 'test' # set split to train if using training data
name = 'hoh_{}'.format(split)
_sets[name] = (lambda split=split: lambda handtype: HoHDataset(name, 'data/HOH/filtered', handtype))

def get_dataset(name, handtype):
    """Gets a dataset by name.

    Args:
        name: Dataset name. E.g., 'hoh_test'.
        handtype: The type of hand ('giver' or 'receiver').

    Returns:
        A dataset.

    Raises:
        KeyError: If name is not supported.
    """
    if name not in _sets:
        raise KeyError('Unknown dataset name: {}'.format(name))
    return _sets[name]()(handtype)

#Test the dataset
#dataset = get_dataset('hoh_test', 'giver')
#sample = dataset[1]
#print(sample)
