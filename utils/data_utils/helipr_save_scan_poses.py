# Based on submap-extraction tools from: https://sites.google.com/view/mulran-pr/tool

import glob
import os
import numpy as np
import csv
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.eval_config import get_config_eval
cfg = get_config_eval()
basedir = cfg.helipr_dir


def findNnPoseUsingTime(target_time, all_times, data_poses):
    time_diff = np.abs(all_times - target_time)
    nn_idx = np.argmin(time_diff)
    return data_poses[nn_idx]

sequences = [
             'DCC04/Aeva', 'DCC04/Ouster',
             'DCC05/Aeva', 'DCC05/Avia', 'DCC05/Ouster', 'DCC05/Velodyne',
             'KAIST04/Aeva', 'KAIST04/Ouster',
             'KAIST05/Aeva', 'KAIST05/Avia', 'KAIST05/Ouster', 'KAIST05/Velodyne',
             'Riverside04/Aeva', 'Riverside04/Ouster',
             'Riverside05/Aeva', 'Riverside05/Avia', 'Riverside05/Ouster', 'Riverside05/Velodyne',
            
             'Bridge02/Aeva', 'Bridge02/Ouster',
             'Bridge03/Aeva', 'Bridge03/Avia', 'Bridge03/Ouster', 'Bridge03/Velodyne',
             'Roundabout01/Aeva', 'Roundabout01/Ouster',
             'Roundabout03/Aeva', 'Roundabout03/Avia', 'Roundabout03/Ouster', 'Roundabout03/Velodyne',
             'Town01/Aeva', 'Town01/Ouster',
             'Town03/Aeva', 'Town03/Avia', 'Town03/Ouster', 'Town03/Velodyne',
             
             ]

for sequence in sequences:
    sequence_path = os.path.join(basedir, sequence, 'LiDAR')
    scan_names = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))

    with open(basedir + sequence + '/global_pose.csv', newline='') as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    data_poses_ts = np.asarray([int(t) for t in np.asarray(data_poses)[:, 0]])

    for scan_name in scan_names:
        scan_time = int(scan_name.split('/')[-1].split('.')[0])
        scan_pose = findNnPoseUsingTime(scan_time, data_poses_ts, data_poses)
        with open(basedir + sequence + '/scan_poses.csv', 'a', newline='') as csvfile:
            posewriter = csv.writer(
                csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            posewriter.writerow(scan_pose)
