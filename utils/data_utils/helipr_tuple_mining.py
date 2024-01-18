import sys
import os
import json
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.train_config import get_config
cfg = get_config()


def p_dist(pose1, pose2, threshold=3):
    dist = np.linalg.norm(pose1 - pose2)
    if abs(dist) <= threshold:
        return True
    else:
        return False


def t_dist(t1, t2, threshold=10):
    if abs(t1-t2) > threshold:
        return True
    else:
        return False


def get_positive_dict(basedir, sequences, output_dir, d_thresh, t_thresh):
    positive_dict = {}
    print('d_thresh: ', d_thresh)
    print('output_dir: ', output_dir)
    print('')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # sequence is a string like 'DCC04/Aeva'
    # seq_dict is a dictionary with keys like 'DCC04' and values like ['Aeva', 'Ouster']
    seq_dict = {}
    for sequence in sequences:
        parent_seq, child_seq = sequence.split('/')
        if parent_seq not in seq_dict:
            seq_dict[parent_seq] = []
        seq_dict[parent_seq].append(child_seq)
    

    for parent_seq in seq_dict:
        print(parent_seq)
        scan_positions, scan_timestamps = [], []
        for child_seq in seq_dict[parent_seq]:
            with open(os.path.join(basedir, parent_seq, child_seq, 'scan_poses.csv'), newline='') as f:
                reader = csv.reader(f)
                scan_poses = list(reader)

            for scan_pose in scan_poses:
                scan_position = [float(scan_pose[4]), float(
                    scan_pose[8]), float(scan_pose[12])]
                scan_positions.append(np.asarray(scan_position))
                scan_time = int(scan_pose[0])
                scan_timestamps.append(scan_time)
            
        if parent_seq not in positive_dict:
            positive_dict[parent_seq] = {}

        for t1 in tqdm(range(len(scan_timestamps))):
            for t2 in range(len(scan_timestamps)):
                if p_dist(scan_positions[t1], scan_positions[t2], d_thresh) & t_dist(scan_timestamps[t1], scan_timestamps[t2], t_thresh):
                    if t1 not in positive_dict[parent_seq]:
                        positive_dict[parent_seq][t1] = []
                    positive_dict[parent_seq][t1].append(t2)
    

    # for sequence in sequences:
    #     print(sequence)
    #     with open(basedir + sequence + '/scan_poses.csv', newline='') as f:
    #         reader = csv.reader(f)
    #         scan_poses = list(reader)
    #     scan_positions, scan_timestamps = [], []

    #     for scan_pose in scan_poses:
    #         scan_position = [float(scan_pose[4]), float(
    #             scan_pose[8]), float(scan_pose[12])]
    #         scan_positions.append(np.asarray(scan_position))
    #         scan_time = int(scan_pose[0])
    #         scan_timestamps.append(scan_time)

    #     if sequence not in positive_dict:
    #         positive_dict[sequence] = {}

    #     for t1 in tqdm(range(len(scan_timestamps))):
    #         for t2 in range(len(scan_timestamps)):
    #             if p_dist(scan_positions[t1], scan_positions[t2], d_thresh) & t_dist(scan_timestamps[t1], scan_timestamps[t2], t_thresh):
    #                 if t1 not in positive_dict[sequence]:
    #                     positive_dict[sequence][t1] = []
    #                 positive_dict[sequence][t1].append(t2)


    save_file_name = '{}/positive_sequence_D-{}_T-{}.json'.format(
        output_dir, d_thresh, t_thresh)
    with open(save_file_name, 'w') as f:
        json.dump(positive_dict, f)
    print('Saved: ', save_file_name)

    return positive_dict


#####################################################################################
if __name__ == "__main__":
    import csv

    basedir = cfg.helipr_dir
    sequences = [
        # TRAIN
        'DCC04/Aeva', 'DCC04/Ouster',
        'DCC05/Aeva', 'DCC05/Avia', 'DCC05/Ouster', 'DCC05/Velodyne',
        'KAIST04/Aeva', 'KAIST04/Ouster',
        'KAIST05/Aeva', 'KAIST05/Avia', 'KAIST05/Ouster', 'KAIST05/Velodyne',
        'Riverside04/Aeva', 'Riverside04/Ouster',
        'Riverside05/Aeva', 'Riverside05/Avia', 'Riverside05/Ouster', 'Riverside05/Velodyne',
        
        # TEST
        # 'Bridge02/Aeva', 'Bridge02/Ouster',
        # 'Bridge03/Aeva', 'Bridge03/Avia', 'Bridge03/Ouster', 'Bridge03/Velodyne',
        # 'Roundabout01/Aeva', 'Roundabout01/Ouster',
        # 'Roundabout03/Aeva', 'Roundabout03/Avia', 'Roundabout03/Ouster', 'Roundabout03/Velodyne',
        # 'Town01/Aeva', 'Town01/Ouster',
        # 'Town03/Aeva', 'Town03/Avia', 'Town03/Ouster', 'Town03/Velodyne',   
    ]

    
    output_dir = os.path.join(os.path.dirname(
        __file__), '../../config/helipr_tuples/')

    t_thresh = 0
    # get_positive_dict(basedir, sequences, output_dir, 3, t_thresh)
    get_positive_dict(basedir, sequences, output_dir, 7.5, t_thresh)
    # get_positive_dict(basedir, sequences, output_dir, 20, t_thresh)
