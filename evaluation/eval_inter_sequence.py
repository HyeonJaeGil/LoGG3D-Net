from scipy.spatial.distance import cdist
import logging
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
from termcolor import colored
from tqdm import tqdm
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.pipelines.pipeline_utils import *
from utils.data_loaders.make_dataloader import *
from utils.misc_utils import *
from utils.data_loaders.helipr.helipr_dataset import load_poses_from_csv, load_timestamps_csv
__all__ = ['evaluate_inter_sequence']


def check_nearby(pose1, pose2, threshold=3):
    return True if abs(np.linalg.norm(pose1 - pose2)) <= threshold else False


def check_longtime(t1, t2, threshold=10):
    return True if abs(t1-t2) > threshold else False


def check_revisit(q_scan_position: np.ndarray, q_scan_timestamp: float,
                db_scan_positions: list, db_scan_timestamps: list,
                d_thresh: float, t_thresh: float):
    is_revisit = False
    for t2 in range(len(db_scan_timestamps)):
        if check_nearby(q_scan_position, db_scan_positions[t2], d_thresh) & \
        check_longtime(q_scan_timestamp, db_scan_timestamps[t2], t_thresh):
            is_revisit = True
            # print(colored((f'revisit index: {t2:5d}'), 'green'), end='\r')
            break
    return is_revisit


def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, 'ab')
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()
    logging.debug(f'Finished saving: {file_name}')


def load_pickle(file_name):
    dbfile = open(file_name, 'rb')
    data_variable = pickle.load(dbfile)
    dbfile.close()
    logging.debug(f'Finished loading: {file_name}')
    return data_variable


def parse_sequence(cfg, type):
    assert type in ['query', 'database'], 'type must be either query or database'
    eval_seq = cfg.eval_seq_q if type == 'query' else cfg.eval_seq_db
    sequence_path = [cfg.helipr_dir + seq for seq in eval_seq]
    positions, timestamps = [], []
    for seq in sequence_path:
        positions.append(load_poses_from_csv(seq + '/scan_poses.csv')[1])
        timestamps.append(load_timestamps_csv(seq + '/scan_poses.csv'))
    positions = np.concatenate(positions, axis=0)
    timestamps = np.concatenate(timestamps, axis=0)
    cfg.helipr_data_split['test_q' if type == 'query' else 'test_db'] = eval_seq
    test_loader = make_eval_dataloader(cfg, 'test_q' if type == 'query' else 'test_db')
    iterator = test_loader.__iter__()
    logging.debug(f'length of {type} dataloader {len(test_loader.dataset)}')
    return positions, timestamps, iterator


def extract_global_descriptors(model, iterator, cfg):
    descriptors = []
    for idx in tqdm(range(len(iterator))):
        input_data = next(iterator)
        lidar_pc = input_data[0][0]
        if not len(lidar_pc) > 0:
            logging.debug(f'Corrupt cloud id: {idx}')
            descriptors.append(None)
            continue
        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
        output_desc, _ = model(input)
        global_descriptor = output_desc.cpu().detach().numpy()
        global_descriptor = np.reshape(global_descriptor, (1, -1))
        if len(global_descriptor) < 1:
            logging.debug(f'Corrupt descriptor id: {idx}')
            descriptors.append(None)
            continue
        descriptors.append(global_descriptor)
    return descriptors


def evaluate_inter_sequence(model, cfg, save_dir=None):

    ##### parse query sequences #####
    positions_query, timestamps_query, iterator_query = parse_sequence(cfg, 'query')

    ##### parse database sequences #####
    positions_database, timestamps_database, iterator_database = parse_sequence(cfg, 'database')
    
    ##### Save descriptors and features for Query #####
    descriptors_query = extract_global_descriptors(model, iterator_query, cfg)

    ##### Save descriptors and features for Database #####
    descriptors_database = extract_global_descriptors(model, iterator_database, cfg)

    ##### Delete corrupt clouds #####
    q_idx_to_delete = [idx for idx, desc in enumerate(descriptors_query) if desc is None]    
    positions_query = [pos for idx, pos in enumerate(positions_query) if idx not in q_idx_to_delete]
    timestamps_query = [ts for idx, ts in enumerate(timestamps_query) if idx not in q_idx_to_delete]
    descriptors_query = [desc for idx, desc in enumerate(descriptors_query) if idx not in q_idx_to_delete]

    db_idx_to_delete = [idx for idx, desc in enumerate(descriptors_database) if desc is None]
    positions_database = [pos for idx, pos in enumerate(positions_database) if idx not in db_idx_to_delete]
    timestamps_database = [ts for idx, ts in enumerate(timestamps_database) if idx not in db_idx_to_delete]
    descriptors_database = [desc for idx, desc in enumerate(descriptors_database) if idx not in db_idx_to_delete]

    # # OR load pickle
    # positions_query = load_pickle('/home/hj/Research/LoGG3D-Net/evaluation/Town03-Ouster_vs_Town01-Ouster/logg3d_poses_q.pickle')
    # timestamps_query = load_pickle('/home/hj/Research/LoGG3D-Net/evaluation/Town03-Ouster_vs_Town01-Ouster/logg3d_timestamps_q.pickle')
    # descriptors_query = load_pickle('/home/hj/Research/LoGG3D-Net/evaluation/Town03-Ouster_vs_Town01-Ouster/logg3d_descriptor_q.pickle')
    
    # positions_database = load_pickle('/home/hj/Research/LoGG3D-Net/evaluation/Town03-Ouster_vs_Town01-Ouster/logg3d_poses_db.pickle')
    # timestamps_database = load_pickle('/home/hj/Research/LoGG3D-Net/evaluation/Town03-Ouster_vs_Town01-Ouster/logg3d_timestamps_db.pickle')
    # descriptors_database = load_pickle('/home/hj/Research/LoGG3D-Net/evaluation/Town03-Ouster_vs_Town01-Ouster/logg3d_descriptor_db.pickle')

    ##### Evaluate #####

    # set varying thresholds
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))
    num_thresholds = len(thresholds)

    # variables for recall N
    num_correct_recall_1 = 0
    num_correct_recall_1p = 0
    one_percent = int(len(positions_query) * 0.01)

    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0
    num_correct_loc = 0

    # Find top-1 candidate.
    for q_idx in range(len(positions_query)):
        feat_dists = cdist(descriptors_query[q_idx], np.concatenate(descriptors_database, axis=0),
                            metric=cfg.eval_feature_distance).reshape(-1)
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)

        is_nearby = check_nearby(positions_query[q_idx], positions_database[nearest_idx], 
                                 cfg.revisit_criteria)
        is_revisit = check_revisit(positions_query[q_idx], timestamps_query[q_idx],
                                      positions_database, timestamps_database, 
                                      cfg.revisit_criteria, cfg.skip_time)
        is_correct_loc = 0
        if is_revisit:
            num_revisits += 1
            if is_nearby:
                num_correct_loc += 1
                is_correct_loc = 1

        if is_revisit:
            if is_nearby:
                num_correct_recall_1 += 1
            top_1p_indices = np.argsort(feat_dists)[:one_percent]
            top_1p_dist = [check_nearby(positions_query[q_idx], positions_database[idx], 
                                    cfg.revisit_criteria) for idx in top_1p_indices]
            if sum(top_1p_dist) > 0:
                num_correct_recall_1p += 1

        # info_string = (f'id: {q_idx:4d} n_id: {nearest_idx:4d} ' 
        #                f'q_is_revisit: {is_revisit:1d}  top1_is_nearby: {is_nearby:1d} ' 
        #                f'is_correct_loc: {is_correct_loc:1d} ' 
        #                f'min_dist: {min_dist:8f}')
        
        # if is_revisit and is_nearby: # Found correct Top 1
        #     logging.debug(colored(info_string, 'green'))
        # elif is_revisit and not is_nearby: # Not found correct Top 1
        #     logging.debug(colored(info_string, 'red'))
        # elif not is_revisit and not is_nearby: # first time visit, so not found a revisit
        #     logging.debug(colored(info_string, 'blue'))
        # elif not is_revisit and is_nearby:  # Weird: first time visit, but found a revisit
        #     print('[ERROR] top1 candidate is very nearby, but not a revisit')
        #     logging.debug(colored(info_string, 'yellow'))
        #     exit(0)

        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist

        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]
            if(min_dist < threshold):  # Positive Prediction
                if is_nearby:
                    num_true_positive[thres_idx] += 1
                elif not is_nearby:
                    num_false_positive[thres_idx] += 1
            else:  # Negative Prediction
                if is_revisit:
                    num_false_negative[thres_idx] += 1
                else:
                    num_true_negative[thres_idx] += 1


    ##### Compute precision, recall, and F1. #####
    F1max = 0.0
    Precisions, Recalls = [], []
    for ithThres in range(num_thresholds):
        nTrueNegative = num_true_negative[ithThres]
        nFalsePositive = num_false_positive[ithThres]
        nTruePositive = num_true_positive[ithThres]
        nFalseNegative = num_false_negative[ithThres]

        Precision = 0.0
        Recall = 0.0
        F1 = 0.0

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)
            F1 = 2 * Precision * Recall * (1/(Precision + Recall))

        if F1 > F1max:
            F1max = F1
            F1_TN = nTrueNegative
            F1_FP = nFalsePositive
            F1_TP = nTruePositive
            F1_FN = nFalseNegative
            F1_thresh_id = ithThres
        Precisions.append(Precision)
        Recalls.append(Recall)
    logging.debug(f'num_revisits: {num_revisits}')
    logging.debug(f'num_correct_loc: {num_correct_loc}')
    logging.debug(
        f'percentage_correct_loc: {num_correct_loc*100.0/num_revisits}')
    logging.debug(
        f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
    logging.debug(
        f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
    logging.debug(f'F1_thresh_id: {F1_thresh_id}')
    logging.debug(f'F1max: {F1max}')
    logging.debug(f'recall@1:  {num_correct_recall_1*100.0/num_revisits}')
    logging.debug(f'recall@1%: {num_correct_recall_1p*100.0/num_revisits}')

    ##### Save results #####
    save_descriptors = cfg.eval_save_descriptors
    save_counts = cfg.eval_save_counts
    save_pr_curve = cfg.eval_save_pr_curve

    if save_descriptors or save_counts or save_pr_curve:
        eval_seq_q_save = '_'.join(cfg.eval_seq_q)
        eval_seq_db_save = '_'.join(cfg.eval_seq_db)
        eval_seq_save = eval_seq_q_save + '_vs_' + eval_seq_db_save
        eval_seq_save = eval_seq_save.replace('/', '-')
        
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), str(eval_seq_save))
        os.makedirs(save_dir, exist_ok=True)
        print('Saving results to: ', save_dir)

    if save_pr_curve:
        AUC = np.trapz(Precisions, Recalls)
        logging.debug(f'AUC: {AUC}')
        plt.title('Seq: ' + str(eval_seq_save) +
                    '    AUC: ' + "%.4f" % (AUC))
        plt.plot(Recalls, Precisions, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis([0, 1, 0, 1.1])
        plt.xticks(np.arange(0, 1.01, step=0.1))
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'prcurve.png'))

    if save_counts:
        save_pickle(num_true_positive, os.path.join(save_dir, 'num_true_positive.pickle'))
        save_pickle(num_false_positive, os.path.join(save_dir, 'num_false_positive.pickle'))
        save_pickle(num_true_negative, os.path.join(save_dir, 'num_true_negative.pickle'))
        save_pickle(num_false_negative, os.path.join(save_dir, 'num_false_negative.pickle'))

        # save as txt, for easy copy-paste
        # each line: threshold, TP, FP, TN, FN
        save_txt = os.path.join(save_dir, 'counts.txt')
        with open(save_txt, 'w') as f:
            query = '_'.join(cfg.eval_seq_q)
            database = '_'.join(cfg.eval_seq_db)
            f.write(f'# query: {query}\n')
            f.write(f'# database: {database}\n')
            f.write(f'# revisit_criteria: {cfg.revisit_criteria}\n')
            f.write(f'# distance_threshold TP  FP  TN  FN\n')
            for i in range(num_thresholds):
                f.write(f'{thresholds[i]:.4f} '
                        f'{int(num_true_positive[i]):5d} '
                        f'{int(num_false_positive[i]):5d} '
                        f'{int(num_true_negative[i]):5d} '
                        f'{int(num_false_negative[i]):5d}\n')
    
    if save_descriptors:
        save_pickle(descriptors_database, os.path.join(save_dir, 'logg3d_descriptor_db.pickle'))
        save_pickle(positions_database, os.path.join(save_dir, 'logg3d_poses_db.pickle'))
        save_pickle(timestamps_database, os.path.join(save_dir, 'logg3d_timestamps_db.pickle'))
        
        save_pickle(descriptors_query, os.path.join(save_dir, 'logg3d_descriptor_q.pickle'))
        save_pickle(positions_query, os.path.join(save_dir, 'logg3d_poses_q.pickle'))
        save_pickle(timestamps_query, os.path.join(save_dir, 'logg3d_timestamps_q.pickle'))


    output = {
        'AUC': AUC,
        'F1max': F1max,
        'recall@1': num_correct_recall_1*100.0/num_revisits,
        'recall@1%': num_correct_recall_1p*100.0/num_revisits
    }

    return output


if __name__ == "__main__":
    root = '/home/hj/Research/LoGG3D-Net/evaluation/Town03-Ouster_vs_Town01-Ouster'
    num_tp = load_pickle(os.path.join(root, 'num_true_positive.pickle'))
    num_fp = load_pickle(os.path.join(root, 'num_false_positive.pickle'))
    num_tn = load_pickle(os.path.join(root, 'num_true_negative.pickle'))
    num_fn = load_pickle(os.path.join(root, 'num_false_negative.pickle'))
    print(num_tp)
    print(num_fp)
    print(num_tn)
    print(num_fn)
