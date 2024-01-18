import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Training
trainer_arg = add_argument_group('Train')
trainer_arg.add_argument('--train_pipeline', type=str, default='LOGG3D')
trainer_arg.add_argument('--resume_training', type=str2bool, default=False)
trainer_arg.add_argument('--resume_checkpoint', type=str, default='')

# Batch setting
trainer_arg.add_argument('--batch_size', type=int, default=1) # Batch size is limited to 1.
trainer_arg.add_argument('--train_num_workers', type=int,
                         default=2)  # per gpu in dist. try 8
trainer_arg.add_argument('--subset_size', type=int, default=-1)

# Contrastive
trainer_arg.add_argument('--train_loss_function',
                         type=str, default='quadruplet')  # quadruplet, triplet
trainer_arg.add_argument('--lazy_loss', type=str2bool, default=False)
trainer_arg.add_argument('--ignore_zero_loss', type=str2bool, default=False)
trainer_arg.add_argument('--positives_per_query', type=int, default=2)  # 2
trainer_arg.add_argument('--negatives_per_query', type=int, default=9)  # 2-18
trainer_arg.add_argument('--loss_margin_1', type=float, default=0.5)  # 0.5
trainer_arg.add_argument('--loss_margin_2', type=float, default=0.3)  # 0.3

# Point Contrastive
trainer_arg.add_argument('--point_loss_function', type=str,
                         default='contrastive')  # infonce, contrastive
trainer_arg.add_argument('--point_neg_margin', type=float, default=2.0)  # 1.4
trainer_arg.add_argument('--point_pos_margin', type=float, default=0.1)  # 0.1
trainer_arg.add_argument('--point_neg_weight', type=float, default=1.0)
trainer_arg.add_argument('--point_loss_weight', type=float, default=1.0)  # 0.1
trainer_arg.add_argument('--scene_loss_weight', type=float, default=1.0)  # 0.1

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='adam')  # 'sgd','adam'
opt_arg.add_argument('--max_epoch', type=int, default=50)  # 20
opt_arg.add_argument('--base_learning_rate', type=float, default=1e-3)
opt_arg.add_argument('--momentum', type=float, default=0.8)  # 0.9
opt_arg.add_argument('--scheduler', type=str,
                     default='multistep')  # cosine#multistep

# Dataset specific configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str,
                      default='HeliprPointSparseTupleDataset')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument('--num_points', type=int, default=35000)
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument("--gp_rem", type=str2bool,
                      default=True, help="Remove ground plane.")
data_arg.add_argument("--pnv_preprocessing", type=str2bool,
                      default=False, help="Preprocessing in dataloader for PNV.")

# HeLiPR
data_arg.add_argument('--helipr_dir', type=str,
                      default='/Dataset/', help="Path to the HeLiPR dataset")
data_arg.add_argument("--helipr_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--helipr_tp_json', type=str,
                      default='positive_sequence_D-7.5_T-0.json')
data_arg.add_argument('--helipr_fp_json', type=str,
                      default='positive_sequence_D-20_T-0.json')
data_arg.add_argument('--helipr_data_split', type=dict, default={
    'train': [
        'DCC04/Aeva', 'DCC04/Ouster',
        'DCC05/Aeva', 'DCC05/Avia', 'DCC05/Ouster', 'DCC05/Velodyne',
        'KAIST04/Aeva', 'KAIST04/Ouster',
        'KAIST05/Aeva', 'KAIST05/Avia', 'KAIST05/Ouster', 'KAIST05/Velodyne',
        'Riverside04/Aeva', 'Riverside04/Ouster',
        'Riverside05/Aeva', 'Riverside05/Avia', 'Riverside05/Ouster', 'Riverside05/Velodyne',
    ],
    'val': [],
    'test': [
        'Bridge02/Aeva', 'Bridge02/Ouster',
        'Bridge03/Aeva', 'Bridge03/Avia', 'Bridge03/Ouster', 'Bridge03/Velodyne',
        'Roundabout01/Aeva', 'Roundabout01/Ouster',
        'Roundabout03/Aeva', 'Roundabout03/Avia', 'Roundabout03/Ouster', 'Roundabout03/Velodyne',
        'Town01/Aeva', 'Town01/Ouster',
        'Town03/Aeva', 'Town03/Avia', 'Town03/Ouster', 'Town03/Velodyne',
    ]
})

# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--train_pickles', type=dict, default={
    'new_dataset': "/path/to/new_dataset/training_both_5_50.pickle",
})
data_arg.add_argument('--gp_vals', type=dict, default={
    'apollo': 1.6, 'kitti':1.5, 'mulran':0.9
})
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=True)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=0.8)
data_arg.add_argument('--max_scale', type=float, default=1.2)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--experiment_name', type=str, default='run')
misc_arg.add_argument('--job_id', type=str, default='0')
misc_arg.add_argument('--save_model_after_epoch', type=str2bool, default=True)
misc_arg.add_argument('--eval_model_after_epoch', type=str2bool, default=False)
misc_arg.add_argument('--out_dir', type=str, default='logs')
misc_arg.add_argument('--loss_log_step', type=int, default=10)
misc_arg.add_argument('--checkpoint_epoch_step', type=int, default=3)


def get_config():
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = get_config()
    dconfig = vars(cfg)
    print(dconfig)
