from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_acdc import BaseDataSets as ACDC_dataset
from datasets.dataset_drive import DriveDataset, DriveTileDataset
from datasets.dataset_chasedb import ChaseDBDataset, ChaseDBTileDataset
from datasets.dataset_hrf import HRF_dataset

dataset_config = {
    'ACDC': {
        'Dataset': ACDC_dataset,  # datasets.dataset_acdc.BaseDataSets,
        'root_path': 'data/ACDC',
        'volume_path': 'data/ACDC',
        'list_dir': None,
        'num_classes': 4,
        'z_spacing': 5,
        'info': '3D'
    },
    'Synapse': {
        'Dataset': Synapse_dataset,
        'root_path': 'data/Synapse/train_npz',
        'volume_path': 'data/Synapse/test_vol_h5',
        'list_dir': './lists/lists_Synapse',
        'num_classes': 9,
        'z_spacing': 1,
    },
    'DRIVE': {
        'Dataset': DriveTileDataset,
        'tile': True,  # Whether to use tiling for DRIVE dataset
        'root_path': 'data/DRIVE',
        'volume_path': 'data/DRIVE',
        'list_dir': None,
        'num_classes': 2,
        'z_spacing': 1,
        # 'loss_name': 'dice_ce',
        'loss_name': 'vessel_fg_fov',
    },
    'CHASEDB': {
        'Dataset': ChaseDBTileDataset,
        'tile': True,  # Whether to use tiling for CHASEDB dataset
        'root_path': 'data/CHASEDB',
        'volume_path': 'data/CHASEDB',
        'list_dir': None,
        'num_classes': 2,
        'z_spacing': 1,
        # 'loss_name': 'dice_ce',
        'loss_name': 'vessel_fg',
    },
    'HRF': {
        'Dataset': HRF_dataset,
        'root_path': 'data/HRF',
        'volume_path': 'data/HRF',
        'list_dir': None,
        'num_classes': 2,
        'z_spacing': 1,
        'loss_name': 'vessel',
    },
}
