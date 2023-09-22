import os
import numpy as np
from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmaction.utils import get_root_logger
from mmaction.datasets import build_dataset
from mmaction.models import build_recognizer
from mmaction.apis.train import train_model


def main_worker(gpu, ngpus_per_node, cfg_path, root, model, current_timestamp):
    try:
        print(f"Starting main_worker on GPU {gpu}")
        cfg = Config.fromfile(cfg_path)
        # cfg.optimizer.lr = learning_rate
        cfg.ann_file_train = os.path.join(root, 'data/posec3d', model, 'train.pkl')
        cfg.data.train.ann_file = os.path.join(root, 'data/posec3d', model, 'train.pkl')
        cfg.ann_file_val = os.path.join(root, 'data/posec3d', model, 'val.pkl')
        cfg.data.val.ann_file = os.path.join(root, 'data/posec3d', model, 'val.pkl')
        cfg.data.test.ann_file = os.path.join(root, 'data/posec3d', model, 'test.pkl')
        cfg.work_dir = os.path.join(root, 'work_dirs', f'{model}_{current_timestamp}')

        model = build_recognizer(cfg.model)
        dataset = build_dataset(cfg.data.train)
        logger = get_root_logger(log_level=cfg.get('log_level', 'INFO'))

        dist.init_process_group(backend='nccl', init_method='tcp://localhost:12345', world_size=ngpus_per_node,
                                rank=gpu)
        torch.cuda.set_device(gpu)
        os.environ['LOCAL_RANK'] = str(gpu)
        model = MMDistributedDataParallel(model.cuda(gpu), device_ids=[gpu], find_unused_parameters=True)

        train_model(
            model=model,
            dataset=dataset,
            cfg=cfg,
            distributed=True,
            validate=True,
            test=dict(test_best=True, test_last=True),
            timestamp=None,
            meta=None
        )
        print(f"Training completed for GPU {gpu}")

    except Exception as e:
        print(f"An error occurred in the main_worker function: {e}")


def distributed_training(args, cfg):
    model = args.mm_dataset
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ngpus_per_node = cfg['mm']['ngpus_per_node']
    root = os.path.join(os.getcwd(), 'models', 'mmaction2')
    cfg_path = os.path.join(root, 'configs/skeleton/posec3d/asbar.py')

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(
        ngpus_per_node, cfg_path, root, model, current_timestamp))
