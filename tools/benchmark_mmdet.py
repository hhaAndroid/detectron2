#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging

from detectron2.config import LazyConfig
from torch.nn.parallel import DistributedDataParallel

from detectron2.config import instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetFromList,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("detectron2")

import os
import subprocess
import time

import torch
import torch.multiprocessing as mp
from torch import distributed as dist


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.SOLVER.BASE_LR = 0.001  # Avoid NaNs. Not useful in this script anyway.
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(distributed_rank=comm.get_rank())
    return cfg


@torch.no_grad()
def benchmark_eval(args, dist):
    cfg = setup(args)
    if args.config_file.endswith(".yaml"):
        model = build_model(cfg)
        # DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    else:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(args.ckpt)

        cfg.dataloader.num_workers = 0
        data_loader = instantiate(cfg.dataloader.test)

    if dist:
        model = DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )

    model.eval()
    logger.info("Model:\n{}".format(model))
    # define ckpt
    DetectionCheckpointer(model).load(args.ckpt)

    dummy_data = DatasetFromList(list(itertools.islice(data_loader, args.max_iter)), copy=False)

    def f():
        while True:
            yield from dummy_data

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with 2000 image and take the average
    for i, d in enumerate(f()):
        # for x in d:
        #     x["image"]=x["image"].cuda()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(d)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {args.max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == args.max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.2f} img / s, '
                f'times per image: {1000 / fps:.2f} ms / img',
                flush=True)
            break


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", choices=["train", "eval", "data"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    assert not args.eval_only

    logger.info("Environment info:\n" + collect_env_info())

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher)

    # only benchmark single-GPU inference.
    assert args.num_gpus == 1 and args.num_machines == 1
    # launch(f, args.num_gpus, args.num_machines, args.machine_rank, args.dist_url, args=(args,))
    benchmark_eval(args, distributed)
