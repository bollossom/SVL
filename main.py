import sys
import os
import logging
import shutil
import data
import models
import torch
from omegaconf import OmegaConf
from datetime import datetime
from param import parse_args
from utils.misc import load_config, dump_config    
from utils.logger import setup_logging
from utils.scheduler import cosine_scheduler
from train import Trainer
from models.LogitScaleNetwork import LogitScaleNetwork
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29111'
    dist.init_process_group("nccl", rank=rank, world_size=world_size) 

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, cli_args, extras):
    setup(rank, world_size)

    config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)
    if config.autoresume:
        config.trial_name = config.get('trial_name') + "@autoresume"
    else:
        config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    
    if rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
        os.makedirs(config.ckpt_dir, exist_ok=True)

    
    config.device = 'cuda:{0}'.format(rank) 
    
    if rank == 0:
        config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
        logging.info("Using {} GPU(s).".format(config.ngpu))


    if config.train:
        # ---- Build Model ----
        model = models.make(config).to(config.device)
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(model)
            logging.info("Network:{}, Number of parameters: {}".format(config.model.name, total_params))
        
        torch.cuda.set_device(rank)
        model.cuda(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if config.model.name.startswith('Mink'):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # Spconv only
            logging.info("Using MinkowskiSyncBatchNorm")
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logging.info("Using SyncBatchNorm")

        logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)
        image_proj = torch.nn.Linear(config.clip_embed_dim, config.model.out_channel).to(config.device)
        text_proj = torch.nn.Linear(config.clip_embed_dim, config.model.out_channel).to(config.device)
        print("text_image_proj",image_proj,text_proj)
        
        logit_scale = DDP(logit_scale, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        image_proj = DDP(image_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        text_proj = DDP(text_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        train_loader = data.make(config, 'train', rank, world_size)

        if rank == 0:
            modelnet40_loader = data.make_modelnet40test(config)
            objaverse_lvis_loader = data.make_objaverse_lvis(config)
            scanobjectnn_loader = data.make_scanobjectnntest(config)
        else:
            modelnet40_loader = None
            objaverse_lvis_loader = None
            scanobjectnn_loader = None

        if rank == 0:
            if train_loader is not None:
                logging.info("Train iterations: {}".format(len(train_loader)))

        params = list(model.parameters()) + list(image_proj.parameters()) + list(text_proj.parameters()) + list(logit_scale.parameters()) 

        optimizer = torch.optim.AdamW(
                params,
                lr=config.training.lr,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
                weight_decay=config.training.wd
            )
        scheduler = cosine_scheduler(config.training.lr, cli_args.lr_end, config.training.max_epoch,
            len(train_loader), warmup_epochs=config.training.warmup, start_warmup_value=cli_args.lr_start)

        trainer = Trainer(rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, \
                        train_loader, modelnet40_loader, objaverse_lvis_loader, scanobjectnn_loader)

        if config.resume is not None:
            trainer.load_from_checkpoint(config.resume)
        elif config.autoresume:
            if os.path.exists(os.path.join(config.ckpt_dir, '{}.pt'.format('latest'))):
                trainer.load_from_checkpoint(os.path.join(config.ckpt_dir, '{}.pt'.format('latest')))

        trainer.train()

    cleanup()


if __name__ == '__main__':
    cli_args, extras = parse_args(sys.argv[1:])
    world_size = cli_args.ngpu  
    mp.spawn(
        main,
        args=(world_size, cli_args, extras),
        nprocs=world_size
    )    
    # rank = 0
    # main(rank, world_size=1, cli_args=cli_args, extras=extras)  