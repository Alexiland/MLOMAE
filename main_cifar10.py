import argparse
import datetime
import json
import numpy as np
import os
import time
import math
import sys
from typing import Iterable
import glob
import shutil
from pathlib import Path
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from betty.utils import log_from_loss_dict
from betty.logging import logger
import wandb

import torch.nn.functional as F

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import cifar_mae_model as models_mae_mlo_ddp
import cifar_vit_mlo as models_vit_mlo_ddp

import torch.distributed as dist

# from engine_pretrain import train_one_epoch
from util.pos_embed import get_2d_sincos_pos_embed

#TRAIN_ITERS = 10000
#initiate best accuracy
best_acc = -1

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
#        os.mkdir(path)
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
#        os.mkdir(os.path.join(path, 'scripts'))
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(epoch, model, optimizer, output_directory="./checkpoint", filename="checkpoint.pth.tar"):
    """Save model checkpoint."""
    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    file_path = os.path.join(output_directory, f"{filename}")
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    if epoch is None and optimizer is None:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, file_path)

    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, file_path)

path_save = './checkpoint'
create_exp_dir(path_save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(path_save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='cifar10_mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./ckpt/test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log/test',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--valid_step', default=625, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--unroll_steps_pretrain', type=int, default=2)
    parser.add_argument('--unroll_steps_finetune', type=int, default=1)
    parser.add_argument('--unroll_steps_mask', type=int, default=1)
    parser.add_argument('--freq_report', type=int, default=10)

    
    parser.add_argument('--finetune_lr', type=float, default=0.001)
    parser.add_argument('--finetune_batchsize', type=int, default=128)
    parser.add_argument('--finetune_weight_decay', type=float, default=0.001)
    parser.add_argument('--masking_lr', type=float, default=0.001)
    parser.add_argument('--masking_batchsize', type=int, default=128)
    parser.add_argument('--masking_weight_decay', type=float, default=0.001)
    # parser.add_argument('--finetune_lr', type=float, default=0.0005)
    # parser.add_argument('--finetune_batchsize', type=int, default=48)
    # parser.add_argument('--finetune_weight_decay', type=float, default=0.001)
    # parser.add_argument('--masking_lr', type=float, default=0.0001)
    # parser.add_argument('--masking_batchsize', type=int, default=48)
    # parser.add_argument('--masking_weight_decay', type=float, default=0.001)

    # mlo specific args
    parser.add_argument('--strategy', type=str, default='default')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--grad_clip', action='store_true')

    # additional setting
    parser.add_argument('--wandb_project', type=str, default='MAE_single_image_overfitting')
    parser.add_argument('--wandb_run_name', type=str, default='single_image_overfitting_2')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    # parser.add_argument('--wandb_mode', type=str, default='enabled')

    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))


    
    
    device = torch.device(args.device)
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, mode=args.wandb_mode)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    train_data = datasets.CIFAR10(args.data_path, train=True, download=True,
                                                 transform=train_transform)
    valid_data = datasets.CIFAR10(args.data_path, train=False, download=True,
                                               transform=valid_transform)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))



    data_loader_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        shuffle = True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
#        drop_last=True,
    )

    data_loader_finetune = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.finetune_batchsize,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_mask = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.masking_batchsize,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    test_dataloader = torch.utils.data.DataLoader(
        valid_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    
    # masking_module = models_mae_mlo_ddp.MaskingModule().to(device)
    masking_module = models_mae_mlo_ddp.MLOPatchMasking().to(device)
    mask_optimizer = torch.optim.AdamW(masking_module.parameters(), lr=args.masking_lr, betas=(0.9, 0.95))

    # finetune_module = models_vit_mlo_ddp.cifar10_vit_base_patch2().to(device)
    finetune_module = models_vit_mlo_ddp.FinetuneVisionTransformer().to(device)
    finetune_optimizer = torch.optim.AdamW(finetune_module.parameters(), lr=args.finetune_lr, betas=(0.9, 0.95))

    TRAIN_ITERS = int(
        args.epochs
        * (num_train * args.train_portion // args.batch_size + 1)) * args.unroll_steps_pretrain*args.unroll_steps_finetune*args.unroll_steps_mask

    # define the model
    model = models_mae_mlo_ddp.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    # print(model)
    # print(finetune_module)

    model.to(device)
    eff_batch_size = args.batch_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=TRAIN_ITERS, )

    lr_scheduler2 = CosineAnnealingLR(optimizer=finetune_optimizer, T_max=TRAIN_ITERS, )
    lr_scheduler3 = CosineAnnealingLR(optimizer=mask_optimizer, T_max=TRAIN_ITERS, )



    class Pretraining(ImplicitProblem):
        def training_step(self, batch):
            inputs, targets = batch
            inputs = inputs.to(device)
            
            x = self.module.patch_embed(inputs)
            x = x + self.module.pos_embed[:, 1:, :]
            # x, mask, ids_restore, mask_prob = self.module.mask(x, args.mask_ratio, self.mask.module, random=args.baseline)
            x, mask, ids_restore, mask_prob = self.mask(x, args.mask_ratio, random=args.baseline)
            # pred and target is of shape [B, N, C * P]
            pred = self.forward(x, mask, ids_restore)
            target = self.module.patchify(inputs)
            if args.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6) ** .5

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [B, N], mean loss per patch

            # update loss with masking probability to enable back propagation of masking module
            if not args.baseline:
                loss = loss * mask_prob

            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            wandb.log({'pretrain loss': loss})
            return loss

    class Finetuning(ImplicitProblem):
        def training_step(self, batch):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_embed,_,_ = self.pretrain.module.forward_encoder(inputs)
            # x = self.pretrain.module.unpatchify(inputs_embed)
            outs = self.forward(inputs_embed[:, 0])
            # outs = self.forward(inputs_embed.mean(dim=1))
            loss = F.cross_entropy(outs, targets)
            return loss

    class Masking(ImplicitProblem):
        def training_step(self, batch):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_embed,_,_ = self.pretrain.module.forward_encoder(inputs)
            outs = self.finetune(inputs_embed[:, 0])
            # outs = self.finetune(inputs_embed.mean(dim=1))
            loss = F.cross_entropy(outs, targets)
            wandb.log({'masking loss': loss})
            return loss




    
    class MaskEngine(Engine):
        @torch.no_grad()
        def validation(self):
            correct = 0
            total = 0
            global best_acc
            for x, target in test_dataloader:
                x, target = x.to(device), target.to(device)
                # pred and target is of shape [B, N, C * P]
                with torch.no_grad():
                    x,_,_ = self.pretrain.module.forward_encoder(x)
                    outs = self.finetune.module(x[:,0])
                    # outs = self.finetune.module(x.mean(dim=1))
                correct += (outs.argmax(dim=1) == target).sum().item()
                total += x.size(0)
            acc = correct / total * 100
            #update best accuracy if the new accuracy is greater than the previous accuracy
            if best_acc < acc:
                best_acc = acc
            if not args.baseline:
                save_checkpoint(None, model, None, output_directory=path_save, filename="pretrain.pth")
                save_checkpoint(None, finetune_module, None, output_directory=path_save, filename="finetune.pth")
                save_checkpoint(None, masking_module, None, output_directory=path_save, filename="masking.pth")
            else:
                save_checkpoint(None, model, None, output_directory=path_save, filename="pretrain_baseline.pth")
            print('acc:', acc, 'best_acc:', best_acc)
            
    engine_config = EngineConfig(
#        strategy="distributed",
        train_iters=TRAIN_ITERS,
        valid_step=args.valid_step,
        # roll_back=True
    )

    pretrain_config = Config(
        type="darts",
#        type="sama",
#        fp16=args.fp16,
        retain_graph=True,
        log_step=args.freq_report,
        unroll_steps=args.unroll_steps_pretrain,
        # darts_preconditioned=True,
        gradient_accumulation=args.accum_iter,
        allow_unused=True
    )
    finetune_config = Config(
        type="darts",
#        type="sama",
        retain_graph=True,
#        fp16=args.fp16,
        log_step=args.freq_report,
        allow_unused=True,
        gradient_accumulation=args.accum_iter,
        unroll_steps=args.unroll_steps_finetune,
        # darts_preconditioned=True
    )
    masking_config = Config(
        type="darts",
#        type="sama",
        retain_graph=True,
        unroll_steps=args.unroll_steps_mask,
        log_step=args.freq_report,
#        fp16=args.fp16,
        gradient_accumulation=args.accum_iter,
#        first_order=True,
        allow_unused=True,
    )

    pretrain = Pretraining(
        name="pretrain",
        module=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_data_loader=data_loader_train,
        config=pretrain_config,
    )
    finetune = Finetuning(
        name="finetune",
        module=finetune_module,
        optimizer=finetune_optimizer,
        scheduler=lr_scheduler2,
        train_data_loader=data_loader_finetune,
        config=finetune_config,
    )

    mask = Masking(
        name="mask",
        module=masking_module,
        optimizer=mask_optimizer,
        scheduler=lr_scheduler3,
        train_data_loader=data_loader_mask,
        config=masking_config,
    )

    if args.baseline:
        problems = [pretrain]
        u2l, l2u = {}, {}
    else:
        print("Constructing graph dependencies")
        problems = [pretrain, finetune, mask]
        
        u2l = {mask: [pretrain]}
        # u2l = {mask: [finetune, pretrain]}
        # u2l = {mask: [finetune, pretrain], finetune: [pretrain]}
        l2u = {pretrain: [finetune, mask], finetune: [mask]}

    dependencies = {"l2u": l2u, "u2l": u2l}

    engine = MaskEngine(
        config=engine_config,
        problems=problems,
        dependencies=dependencies,
    )
    engine.run()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
