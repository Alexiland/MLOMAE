import argparse
import datetime
import json
import numpy as np
import os
import time
import math
import sys
from typing import Iterable
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
sys.path.append('/home/[path]')
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

import model_mae_new as models_mae_mlo_ddp
import model_vit_new as models_vit_mlo_ddp

import torch.distributed as dist

# from engine_pretrain import train_one_epoch
from util.pos_embed import get_2d_sincos_pos_embed

TRAIN_ITERS = 10000




def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./out',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--valid_step', default=625, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--unroll_steps_pretrain', type=int, default=2)
    parser.add_argument('--unroll_steps_finetune', type=int, default=1)
    parser.add_argument('--unroll_steps_mask', type=int, default=1)
    parser.add_argument('--freq_report', type=int, default=10)

    parser.add_argument('--finetune_lr', type=float, default=5e-4)
    parser.add_argument('--base_finetune_lr', type=float, default=5e-4)
    parser.add_argument('--finetune_batchsize', type=int, default=64)
    parser.add_argument('--finetune_weight_decay', type=float, default=0.05)
    parser.add_argument('--masking_lr', type=float, default=5e-5)
    parser.add_argument('--base_masking_lr', type=float, default=5e-5)
    parser.add_argument('--masking_batchsize', type=int, default=64)
    parser.add_argument('--masking_weight_decay', type=float, default=0.05)

    parser.add_argument('--best_acc', type=float, default=0)

    # mlo specific args
    parser.add_argument('--strategy', type=str, default='default')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--grad_clip', action='store_true')

    # additional setting
    parser.add_argument('--wandb_project', type=str, default='MAE_single_image_overfitting')
    parser.add_argument('--wandb_run_name', type=str, default='single_image_overfitting_2')
    parser.add_argument('--wandb_mode', type=str, default='disabled')

    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
#    parser.add_argument('--local_rank', type=int)
#     parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
#    torch.cuda.set_device(args.local_rank)
#    args.local_rank = int(os.environ["LOCAL_RANK"])
#    device = torch.device("cuda", args.local_rank)

    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_finetune = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_mask = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_train)

#    dataset_finetune = dataset_train
#    dataset_mask = dataset_train

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_finetune = torch.utils.data.DataLoader(
        dataset_finetune,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_mask = torch.utils.data.DataLoader(
        dataset_mask,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.model == "mae_vit_large_patch16":
        embed_dim = 1024
    elif args.model == "mae_vit_base_patch16":
        embed_dim = 768
    else:
        embed_dim = 768
        
    print(dataset_mask)
    
#    test_dataloader = data_loader_mask
    test_dataloader = torch.utils.data.DataLoader(
#        dataset_mask[:10000,:10000],
        dataset_mask,
        shuffle=False,
        batch_size=args.batch_size*36,
#        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    masking_module = models_mae_mlo_ddp.MLOPatchMasking(in_features= 14 * 14 * embed_dim)
    mask_optimizer = torch.optim.AdamW(masking_module.parameters(), lr=args.masking_lr, betas=(0.9, 0.95))

    finetune_module = models_vit_mlo_ddp.FinetuneVisionTransformer(embed_dim=embed_dim)
    finetune_optimizer = torch.optim.AdamW(finetune_module.parameters(), lr=args.finetune_lr, betas=(0.9, 0.95))

    # define the model
    model = models_mae_mlo_ddp.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    # model.to(device)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')

            # Load the saved states
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
#            args=checkpoint['args']
#            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # Repeat for any other components like finetune_module, masking_module if they exist
            if 'finetune_module' in checkpoint:
                finetune_module.load_state_dict(checkpoint['finetune_module'])
            if 'masking_module' in checkpoint:
                masking_module.load_state_dict(checkpoint['masking_module'])

            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    print("Model = %s" % str(model))
    
    
    TRAIN_ITERS = (args.epochs-args.start_epoch) * len(dataset_train) // args.batch_size // args.world_size * args.unroll_steps_pretrain * \
                  args.unroll_steps_finetune*args.unroll_steps_mask
    print(f"TRAIN_ITERS {TRAIN_ITERS}")
    args.valid_step = len(dataset_train) // args.batch_size // args.world_size * args.unroll_steps_pretrain * \
                  args.unroll_steps_finetune*args.unroll_steps_mask

    ITERS_PER_EPOCH = int(
        (len(dataset_train) * args.train_portion // args.batch_size + 1)) \
                      * args.unroll_steps_pretrain \
                      * args.unroll_steps_finetune \
                      * args.unroll_steps_mask


    eff_batch_size = args.batch_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    args.finetune_lr = args.base_finetune_lr * eff_batch_size / 256
    args.masking_lr = args.base_masking_lr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=TRAIN_ITERS, )
    lr_scheduler2 = CosineAnnealingLR(optimizer=finetune_optimizer, T_max=TRAIN_ITERS, )
    lr_scheduler3 = CosineAnnealingLR(optimizer=mask_optimizer, T_max=TRAIN_ITERS, )

    # optimizer.zero_grad()

    class Pretraining(ImplicitProblem):
        def training_step(self, batch):
            inputs, targets = batch
            inputs = inputs
            
            x = self.module.module.patch_embed(inputs)
            x = x + self.module.module.pos_embed[:, 1:, :]
            # x, mask, ids_restore, mask_prob = self.module.mask(x, args.mask_ratio, self.mask.module, random=args.baseline)
            x, mask, ids_restore, mask_prob = self.mask.module.module.forward(x, args.mask_ratio, random=args.baseline)
            # pred and target is of shape [B, N, C * P]
            pred = self.module.forward(x, mask, ids_restore)
            target = self.module.module.patchify(inputs)
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
            if self.is_rank_zero():
                wandb.log({'pretrain loss': loss})
            return loss

    class Finetuning(ImplicitProblem):
        def training_step(self, batch):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_embed, _, _ = self.pretrain.module.module.forward_encoder(inputs)
            # x = self.pretrain.module.unpatchify(inputs_embed)
            outs = self.forward(inputs_embed[:, 0])
            # outs = self.forward(inputs_embed.mean(dim=1))
            loss = F.cross_entropy(outs, targets)
            if self.is_rank_zero():
                wandb.log({'finetune loss': loss})
            return loss

    class Masking(ImplicitProblem):
        def training_step(self, batch):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_embed,_,_ = self.pretrain.module.module.forward_encoder(inputs)
            outs = self.finetune.module(inputs_embed[:, 0])
            # outs = self.finetune(inputs_embed.mean(dim=1))
            loss = F.cross_entropy(outs, targets)
            if self.is_rank_zero():
                wandb.log({'masking loss': loss})
            return loss

    class MaskEngine(Engine):
        @torch.no_grad()
        def validation(self):
        
#            # Accumulate all predictions and targets first
#            all_preds = []
#            all_targets = []
#            self.logger.info("f[Validation]: enter validation")
#            for x, target in test_dataloader:
#                x, target = x.to(device), target.to(device)
#                outs = self.finetune.module(x[:,0])
#                all_preds.append(outs)
#                all_targets.append(target)
#
#            # Concatenate all predictions and targets
#            all_preds = torch.cat(all_preds)
#            all_targets = torch.cat(all_targets)
#
#            # Compute accuracy in one go
#            correct = (all_preds.argmax(dim=1) == all_targets).sum().item()
#            total = all_targets.size(0)
#            acc = correct / total * 100

            correct = 0
            total = 0
            self.logger.info("f[Validation]: enter validation")
            for x, target in test_dataloader:
                x, target = x.to(device), target.to(device)
                # pred and target is of shape [B, N, C * P]
#                with torch.no_grad():
                x, _, _ = self.pretrain.module.module.forward_encoder(x)
                outs = self.finetune.module(x[:,0])
                # outs = self.finetune.module(x.mean(dim=1))

                correct += (outs.argmax(dim=1) == target).sum().item()
                total += x.size(0)
            acc = correct / total * 100
            
            
            
            
            
            self.logger.info("f[Validation]: after acc is calculated")
            #update best accuracy if the new accuracy is greater than the previous accuracy
            if args.best_acc < acc:
                args.best_acc = acc
            epochs = calculate_epoch(self.global_step, ITERS_PER_EPOCH)
            self.logger.info("f[Validation]: after calculate_epoch call")
#            dist.barrier()
            self.logger.info("f[Validation]: after barrier call")
            if not args.baseline:
                if args.rank == 0:
                    self.logger.info("f[Validation]: enter save model call")
                    save_model(
                        args=args, model=model, finetune_module=finetune_module, masking_module=masking_module,
                        optimizer=optimizer, epoch=epochs)
            else:
                save_model(
                    args=args, model=model, finetune_module=None, masking_module=None,
                    optimizer=optimizer, epoch=epochs)
            self.logger.info("f[Validation]: after model is saved")
            print('acc:', acc, 'best_acc:', args.best_acc)
        def run(self):
            """
            Execute multilevel optimization by running gradient descent for leaf problems.
            """
            self.train()
            if args.resume and self.global_step==0:
                self.global_step = args.start_epoch * len(dataset_train) // args.batch_size // args.world_size * args.unroll_steps_pretrain * \
                  args.unroll_steps_finetune*args.unroll_steps_mask
            start_iter = self.global_step
            for it in range(start_iter, self.train_iters + 1):
                self.global_step += 1
                self.train_step()

                dist.barrier()
                if self.global_step % 1000 == 0:
                    epochs = calculate_epoch(self.global_step, ITERS_PER_EPOCH)
                    save_model_iters(
                    args=args, model=model, finetune_module=finetune_module, masking_module=masking_module,
                    optimizer=optimizer, epoch=epochs, iters=self.global_step)

                if it % self.valid_step == 0 and self.do_validation():
                    self.eval()
                    validation_stats = self.validation() or {}
                    log_loss = log_from_loss_dict(validation_stats)
                    self.logger.info(
                        f"[Validation] [Global Step {self.global_step}] " f"{log_loss}"
                    )
                    self.logger.log(
                        validation_stats, tag="validation", step=self.global_step
                    )
                    self.train()

                    # early stopping
                    if self.early_stopping is not None:
                        stop = self.early_stopping(validation_stats)
                        if stop:
                            self.logger.info("Early stopping is executed!")
                            break

            self.cleanup()

        def initialize(self):
            """
            Initialize dependencies (computational graph) between problems.
            """
            # Parse config
            self.parse_config()

            # initialize distributed training
            dist_dict = self.configure_systems()

            if dist_dict["rank"] == 0:
                wandb.login(key="")
                wandb.init(project=args.wandb_project, name=args.wandb_run_name, mode=args.wandb_mode)

            # initialize logger
            self.logger = logger(logger_type=self.logger_type)
            if self.is_rank_zero():
                self.logger.info("Initializing Multilevel Optimization...\n")
            start = time.time()

            # parse problem dependency
            self.parse_dependency()

            # set problem attributes
            for problem in self.problems:
                self.set_problem_attr(problem)

            # env initialization
            if self.env is not None:
                self.env.configure_distributed_training(dist_dict)
                self.env.configure_device(self.device)
                self.env.initialize()

            # problem initialization
            for problem in self.problems:
                problem.add_logger(self.logger)
                problem.configure_distributed_training(dist_dict)
                problem.configure_device(self.device)
                problem.configure_roll_back(self._roll_back)
                problem.initialize()
                if self.env is not None:
                    problem.add_env(self.env)

            end = time.time()
            if self.is_rank_zero():
                self.logger.info(f"Time spent on initialization: {end - start:.3f} (s)\n")

        def configure_systems(self):
            """
            Configure basic systems set-up like distributed training and device placement.
            """
            # configure distributed training
            if self._strategy in ["distributed", "zero", "fsdp"]:
                dist.init_process_group(backend=self._backend, timeout=datetime.timedelta(seconds=64000))

                self._world_size = dist.get_world_size()
                assert self._world_size > 1
                self._rank = dist.get_rank()

                device_count = torch.cuda.device_count()
                self._local_rank = self._rank % device_count

            dist_dict = {}
            dist_dict["strategy"] = self._strategy
            dist_dict["backend"] = self._backend
            dist_dict["world_size"] = self._world_size
            dist_dict["rank"] = self._rank
            dist_dict["local_rank"] = self._local_rank

            args.rank = dist_dict["rank"]

            # configure device for the current rank
            if self._strategy in ["distributed", "zero", "fsdp"]:
                torch.cuda.set_device(self._local_rank)
                self.device = torch.device("cuda", self._local_rank)
            elif self._strategy == "accelerate":
                self.device = self.accelerator.device
            elif self._strategy == "cpu":
                self.device = "cpu"
            elif self._strategy == "gpu":
                self.device = "cuda"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.dist_dict = dist_dict

            return dist_dict

    engine_config = EngineConfig(
        strategy="distributed",
        train_iters=TRAIN_ITERS,
        valid_step=args.valid_step,
        logger_type='tensorboard'
#        valid_step=20,
        # roll_back=True
    )

    pretrain_config = Config(
        type="darts",
#        type="sama",
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
        gradient_accumulation=args.accum_iter,
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def save_model(args, epoch, model, finetune_module, masking_module, optimizer):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]

    print(f"save_model: args.rank: {args.rank}, output_dir:{output_dir}")
    for checkpoint_path in checkpoint_paths:
        if finetune_module is not None and masking_module is not None:
            to_save = {
                'model': model.state_dict(),
                'finetune_module': finetune_module.state_dict(),
                'masking_module': masking_module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
        else:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
        if args.rank == 0:
            print("f[save_model]: saving model")
            torch.save(to_save, checkpoint_path)

def save_model_iters(args, epoch, model, finetune_module, masking_module, optimizer, iters):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_path = os.path.join(output_dir, f'checkpoint-{epoch_name}-{iters}-iters.pth') 
    print(f"save_model: args.rank: {args.rank}, output_dir:{output_dir}")
    if finetune_module is not None and masking_module is not None:
        to_save = {
            'model': model.state_dict(),
            'finetune_module': finetune_module.state_dict(),
            'masking_module': masking_module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }
    else:
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }
    if args.rank == 0:
        torch.save(to_save, checkpoint_path)

def calculate_epoch(total_iters, iters_per_epoch):
    return total_iters // iters_per_epoch

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
