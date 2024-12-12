# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import utils.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model, build_criterion

import wandb
import os

def get_args_parser():
    """
    Creates and returns the argument parser for the AGILE3D model training script.

    Returns:
        argparse.ArgumentParser: The argument parser with all arguments configured.
    """
    parser = argparse.ArgumentParser('AGILE3D', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset_mode', default='multi_obj', help='Dataset mode, either single_obj or multi_obj')
    parser.add_argument('--scan_folder', default='data/ScanNet/scans', type=str, help='Path to the scan folder')
    parser.add_argument('--train_list', default='data/ScanNet/train_list.json', type=str, help='Path to the training list')
    parser.add_argument('--val_list', default='data/ScanNet/val_list.json', type=str, help='Path to the validation list')

    # Model parameters
    ## Backbone
    parser.add_argument('--dialations', default=[1, 1, 1, 1], type=list, help='Dialation rates for the backbone')
    parser.add_argument('--conv1_kernel_size', default=5, type=int, help='Kernel size for the first convolutional layer')
    parser.add_argument('--bn_momentum', default=0.02, type=int, help='Batch normalization momentum')
    parser.add_argument('--voxel_size', default=0.05, type=float, help='Voxel size for point cloud data')

    ## Transformer
    parser.add_argument('--hidden_dim', default=128, type=int, help='Hidden dimension size for transformer layers')
    parser.add_argument('--dim_feedforward', default=1024, type=int, help='Feedforward dimension size in transformer')
    parser.add_argument('--num_heads', default=8, type=int, help='Number of attention heads in the transformer')
    parser.add_argument('--num_decoders', default=3, type=int, help='Number of decoder layers in the transformer')
    parser.add_argument('--num_bg_queries', default=10, type=int, help='Number of learnable background queries')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate in transformer layers')
    parser.add_argument('--pre_norm', default=False, type=bool, help='Whether to use pre-norm in transformer layers')
    parser.add_argument('--normalize_pos_enc', default=True, type=bool, help='Normalize positional encoding')
    parser.add_argument('--positional_encoding_type', default="fourier", type=str, help='Type of positional encoding to use')
    parser.add_argument('--gauss_scale', default=1.0, type=float, help='Gauss scale for positional encoding')
    parser.add_argument('--hlevels', default=[4], type=list, help='Hierarchy levels for feature extraction')
    parser.add_argument('--shared_decoder', default=False, type=bool, help='Whether to use shared decoder layers')

    # Loss parameters
    parser.add_argument('--losses', default=['bce', 'dice'], type=list, help='List of loss functions to use')
    parser.add_argument('--bce_loss_coef', default=1.0, type=float, help='Coefficient for BCE loss')
    parser.add_argument('--dice_loss_coef', default=2.0, type=float, help='Coefficient for Dice loss')
    parser.add_argument('--aux', default=True, type=bool, help='Use auxiliary loss')

    # Training parameters
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay for optimization')
    parser.add_argument('--lr_drop', default=[1000], type=int, nargs='+', help='Epochs to drop learning rate')
    parser.add_argument('--epochs', default=1100, type=int, help='Total number of training epochs')
    parser.add_argument('--val_epochs', default=50, type=int, help='Validation frequency in epochs')
    parser.add_argument('--batch_size', default=5, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Validation batch size')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='Max gradient norm for clipping')
    parser.add_argument('--device', default='cuda', help='Device to use for training/testing')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', default='output', help='Output directory for saving results')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch number')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for data loading')
    parser.add_argument('--resume', default='', help='Path to resume training from checkpoint')
    parser.add_argument('--max_num_clicks', default=20, type=int, help='Maximum number of clicks per object on average')
    parser.add_argument('--job_name', default='test', type=str, help='Name of the job for logging')

    # Add support for gradient accumulation
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Steps for gradient accumulation')

    return parser


def main(args):

    # setup wandb for logging
    utils.setup_wandb()
    wandb.init(project="AGILE3D",settings=wandb.Settings(start_method="fork"))
    wandb.run.name = args.run_id
    
    device = torch.device(args.device)

    # Something to fiddle with for speed
    torch.set_float32_matmul_precision('medium')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    criterion = build_criterion(args)
    model.to(device)
    criterion.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build dataset and dataloader
    dataset_train, collation_fn_train = build_dataset(split='train', args=args)
    dataset_val, collation_fn_val = build_dataset(split='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train,
                                   collate_fn=collation_fn_train, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.val_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collation_fn_val, num_workers=args.num_workers,
                                 pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = False
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

        test_stats = evaluate(
            model, criterion, data_loader_val, args, args.start_epoch, device
        )

        wandb.log({
            "val/epoch": args.start_epoch,
            "val/loss_epoch": test_stats['loss'],
            "val/loss_bce_epoch": test_stats['loss_bce'],
            "val/loss_dice_epoch": test_stats['loss_dice'],
            "val/mIoU_epoch": test_stats['mIoU'],

            "val_metrics/NoC_50": test_stats['NoC@50'],
            "val_metrics/NoC_65": test_stats['NoC@65'],
            "val_metrics/NoC_80": test_stats['NoC@80'],
            "val_metrics/NoC_85": test_stats['NoC@85'],
            "val_metrics/NoC_90": test_stats['NoC@90'],
            "val_metrics/IoU_1": test_stats['IoU@1'],
            "val_metrics/IoU_3": test_stats['IoU@3'],
            "val_metrics/IoU_5": test_stats['IoU@5'],
            "val_metrics/IoU_10": test_stats['IoU@10'],
            "val_metrics/IoU_15": test_stats['IoU@15'],
            })

    print("Start training")

    train_total_iter = 0
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):

        train_stats, train_total_iter = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, train_total_iter, args.clip_max_norm, args.accumulation_steps)
        
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 20 epochs
            if (epoch + 1) in args.lr_drop or (epoch + 1) % 20 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if (epoch + 1) % args.val_epochs == 0:
            test_stats = evaluate(
                model, criterion, data_loader_val, args, epoch, device
            )

        wandb.log({"lr_rate": train_stats['lr']})
        wandb.log({
                "train/epoch": epoch,
                "train/loss_epoch": train_stats['loss'],
                "train/loss_bce_epoch": train_stats['loss_bce'],
                "train/loss_dice_epoch": train_stats['loss_dice'],
                "train/mIoU_epoch": train_stats['mIoU']
                })

        if (epoch + 1) % args.val_epochs == 0:

            wandb.log({
                    "val/epoch": epoch,
                    "val/loss_epoch": test_stats['loss'],
                    "val/loss_bce_epoch": test_stats['loss_bce'],
                    "val/loss_dice_epoch": test_stats['loss_dice'],
                    "val/mIoU_epoch": test_stats['mIoU'],

                    "val_metrics/NoC_50": test_stats['NoC@50'],
                    "val_metrics/NoC_65": test_stats['NoC@65'],
                    "val_metrics/NoC_80": test_stats['NoC@80'],
                    "val_metrics/NoC_85": test_stats['NoC@85'],
                    "val_metrics/NoC_90": test_stats['NoC@90'],
                    "val_metrics/IoU_1": test_stats['IoU@1'],
                    "val_metrics/IoU_3": test_stats['IoU@3'],
                    "val_metrics/IoU_5": test_stats['IoU@5'],
                    "val_metrics/IoU_10": test_stats['IoU@10'],
                    "val_metrics/IoU_15": test_stats['IoU@15']

                    })

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AGILE3D training script', parents=[get_args_parser()])
    args = parser.parse_args()
    now = datetime.datetime.now()
    run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    args.run_id = run_id + '_' + args.job_name
    args.output_dir = os.path.join(args.output_dir, run_id)
    args.valResults_dir = os.path.join(args.output_dir, 'valResults')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.valResults_dir).mkdir(parents=True, exist_ok=True)

    main(args)