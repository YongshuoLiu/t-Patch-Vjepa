import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchCAdetr import *


def save_checkpoint(path, model, optimizer, epoch, best_val_BCE, best_iter, experimentID, args):
    ckpt = {
        "epoch": epoch,
        "best_val_BCE": float(best_val_BCE),
        "best_iter": int(best_iter),
        "experimentID": int(experimentID),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "args": vars(args),  
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=True)

    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])

    epoch = int(ckpt.get("epoch", 0))
    best_val_BCE = float(ckpt.get("best_val_BCE", np.inf))
    best_iter = int(ckpt.get("best_iter", -1))
    experimentID = int(ckpt.get("experimentID", -1))
    return ckpt, epoch, best_val_BCE, best_iter, experimentID


parser = argparse.ArgumentParser('IMTS Patch-Level Classification')

# --------------------
# Run / misc
# --------------------
parser.add_argument('--state', type=str, default='def')
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

# --------------------
# Data
# --------------------
parser.add_argument('-n', type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--dataset', type=str, default='mulity_source',
                    help="Dataset to load. Available: physionet, mimic, ushcn, mulity_source")

# quantization (optional dataset preprocessing)
parser.add_argument('--quantization', type=float, default=0.0,
                    help="Quantization on timestamps (dataset-side). 0 means no quantization.")

# (optional legacy sliding-window params; DO NOT use them to compute npatch anymore)
parser.add_argument('--history', type=int, default=24,
                    help="Historical window for legacy patching; dataset-side only.")
parser.add_argument('-ps', '--patch_size', type=float, default=24,
                    help="Legacy window size for a patch; dataset-side only.")
parser.add_argument('--stride', type=float, default=24,
                    help="Legacy stride for patch sliding; dataset-side only.")

# --------------------
# Training
# --------------------
parser.add_argument('--epoch', type=int, default=200, help="training epochs")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--lr', type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=2)
parser.add_argument('--patch_minutes', type=int, default=5)

parser.add_argument('--max_patches', type=int, default=15)


# Model (NEW)
# --------------------
parser.add_argument('--model', type=str, default='tPatchGNN_Classifier', help="Model name")

# patch padding dimensions (CRITICAL for your new pipeline)
parser.add_argument('--npatch', type=int, default=15,
                    help="Max number of patches per sample (padding dimension). "
                         "batch['patch_mask'] and batch['patch_index'] must have shape (B, npatch).")
parser.add_argument('--json_path', type=str, default="/home/UNT/yl0826/QAU/t-PatchGNN/lib/test.json",
                    help="dataset: json file.")

parser.add_argument('--patch_len', type=int, default=64,
                    help="Max number of timesteps per patch after padding/truncation.")

# input feature dim (data_sequence last dim D)
parser.add_argument('--in_dim', type=int, default=4,
                    help="Input feature dimension D in data_sequence: (B, Lmax, D).")

# architecture
parser.add_argument('--nhead', type=int, default=4, help="Heads in attention modules")
parser.add_argument('--tf_layer', type=int, default=1, help="#layers inside TransformerEncoder")
parser.add_argument('--nlayer', type=int, default=2, help="#stacks of (graph-attn + self-attn)")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Hidden dimension for patch tokens")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Time encoding dimension")

# time-diff weighting for patch graph attention
parser.add_argument('--tau_seconds', type=float, default=300.0,
                    help="Decay scale (seconds) for time-diff bias in patch graph attention. "
                         "Larger -> more uniform weights across patches.")

# classification
parser.add_argument('--num_classes', type=int, default=5,
                    help="Number of anomaly classes. For your setting must be 5.")
parser.add_argument('--label_smoothing', type=float, default=0.0,
                    help="Optional label smoothing for CE loss (0.0 means none).")


parser.add_argument('--max_total_steps', type=int, default=15,
                    help="Number of max_total_steps. after padding the number time step must be 15.")


args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################


if __name__ == '__main__':

    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)

    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + ".ckpt")

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    # -------- data --------
    data_obj = parse_datasets(args, patch_ts=True)

    # -------- model --------
    model = tPatchGNN_Classifier(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -------- logger --------
    if args.n < 12000:
        args.state = "debug"
        log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
    else:
        log_path = "logs/{}_{}_{}_{}patch_{}stride_{}layer_{}lr.log".format(
            args.dataset, args.model, args.state, args.patch_size, args.stride, args.nlayer, args.lr
        )

    if not os.path.exists("logs/"):
        utils.makedirs("logs/")

    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    # -------- resume (if args.load given and file exists) --------
    best_val_BCE = np.inf
    best_iter = -1
    start_epoch = 0
    test_res = None

    if args.load is not None and os.path.exists(ckpt_path):
        _, start_epoch, best_val_BCE, best_iter, _ = load_checkpoint(
            ckpt_path, model, optimizer=optimizer, map_location=args.device
        )
        logger.info("Loaded checkpoint: {} (start_epoch={}, best_val_BCE={:.6f}, best_iter={})".format(
            ckpt_path, start_epoch, best_val_BCE, best_iter
        ))
    else:
        logger.info("No checkpoint loaded. Training from scratch. Will save to: {}".format(ckpt_path))

    # -------- train loop --------
    num_batches = data_obj["n_train_batches"]
    print("n_train_batches:", num_batches)

    for itr in range(start_epoch, args.epoch):
        st = time.time()

        # --- Training ---
        model.train()
        last_train_loss = None
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

            train_res = compute_classfaction_losses(model, batch_dict)

            loss = train_res["loss"]
            loss.backward()
            optimizer.step()

            last_train_loss = loss

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])

            # --- Save best + test ---
            if val_res["BCE"] < best_val_BCE:
                best_val_BCE = val_res["BCE"]
                best_iter = itr


                test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])


                if not os.path.exists(args.save):
                    utils.makedirs(args.save)
                save_checkpoint(
                    ckpt_path, model, optimizer,
                    epoch=itr,
                    best_val_BCE=best_val_BCE,
                    best_iter=best_iter,
                    experimentID=experimentID,
                    args=args
                )
                logger.info("Saved checkpoint to {} (best_val_BCE={:.6f})".format(ckpt_path, best_val_BCE))

            # --- logging ---
            logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))

            if last_train_loss is not None:
                logger.info("Train - Loss (last batch): {:.5f}".format(last_train_loss.item()))

            logger.info("Val - Loss, BCE: {:.6f}".format(val_res["BCE"]))

            if test_res is not None:
                logger.info("Test - Best epoch, Loss, BCE: {}, {:.6f}".format(best_iter, test_res["BCE"]))

            logger.info("Time spent: {:.2f}s".format(time.time() - st))

        # --- early stop ---
        if best_iter >= 0 and (itr - best_iter) >= args.patience:
            logger.info("Early stopped! best_iter={}, best_val_BCE={:.6f}".format(best_iter, best_val_BCE))
            sys.exit(0)
