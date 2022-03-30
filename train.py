import os
import sys
import time
import logging
from collections import namedtuple

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
import megengine.autodiff as autodiff
import megengine.optimizer as optim

import yaml
from tensorboardX import SummaryWriter

from nets import Model
from dataset import CREStereoDataset
from megengine.data import DataLoader, RandomSampler, Infinite


def parse_yaml(file_path: str) -> namedtuple:
    """Parse yaml configuration file and return the object in `namedtuple`."""
    with open(file_path, "rb") as f:
        cfg: dict = yaml.safe_load(f)
    args = namedtuple("train_args", cfg.keys())(*cfg.values())
    return args


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def adjust_learning_rate(optimizer, epoch):

    warm_up = 0.02
    const_range = 0.6
    min_lr_rate = 0.05

    if epoch <= args.n_total_epoch * warm_up:
        lr = (1 - min_lr_rate) * args.base_lr / (
            args.n_total_epoch * warm_up
        ) * epoch + min_lr_rate * args.base_lr
    elif args.n_total_epoch * warm_up < epoch <= args.n_total_epoch * const_range:
        lr = args.base_lr
    else:
        lr = (min_lr_rate - 1) * args.base_lr / (
            (1 - const_range) * args.n_total_epoch
        ) * epoch + (1 - min_lr_rate * const_range) / (1 - const_range) * args.base_lr

    optimizer.param_groups[0]["lr"] = lr


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8):

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = F.abs(flow_preds[i] - flow_gt)
        flow_loss += i_weight * (F.expand_dims(valid, axis=1) * i_loss).mean()

    return flow_loss


def main(args):
    # initial info
    mge.random.seed(args.seed)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    mge.dtr.enable()  # Dynamic tensor rematerialization for memory optimization

    # directory check
    log_model_dir = os.path.join(args.log_dir, "models")
    ensure_dir(log_model_dir)

    # model / optimizer
    model = Model(
        max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=False
    )
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    dist_callbacks = None if world_size == 1 else [dist.make_allreduce_cb("mean")]
    gm = autodiff.GradManager().attach(model.parameters(), callbacks=dist_callbacks)
    scaler = mge.amp.GradScaler() if args.mixed_precision else None

    if rank == 0:
        # tensorboard
        tb_log = SummaryWriter(os.path.join(args.log_dir, "train.events"))

        # worklog
        logging.basicConfig(level=eval(args.log_level))
        worklog = logging.getLogger("train_logger")
        worklog.propagate = False
        fileHandler = logging.FileHandler(
            os.path.join(args.log_dir, "worklog.txt"), mode="a", encoding="utf8"
        )
        formatter = logging.Formatter(
            fmt="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
        )
        fileHandler.setFormatter(formatter)
        consoleHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="\x1b[32m%(asctime)s\x1b[0m %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
        )
        consoleHandler.setFormatter(formatter)
        worklog.handlers = [fileHandler, consoleHandler]

        # params stat
        worklog.info(f"Use {world_size} GPU(s)")
        worklog.info("Params: %s" % sum([p.size for p in model.parameters()]))

    # load pretrained model if exist
    chk_path = os.path.join(log_model_dir, "latest.mge")
    if args.loadmodel is not None:
        chk_path = args.loadmodel
    elif not os.path.exists(chk_path):
        chk_path = None

    if chk_path is not None:
        if rank == 0:
            worklog.info(f"loading model: {chk_path}")

        pretrained_dict = mge.load(chk_path, map_location="cpu")
        resume_epoch_idx = pretrained_dict["epoch"]
        resume_iters = pretrained_dict["iters"]
        model.load_state_dict(pretrained_dict["state_dict"], strict=True)
        optimizer.load_state_dict(pretrained_dict["optim_state_dict"])
        start_epoch_idx = resume_epoch_idx + 1
        start_iters = resume_iters
    else:
        start_epoch_idx = 1
        start_iters = 0

    # auxiliary
    if world_size > 1:
        dist.bcast_list_(model.tensors())

    # datasets
    dataset = CREStereoDataset(args.training_data_path)
    if rank == 0:
        worklog.info(f"Dataset size: {len(dataset)}")
    inf_sampler = Infinite(
        RandomSampler(
            dataset,
            batch_size=args.batch_size_single,
            drop_last=False,
            world_size=world_size,
            rank=rank,
            seed=args.seed,
        )
    )
    dataloader = DataLoader(
        dataset, sampler=inf_sampler, num_workers=0, divide=False, preload=True
    )

    # counter
    cur_iters = start_iters
    total_iters = args.minibatch_per_epoch * args.n_total_epoch
    t0 = time.perf_counter()
    for epoch_idx in range(start_epoch_idx, args.n_total_epoch + 1):

        # adjust learning rate
        epoch_total_train_loss = 0
        adjust_learning_rate(optimizer, epoch_idx)
        model.train()

        t1 = time.perf_counter()
        batch_idx = 0

        for mini_batch_data in dataloader:

            if batch_idx % args.minibatch_per_epoch == 0 and batch_idx != 0:
                break
            batch_idx += 1
            cur_iters += 1

            # parse data
            left, right, gt_disp, valid_mask = (
                mini_batch_data["left"],
                mini_batch_data["right"],
                mini_batch_data["disparity"],
                mini_batch_data["mask"],
            )

            t2 = time.perf_counter()

            with gm:  # GradManager
                with mge.amp.autocast(enabled=args.mixed_precision):

                    # pre-process
                    left = mge.tensor(left)
                    right = mge.tensor(right)
                    gt_disp = mge.tensor(gt_disp)
                    valid_mask = mge.tensor(valid_mask)
                    gt_disp = F.expand_dims(gt_disp, axis=1)
                    gt_flow = F.concat([gt_disp, gt_disp * 0], axis=1)

                    # forward
                    flow_predictions = model(left, right)

                    # loss & backword
                    loss = sequence_loss(
                        flow_predictions, gt_flow, valid_mask, gamma=0.8
                    )
                    if args.mixed_precision:
                        scaler.backward(gm, loss)
                    else:
                        gm.backward(loss)
                    optimizer.step().clear_grad()

            # loss stats
            loss_item = loss.item()
            epoch_total_train_loss += loss_item
            t3 = time.perf_counter()

            # terminal print log
            if rank == 0:
                if cur_iters % 5 == 0:
                    tdata = t2 - t1
                    time_train_passed = t3 - t0
                    time_iter_passed = t3 - t1
                    step_passed = cur_iters - start_iters
                    eta = (
                        (total_iters - cur_iters)
                        / max(step_passed, 1e-7)
                        * time_train_passed
                    )

                    meta_info = list()
                    meta_info.append("{:.2g} b/s".format(1.0 / time_iter_passed))
                    meta_info.append("passed:{}".format(format_time(time_train_passed)))
                    meta_info.append("eta:{}".format(format_time(eta)))
                    meta_info.append(
                        "data_time:{:.2g}".format(tdata / time_iter_passed)
                    )
                    meta_info.append(
                        "lr:{:.5g}".format(optimizer.param_groups[0]["lr"])
                    )
                    meta_info.append(
                        "[{}/{}:{}/{}]".format(
                            epoch_idx,
                            args.n_total_epoch,
                            batch_idx,
                            args.minibatch_per_epoch,
                        )
                    )
                    loss_info = [" ==> {}:{:.4g}".format("loss", loss_item)]
                    # exp_name = ['\n' + os.path.basename(os.getcwd())]

                    info = [",".join(meta_info)] + loss_info
                    worklog.info("".join(info))

                    # minibatch loss
                    tb_log.add_scalar("train/loss_batch", loss_item, cur_iters)
                    tb_log.add_scalar(
                        "train/lr", optimizer.param_groups[0]["lr"], cur_iters
                    )
                    tb_log.flush()

            t1 = time.perf_counter()

        if rank == 0:
            # epoch loss
            tb_log.add_scalar(
                "train/loss",
                epoch_total_train_loss / args.minibatch_per_epoch,
                epoch_idx,
            )
            tb_log.flush()

            # save model params
            ckp_data = {
                "epoch": epoch_idx,
                "iters": cur_iters,
                "batch_size": args.batch_size_single * args.nr_gpus,
                "epoch_size": args.minibatch_per_epoch,
                "train_loss": epoch_total_train_loss / args.minibatch_per_epoch,
                "state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
            }
            mge.save(ckp_data, os.path.join(log_model_dir, "latest.mge"))
            if epoch_idx % args.model_save_freq_epoch == 0:
                save_path = os.path.join(log_model_dir, "epoch-%d.mge" % epoch_idx)
                worklog.info(f"Model params saved: {save_path}")
                mge.save(ckp_data, save_path)

    if rank == 0:
        worklog.info("Training is done, exit.")


if __name__ == "__main__":
    # train configuration
    args = parse_yaml("cfgs/train.yaml")

    # distributed training
    run = main if mge.get_device_count("gpu") == 1 else dist.launcher(main)
    run(args)
