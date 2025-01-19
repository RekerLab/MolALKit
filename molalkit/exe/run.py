#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import pandas as pd
from molalkit.active_learning.learner import ActiveLearner
from molalkit.exe.args import LearningArgs


def molalkit_run(arguments=None):
    args = LearningArgs().parse_args(arguments)
    logger = args.logger
    if args.load_checkpoint and os.path.exists("%s/al.pkl" % args.save_dir):
        logger.info("Restart active learning from checkpoint file %s/al.pkl" % args.save_dir)
        active_learner = ActiveLearner.load(path=args.save_dir)
        current_loop = active_learner.current_loop
    else:
        logger.info("Start active learning from scratch")
        active_learner = ActiveLearner(
            save_dir=args.save_dir,
            selector=args.selector,
            forgetter=args.forgetter,
            models=args.models,
            id2datapoints=args.id2datapoints,
            datasets_train=args.datasets_train,
            datasets_pool=args.datasets_pool,
            datasets_val=args.datasets_val,
            metrics=args.metrics,
            top_uidx=args.top_uidx,
            kernel=args.kernels[0],
            detail=args.detail,
        )
        current_loop = 0
        active_learner.evaluate()
    for i in range(current_loop, args.max_iter or 100):
        logger.info("Active learning loop %d" % i)
        for _ in range(args.n_select):
            active_learner.step_select()
            logger.debug("Select step %d" % _)
        for _ in range(args.n_forget):
            active_learner.step_forget()
            logger.debug("Forget step %d" % _)
        if args.evaluate_stride is not None and i % args.evaluate_stride == 0:
            active_learner.evaluate()
            logger.debug("Evaluate step")
        if i % args.write_traj_stride == 0:
            active_learner.write_traj()
        if args.save_cpt_stride is not None and i % args.save_cpt_stride == 0:
            active_learner.current_loop = i + 1
            active_learner.save(path=args.save_dir, filename="al_temp.pkl", overwrite=True)
            shutil.move(os.path.join(args.save_dir, "al_temp.pkl"), os.path.join(args.save_dir, "al.pkl"))
            logger.info("Save checkpoint file %s/al.pkl" % args.save_dir)
    df = pd.read_csv(f"{args.save_dir}/full.csv")
    df[df["uidx"].isin([data.uidx for data in active_learner.datasets_train[0]])].to_csv(f"{args.save_dir}/train_end.csv", index=False)
    df[df["uidx"].isin([data.uidx for data in active_learner.datasets_pool[0]])].to_csv(f"{args.save_dir}/pool_end.csv", index=False)
