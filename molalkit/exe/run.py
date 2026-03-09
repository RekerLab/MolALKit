#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import torch
torch.cuda.is_available()
import os
import shutil
import numpy as np
import pandas as pd
from molalkit.active_learning.learner import ActiveLearner
from molalkit.data.utils import get_subset_from_uidx
from molalkit.exe.args import LearningArgs


def _free_init_memory(args):
    """Free memory from data loading caches and redundant dataset references."""
    try:
        from chemprop.data.data import (
            empty_cache, set_cache_graph, set_cache_mol, set_cache_features
        )
        empty_cache()
        set_cache_graph(False)
        set_cache_mol(False)
        set_cache_features(False)
    except ImportError:
        pass
    for attr in ("_datasets_full", "_datasets_empty"):
        if hasattr(args, attr):
            delattr(args, attr)
    gc.collect()


def _split_pool_uidx(datasets_pool, n_subsets, seed=0):
    """Split pool dataset uidx list into n equal-sized subsets.

    Returns a list of n lists of uidx values.
    """
    all_uidx = [data.uidx for data in datasets_pool[0]]
    rng = np.random.RandomState(seed)
    rng.shuffle(all_uidx)
    return [arr.tolist() for arr in np.array_split(all_uidx, n_subsets)]


def _build_pool_datasets(uidx_list, id2datapoints, datasets_pool):
    """Build pool datasets from a list of uidx values.

    Uses the first element of datasets_pool as a template (for copy structure).
    """
    return [get_subset_from_uidx(ds, id2dp, uidx_list)
            for ds, id2dp in zip(datasets_pool, id2datapoints)]


def _should_stop_early(active_learner, uncertainty_cutoff):
    """Check if the latest selected sample's uncertainty is below the threshold.

    Returns True if AL should stop on the current pool subset.
    """
    if uncertainty_cutoff is None:
        return False
    results = active_learner.active_learning_traj.results
    if not results:
        return False
    latest = results[-1]
    if not latest.acquisition_select:
        return False
    # acquisition_select contains uncertainties of selected samples;
    # check the minimum (the least uncertain selected sample)
    min_acquisition = min(latest.acquisition_select)
    return min_acquisition < uncertainty_cutoff


def _run_al_loop(active_learner, args, start_iter, max_iter):
    """Run the standard AL loop. Returns the iteration count reached and whether early-stopped."""
    logger = args.logger
    for i in range(start_iter, max_iter):
        logger.info("Active learning loop %d" % i)
        for _ in range(args.n_select):
            active_learner.step_select()
            logger.debug("Select step %d" % _)
        # Check early stopping for sequential-pool
        if args.n_pool_subsets is not None and _should_stop_early(active_learner, args.sp_uncertainty_cutoff):
            logger.info("Early stopping: uncertainty below threshold %.4f at iteration %d"
                        % (args.sp_uncertainty_cutoff, i))
            if args.evaluate_stride is not None:
                active_learner.evaluate()
            active_learner.write_traj()
            return i + 1, True
        if args.f_min_train_size is None or len(active_learner.datasets_train[0]) >= args.f_min_train_size:
            for _ in range(args.n_forget):
                active_learner.step_forget()
                logger.debug("Forget step %d" % _)
        if args.evaluate_stride is not None and i % args.evaluate_stride == 0:
            active_learner.evaluate()
            logger.debug("Evaluate step")
        if i % args.write_traj_stride == 0:
            active_learner.write_traj()
        if args.save_cpt_stride is not None and i % args.save_cpt_stride == 0:
            active_learner.current_iter = i + 1
            active_learner.save(path=args.save_dir, filename="al_temp.pkl", overwrite=True)
            shutil.move(os.path.join(args.save_dir, "al_temp.pkl"), os.path.join(args.save_dir, "al.pkl"))
            logger.info("Save checkpoint file %s/al.pkl" % args.save_dir)
        # Stop if pool is exhausted
        if len(active_learner.datasets_pool[0]) == 0:
            logger.info("Pool exhausted at iteration %d" % i)
            return i + 1, True
    return max_iter, False


def molalkit_run(arguments=None):
    args = LearningArgs().parse_args(arguments)
    logger = args.logger
    if args.load_checkpoint and os.path.exists("%s/al.pkl" % args.save_dir):
        logger.info("Restart active learning from checkpoint file %s/al.pkl" % args.save_dir)
        active_learner = ActiveLearner.load(path=args.save_dir)
        current_iter = active_learner.current_iter
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
        current_iter = 0
        active_learner.evaluate()
    _free_init_memory(args)

    if args.n_pool_subsets is not None:
        # Sequential-pool active learning
        pool_uidx_subsets = _split_pool_uidx(active_learner.datasets_pool, args.n_pool_subsets, seed=args.seed)
        # Store reference datasets_pool templates for building subsets
        template_pools = active_learner.datasets_pool
        max_iter_per_pool = args.max_iter or 100
        # Collect remaining (unselected) uidx across all subsets
        all_remaining_uidx = []

        for pool_idx, uidx_subset in enumerate(pool_uidx_subsets):
            logger.info("Sequential-pool: starting pool subset %d/%d (size=%d)"
                        % (pool_idx + 1, args.n_pool_subsets, len(uidx_subset)))
            subset_pools = _build_pool_datasets(uidx_subset, args.id2datapoints, template_pools)
            active_learner.set_pool(subset_pools)
            active_learner.model_fitted = False
            current_iter, _ = _run_al_loop(active_learner, args, current_iter, current_iter + max_iter_per_pool)
            # Collect remaining pool uidx from this subset
            all_remaining_uidx.extend([data.uidx for data in active_learner.datasets_pool[0]])
        # Add uidx from unprocessed subsets (subsets never reached due to iteration limits are
        # already handled since we always iterate all subsets above)
        # Set final pool to all remaining data for correct pool_end.csv output
        final_pools = _build_pool_datasets(all_remaining_uidx, args.id2datapoints, template_pools)
        active_learner.set_pool(final_pools)
        active_learner.write_traj()
    else:
        # Standard (static-pool) active learning
        _run_al_loop(active_learner, args, current_iter, args.max_iter or 100)

    df = pd.read_csv(f"{args.save_dir}/full.csv")
    df[df["uidx"].isin([data.uidx for data in active_learner.datasets_train[0]])].to_csv(f"{args.save_dir}/train_end.csv", index=False)
    df[df["uidx"].isin([data.uidx for data in active_learner.datasets_pool[0]])].to_csv(f"{args.save_dir}/pool_end.csv", index=False)
