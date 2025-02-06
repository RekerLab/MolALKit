#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Literal
import copy
import pandas as pd


def get_data(data_format: Literal["mgktools", "chemprop"],
             path: str,
             smiles_columns: List[str] = None,
             targets_columns: List[str] = None,
             features_columns: List[str] = None,
             features_generators: List[str] = None,
             features_combination: Literal["concat", "mean"] = None,
             graph_kernel_type: Literal["graph", "pre-computed", "no"] = None,
             n_jobs: int = 8):
    if features_generators is not None and features_combination is None:
        features_combination = "concat"
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    if data_format == "chemprop":
        from chemprop.data.utils import get_data
        assert features_columns is None
        dataset = get_data(path=path,
                           smiles_columns=smiles_columns,
                           target_columns=targets_columns,
                           features_generator=features_generators)
    else:
        from mgktools.data.data import Dataset
        dataset = Dataset.from_df(df=df,
                                  smiles_columns=smiles_columns,
                                  features_columns=features_columns,
                                  targets_columns=targets_columns,
                                  n_jobs=n_jobs)
        dataset.set_status(graph_kernel_type=graph_kernel_type,
                           features_generators=features_generators, 
                           features_combination=features_combination)
        if graph_kernel_type in ["graph", "pre-computed"]:
            dataset.create_graphs(n_jobs=n_jobs)
            dataset.unify_datatype()
        if features_generators is not None:
            dataset.create_features_mol(n_jobs=n_jobs)
    if "uidx" not in df:
        df["uidx"] = range(len(df))
    for i, data in enumerate(dataset):
        data.uidx = df.iloc[i]["uidx"]
    return dataset


def get_subset_from_idx(dataset, idx):
    """ Create a subset of the dataset by index.

    Parameters
    ----------
    dataset : Chemprop or Mgktools dataset object
        The dataset to be subsetted.
    idx: List[int]
        index of the subset
    """
    subset = copy.deepcopy(dataset)
    subset.data = [data for i, data in enumerate(dataset.data) if i in idx]
    assert len(subset) == len(idx), "Subset length does not match the index length, indicating that some indices are not found in the parent dataset."
    return subset


def get_subset_from_uidx(dataset, id2datapoint, uidx):
    """ Create a subset of the dataset by index of a dict (id2datapoint).

    Parameters
    ----------
    dataset : Chemprop or Mgktools dataset object
        The dataset to be subsetted.
    id2datapoint: Dict
        Dictionary of id to datapoint.
    idx: List[int]
        index of the subset.
    """
    dataset_ = copy.deepcopy(dataset)
    dataset_.data = []
    for i in uidx:
        dataset_.data.append(id2datapoint[i])
    return dataset_
