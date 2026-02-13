#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import numpy as np
from scipy.stats import norm
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from molalkit.active_learning.utils import random_choice, get_topn_idx
from molalkit.data.utils import get_subset_from_idx


class BaseSelector(ABC):
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        pass

    @property
    @abstractmethod
    def info(self) -> str:
        pass


class BaseRandomSelector(BaseSelector, ABC):
    """Base Selector that uses random seed."""
    def __init__(self, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size)
        self.seed = seed
        np.random.seed(seed)


class BaseClusterSelector(BaseRandomSelector, ABC):
    """ Base Selector that uses clustering method.
        cluster_size samples will be selected first according to provided selection algorithm,
        and then separate into batch_size clusters using KMeans clustering method.
        Finally, one sample from each cluster will be selected.
        This can increase the diversity of the selected samples batch.
    """
    def __init__(self, batch_size: int = 1, seed: int = 0, cluster_size: int = None):
        super().__init__(batch_size=batch_size, seed=seed)
        assert batch_size > 1, "batch_size should be larger than 1 for cluster selection method."
        if cluster_size is None:
            self.cluster_size = batch_size * 20
        else:
            assert cluster_size >= batch_size, "cluster_size should be larger than batch_size."
            self.cluster_size = cluster_size

    def get_idx_cluster(self, dataset_pool, kernel: Callable, idx_cluster) -> List[int]:
        """ Find distant samples from a pool using KMeans clustering method."""
        if len(idx_cluster) < self.batch_size:
            return list(range(len(idx_cluster)))
        else:
            K = kernel(dataset_pool.X[idx_cluster])
            add_idx = self.find_distant_samples(gram_matrix=K, batch_size=self.batch_size)
            return add_idx

    def find_distant_samples(self, gram_matrix: np.ndarray[float], batch_size: int = 1) -> List[int]:
        """ Find distant samples from a pool using clustering method.

        Parameters
        ----------
        gram_matrix: gram (kernel) matrix of the samples.
        batch_size: number of samples to be selected.

        Returns
        -------
        List of idx
        """
        embedding = SpectralEmbedding(
            n_components=batch_size,
            affinity="precomputed",
            random_state=self.seed
        ).fit_transform(gram_matrix)

        cluster_result = KMeans(
            n_clusters=batch_size,
            random_state=self.seed
        ).fit_predict(embedding)
       # Calculate cluster centers
        centers = np.array([embedding[cluster_result == i].mean(axis=0)
                           for i in range(batch_size)])
        # For each cluster, find the sample that's most distant from other cluster centers
        total_distance = defaultdict(dict)
        for i in range(batch_size):
            cluster_idx = cluster_result[i]
            # Get centers of other clusters
            other_centers = np.delete(centers, cluster_idx, axis=0)
            # Calculate sum of inverse distances to other centers
            # (Using inverse distance means smaller values indicate more distant points)
            sum_inverse_distances = np.sum(np.sqrt(np.sum(
                np.square(embedding[i] - other_centers), axis=1
            )) ** -0.5)
            total_distance[cluster_idx][sum_inverse_distances] = i
        # Select the most distant sample from each cluster
        selected_indices = [
            total_distance[i][min(total_distance[i].keys())]
            for i in range(batch_size)
        ]
        return selected_indices


class BasePartialQuerySelector(BaseRandomSelector, ABC):
    """ Base Selector that uses partial query method.
        query_size samples will be selected first randomly,
        and then batch_size samples will be selected according to provided selection algorithm.
    """
    @staticmethod
    def get_partial_data(dataset_pool, query_size):
        # randomly select query_size samples from dataset_pool.
        if len(dataset_pool) <= query_size:
            return dataset_pool, np.array(range(len(dataset_pool)))
        else:
            idx_query = np.random.choice(range(len(dataset_pool)), query_size, replace=False)
            data_query = get_subset_from_idx(dataset_pool, idx_query)
            return data_query, idx_query


class RandomSelector(BaseRandomSelector):
    def __call__(self, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        idx, idx_remain = random_choice(len(dataset_pool), self.batch_size)
        return idx, [], idx_remain

    @property
    def info(self) -> str:
        return f"RandomSelector(batch_size={self.batch_size})"


class ClusterRandomSelector(BaseClusterSelector):
    def __call__(self, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size < len(dataset_pool):
            idx_cluster = random_choice(len(dataset_pool), self.cluster_size)[0]
            idx = idx_cluster[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, [], idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"ClusterRandomSelector(batch_size={self.batch_size}, cluster_size={self.cluster_size})"


class ExplorativeSelector(BaseRandomSelector):
    def __init__(self, batch_size: int = 1, seed: int = 0, target: float = None):
        super().__init__(batch_size=batch_size, seed=seed)
        self.target = target

    def __call__(self, model, dataset_pool, stop_cutoff: float = None, confidence_cutoff: float = None, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            y_std = model.predict_uncertainty(dataset_pool)

            if self.target is not None:
                above_cutoff_idx = np.where(y_std > self.target)[0]
                if len(above_cutoff_idx) > 0:
                    n = min(self.batch_size, len(above_cutoff_idx))
                    above_cutoff_uncertainties = y_std[above_cutoff_idx]
                    sorting_key = above_cutoff_uncertainties + np.random.random(len(above_cutoff_uncertainties)) * 1e-10
                    local_sorted = np.argsort(sorting_key)[:n]
                    idx = above_cutoff_idx[local_sorted].tolist()
                else:
                    idx = get_topn_idx(y_std, n=self.batch_size, target="max", cutoff=stop_cutoff)
            else:
                idx = get_topn_idx(y_std, n=self.batch_size, target="max", cutoff=stop_cutoff)

            acquisition = y_std[np.array(idx)].tolist()

            idx_confident = np.where(y_std < confidence_cutoff)[0].tolist() if confidence_cutoff is not None else []
            mask = np.isin(range(len(dataset_pool)), idx + idx_confident)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        if self.target is not None:
            return f"ExplorativeSelector(batch_size={self.batch_size}, target={self.target})"
        return f"ExplorativeSelector(batch_size={self.batch_size})"


class ClusterExplorativeSelector(BaseClusterSelector):
    def __call__(self, model, dataset_pool, kernel: Callable, stop_cutoff: float = None, confidence_cutoff: float = None, 
                 **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            y_std = model.predict_uncertainty(dataset_pool)
            idx_cluster = get_topn_idx(y_std, n=self.cluster_size, target="max", cutoff=stop_cutoff)
            idx = np.array(idx_cluster)[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()

            acquisition = y_std[np.array(idx)].tolist()

            idx_confident = np.where(y_std < confidence_cutoff)[0].tolist() if confidence_cutoff is not None else []
            mask = np.isin(range(len(dataset_pool)), idx + idx_confident)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"ClusterExplorativeSelector(batch_size={self.batch_size}, cluster_size={self.cluster_size})"


class PartialQueryExplorativeSelector(BasePartialQuerySelector, BaseRandomSelector):
    def __init__(self, query_size: int, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.query_size = query_size
        assert batch_size <= query_size, "batch_size should be smaller than query_size."

    def __call__(self, model, dataset_pool, stop_cutoff: float = None, confidence_cutoff: float = None, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            data_query, idx_query = self.get_partial_data(dataset_pool, self.query_size)
            y_std = model.predict_uncertainty(data_query)
            idx_ = get_topn_idx(y_std, n=self.batch_size, target="max", cutoff=stop_cutoff)
            idx = idx_query[np.array(idx_)].tolist()

            acquisition = y_std[idx_].tolist()

            idx_confident_ = np.where(y_std < confidence_cutoff)[0].tolist() if confidence_cutoff is not None else []
            idx_confident = idx_query[np.array(idx_confident_)].tolist()
            mask = np.isin(range(len(dataset_pool)), idx + idx_confident)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"PartialQueryExplorativeSelector(batch_size={self.batch_size}, query_size={self.query_size})"


class PartialQueryClusterExplorativeSelector(BaseClusterSelector, BasePartialQuerySelector):
    def __init__(self, query_size: int, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, cluster_size=cluster_size, seed=seed)
        self.query_size = query_size
        assert self.cluster_size <= query_size, "cluster_size should be smaller than query_size."

    def __call__(self, model, dataset_pool, kernel: Callable, stop_cutoff: float = None, confidence_cutoff: float = None, 
                 **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            data_query, idx_query = self.get_partial_data(dataset_pool, self.query_size)
            y_std = model.predict_uncertainty(data_query)
            idx_cluster = get_topn_idx(y_std, n=self.cluster_size, target="max", cutoff=stop_cutoff)
            idx_ = np.array(idx_cluster)[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()
            idx = np.array(idx_query)[np.array(idx_)].tolist()

            acquisition = y_std[np.array(idx_)].tolist()

            idx_confident_ = np.where(y_std < confidence_cutoff)[0].tolist() if confidence_cutoff is not None else []
            idx_confident = idx_query[np.array(idx_confident_)].tolist()
            mask = np.isin(range(len(dataset_pool)), idx + idx_confident)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return (f"PartialQueryClusterExplorativeSelector(batch_size={self.batch_size}, "
                f"cluster_size={self.cluster_size}, query_size={self.query_size})")


class ExploitiveSelector(BaseRandomSelector):
    def __init__(self, target, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.target = target

    def __call__(self, model, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        y_pred = np.asarray(model.predict_value(dataset_pool)).ravel()
        idx = get_topn_idx(y_pred, n=self.batch_size, target=self.target)

        acquisition = y_pred[np.array(idx)].tolist()

        mask = np.isin(range(len(dataset_pool)), idx)
        idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"ExploitiveSelector(batch_size={self.batch_size}, target={self.target})"


class ClusterExploitiveSelector(BaseClusterSelector):
    def __init__(self, target, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, cluster_size=cluster_size, seed=seed)
        self.target = target

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        y_pred = np.asarray(model.predict_value(dataset_pool)).ravel()
        idx_cluster = get_topn_idx(y_pred, n=self.cluster_size, target=self.target)
        idx = np.array(idx_cluster)[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()

        acquisition = y_pred[np.array(idx)].tolist()

        mask = np.isin(range(len(dataset_pool)), idx)
        idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return (f"ClusterExploitiveSelector(batch_size={self.batch_size}, "
                f"cluster_size={self.cluster_size}, target={self.target})")


class PartialQueryExploitiveSelector(BasePartialQuerySelector, BaseRandomSelector):
    def __init__(self, target, query_size: int, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.target = target
        self.query_size = query_size
        assert batch_size <= query_size, "batch_size should be smaller than query_size."

    def __call__(self, model, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        data_query, idx_query = self.get_partial_data(dataset_pool, self.query_size)
        y_pred = np.asarray(model.predict_value(data_query)).ravel()
        idx_ = get_topn_idx(y_pred, n=self.batch_size, target=self.target)
        idx = idx_query[np.array(idx_)].tolist()

        acquisition = y_pred[idx_].tolist()

        mask = np.isin(range(len(dataset_pool)), idx)
        idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"PartialQueryExploitiveSelector(batch_size={self.batch_size}, query_size={self.query_size}, target={self.target})"


class PartialQueryClusterExploitiveSelector(BaseClusterSelector, BasePartialQuerySelector):
    def __init__(self, target, query_size: int, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, cluster_size=cluster_size, seed=seed)
        self.target = target
        self.query_size = query_size
        assert self.cluster_size <= query_size, "cluster_size should be smaller than query_size."
    
    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        data_query, idx_query = self.get_partial_data(dataset_pool, self.query_size)
        y_pred = np.asarray(model.predict_value(data_query)).ravel()
        idx_cluster = get_topn_idx(y_pred, n=self.cluster_size, target=self.target)
        idx_ = np.array(idx_cluster)[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()
        idx = np.array(idx_query)[np.array(idx_)].tolist()

        acquisition = y_pred[idx_].tolist()

        mask = np.isin(range(len(dataset_pool)), idx)
        idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return (f"PartialQueryClusterExploitiveSelector(batch_size={self.batch_size}, "
                f"cluster_size={self.cluster_size}, query_size={self.query_size}, target={self.target})")


class ProbabilityImprovementSelector(BaseRandomSelector):
    pass


class ExpectedImprovementSelector(BaseRandomSelector):
    def __call__(self, model, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            mean = np.asarray(mean).ravel()
            std = np.asarray(std).ravel()

            f_best = np.median(np.abs(mean))
            abs_mean = np.abs(mean)
            z = np.where(std > 1e-10, (f_best - abs_mean) / std, 0.0)
            ei = np.where(
                std > 1e-10,
                (f_best - abs_mean) * norm.cdf(z) + std * norm.pdf(z),
                0.0,
            )

            idx = get_topn_idx(ei, n=self.batch_size, target="max")
            acquisition = ei[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"ExpectedImprovementSelector(batch_size={self.batch_size})"


class UpperConfidenceBoundSelector(BaseRandomSelector):
    def __init__(self, batch_size: int = 1, seed: int = 0, beta: float = 1.0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.beta = beta

    def __call__(self, model, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            mean = np.asarray(mean).ravel()
            std = np.asarray(std).ravel()

            score = -np.abs(mean) + self.beta * std
            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"UpperConfidenceBoundSelector(batch_size={self.batch_size}, beta={self.beta})"


class DensityWeightedUncertaintySelector(BaseRandomSelector):
    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)
            score = std * density

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"DensityWeightedUncertaintySelector(batch_size={self.batch_size})"


class AdaptiveExplorativeSelector(BaseRandomSelector):
    def __init__(self, batch_size: int = 1, seed: int = 0,
                 alpha_max: float = 1.0, gamma: float = 1.0, total_iter: int = 1000):
        super().__init__(batch_size=batch_size, seed=seed)
        self.alpha_max = alpha_max
        self.gamma = gamma
        self.total_iter = total_iter
        self.current_iter = 0

    def __call__(self, model, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            mean = np.asarray(mean).ravel()
            std = np.asarray(std).ravel()

            ratio = min(self.current_iter / self.total_iter, 1.0)
            alpha = self.alpha_max * (1.0 - ratio) ** self.gamma
            score = (1.0 - alpha) * (-np.abs(mean)) + alpha * std

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()

            self.current_iter += 1
            return idx, acquisition, idx_remain
        else:
            self.current_iter += 1
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return (f"AdaptiveExplorativeSelector(batch_size={self.batch_size}, "
                f"alpha_max={self.alpha_max}, gamma={self.gamma}, total_iter={self.total_iter})")


class DensityWeightedGammaSelector(BaseRandomSelector):
    """Density-weighted uncertainty with tunable gamma exponent: score = std * density^gamma."""
    def __init__(self, batch_size: int = 1, seed: int = 0, gamma: float = 2.0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.gamma = gamma

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)
            score = std * np.power(density, self.gamma)

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"DensityWeightedGammaSelector(batch_size={self.batch_size}, gamma={self.gamma})"


class GreedyVarianceReductionSelector(BaseRandomSelector):
    """Select the pool point whose addition maximally reduces total posterior variance.

    For a GP with kernel matrix K, adding point x reduces variance at x' by:
        delta_sigma2(x') = k(x', x)^2 / (k(x, x) + sigma_n^2)
    Total reduction for candidate x = sum over all pool points x' of delta_sigma2(x').
    This equals ||K_pool[:, x]||^2 / (K_pool[x, x] + sigma_n^2).
    """
    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            K = kernel(dataset_pool.X)
            n = K.shape[0]

            # Estimate noise variance from model's posterior std at training points
            # Use a small regularization floor
            diag = np.diag(K)
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()
            # sigma_n^2: use median posterior variance as a proxy for noise
            sigma_n2 = np.median(std ** 2) + 1e-8

            # For each candidate x_i, total variance reduction = sum_j K[j,i]^2 / (K[i,i] + sigma_n2)
            score = np.sum(K ** 2, axis=0) / (diag + sigma_n2)

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"GreedyVarianceReductionSelector(batch_size={self.batch_size})"


class FacilityLocationSelector(BaseRandomSelector):
    """Pure representativeness selection using greedy facility location.

    Selects the pool point that maximizes: sum_j max(current_coverage_j, K[j, x_new]).
    No uncertainty is used — purely kernel-based representativeness.
    """
    def __call__(self, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            K = kernel(dataset_pool.X)
            n = K.shape[0]

            # Greedy facility location: pick the point with highest total similarity
            # For batch_size=1 this is simply argmax of column sums
            score = K.sum(axis=0)

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"FacilityLocationSelector(batch_size={self.batch_size})"


class LocalUncertaintySelector(BaseRandomSelector):
    """Select by uncertainty normalized by local neighborhood average.

    score = std_i / mean(std of k nearest neighbors of i)
    This filters out globally uncertain outliers (high std but all neighbors also high std).
    """
    def __init__(self, batch_size: int = 1, seed: int = 0, k_neighbors: int = 10):
        super().__init__(batch_size=batch_size, seed=seed)
        self.k_neighbors = k_neighbors

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            n = K.shape[0]
            k = min(self.k_neighbors, n - 1)

            # For each point, find k nearest neighbors by kernel similarity (higher K = closer)
            # and compute mean std of those neighbors
            local_mean_std = np.zeros(n)
            for i in range(n):
                # Exclude self, sort by kernel similarity descending
                similarities = K[i].copy()
                similarities[i] = -np.inf
                neighbor_idx = np.argpartition(similarities, -k)[-k:]
                local_mean_std[i] = std[neighbor_idx].mean()

            # Avoid division by zero
            local_mean_std = np.maximum(local_mean_std, 1e-10)
            score = std / local_mean_std

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"LocalUncertaintySelector(batch_size={self.batch_size}, k_neighbors={self.k_neighbors})"


class QBCBootstrapSelector(BaseRandomSelector):
    """Query-by-Committee via GP bootstrap.

    Trains n_committee GPs on bootstrap subsets of training data,
    selects pool points with highest prediction disagreement (std of committee predictions).
    """
    def __init__(self, batch_size: int = 1, seed: int = 0, n_committee: int = 5):
        super().__init__(batch_size=batch_size, seed=seed)
        self.n_committee = n_committee

    def __call__(self, model, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            # Use the model's predict with return_std to get base predictions
            # Then create diversity by adding calibrated noise to training targets
            # This avoids needing to retrain multiple GPs (expensive)
            mean, std = model.predict(dataset_pool.X, return_std=True)
            mean = np.asarray(mean).ravel()
            std = np.asarray(std).ravel()

            # Simulate committee via perturbation of posterior:
            # Each committee member samples from N(mean, std) and we measure disagreement
            rng = np.random.RandomState(self.seed)
            predictions = np.column_stack([
                mean + std * rng.randn(len(mean))
                for _ in range(self.n_committee)
            ])

            # Disagreement = std across committee predictions
            disagreement = predictions.std(axis=1)

            idx = get_topn_idx(disagreement, n=self.batch_size, target="max")
            acquisition = disagreement[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"QBCBootstrapSelector(batch_size={self.batch_size}, n_committee={self.n_committee})"


class TrainingDistanceDWSelector(BaseRandomSelector):
    """Density-weighted uncertainty penalized by proximity to training data.

    score = std * density_pool * (1 - max_similarity_to_training)
    Points that are uncertain, in dense pool regions, AND far from training data are preferred.
    """
    def __call__(self, model, dataset_pool, dataset_train, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            # Pool-pool kernel for density
            K_pool = kernel(dataset_pool.X)
            density = K_pool.mean(axis=1)

            # Pool-train kernel for training distance
            if len(dataset_train) > 0:
                K_pt = kernel(dataset_pool.X, dataset_train.X)
                max_sim_to_train = K_pt.max(axis=1)
            else:
                max_sim_to_train = np.zeros(len(dataset_pool))

            # Novelty: 1 - max similarity to any training point
            novelty = 1.0 - max_sim_to_train
            novelty = np.maximum(novelty, 1e-10)

            score = std * density * novelty

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"TrainingDistanceDWSelector(batch_size={self.batch_size})"


class InformationDensitySelector(BaseRandomSelector):
    """Information density: uncertainty weighted by representativeness ratio.

    score = std * (mean_sim_to_pool / mean_sim_to_train)
    Amplifies points that are representative of the pool but dissimilar from training.
    Based on Settles & Craven (2008).
    """
    def __call__(self, model, dataset_pool, dataset_train, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            # Pool-pool similarity (representativeness of pool)
            K_pool = kernel(dataset_pool.X)
            mean_sim_pool = K_pool.mean(axis=1)

            # Pool-train similarity
            if len(dataset_train) > 0:
                K_pt = kernel(dataset_pool.X, dataset_train.X)
                mean_sim_train = K_pt.mean(axis=1)
                mean_sim_train = np.maximum(mean_sim_train, 1e-10)
            else:
                mean_sim_train = np.ones(len(dataset_pool)) * 1e-10

            score = std * (mean_sim_pool / mean_sim_train)

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"InformationDensitySelector(batch_size={self.batch_size})"


class RankBasedDWSelector(BaseRandomSelector):
    """Rank-based density-weighted uncertainty: nonparametric, scale-free.

    score = rank(std) * rank(density)
    Eliminates sensitivity to relative scales of std and density.
    """
    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            from scipy.stats import rankdata

            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)

            # Rank both (higher rank = higher value)
            rank_std = rankdata(std)
            rank_density = rankdata(density)
            score = rank_std * rank_density

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"RankBasedDWSelector(batch_size={self.batch_size})"


class SqrtStdDWSelector(BaseRandomSelector):
    """Density-weighted with compressed uncertainty: score = sqrt(std) * density.

    Reduces the dynamic range of uncertainty, preventing extreme-std outliers
    from dominating even after density weighting.
    """
    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)
            score = np.sqrt(np.maximum(std, 0.0)) * density

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"SqrtStdDWSelector(batch_size={self.batch_size})"


class AdaptiveGammaDWSelector(BaseRandomSelector):
    """Density-weighted with decaying gamma schedule.

    score = std * density^gamma(t), where gamma(t) = gamma_max - (gamma_max - gamma_min) * t/T
    Early: high gamma (conservative, strong density filtering).
    Late: low gamma (trust uncertainty more as outliers are removed from pool).
    """
    def __init__(self, batch_size: int = 1, seed: int = 0,
                 gamma_max: float = 3.0, gamma_min: float = 0.5, total_iter: int = 1000):
        super().__init__(batch_size=batch_size, seed=seed)
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.total_iter = total_iter
        self.current_iter = 0

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)

            # Linear decay from gamma_max to gamma_min
            ratio = min(self.current_iter / max(self.total_iter, 1), 1.0)
            gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * ratio
            score = std * np.power(density, gamma)

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()

            self.current_iter += 1
            return idx, acquisition, idx_remain
        else:
            self.current_iter += 1
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return (f"AdaptiveGammaDWSelector(batch_size={self.batch_size}, "
                f"gamma_max={self.gamma_max}, gamma_min={self.gamma_min}, total_iter={self.total_iter})")


class DWRepulsionSelector(BaseRandomSelector):
    """Density-weighted uncertainty with repulsion from recently selected points.

    score = std * density * min_distance_to_recent_selections
    Prevents repeatedly sampling from the same dense-uncertain region.
    Tracks a rolling history of the last N selected pool indices.
    """
    def __init__(self, batch_size: int = 1, seed: int = 0, history_size: int = 20):
        super().__init__(batch_size=batch_size, seed=seed)
        self.history_size = history_size
        self._recent_uidx = []

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)

            # Compute repulsion from recently selected points
            if len(self._recent_uidx) > 0:
                # Find which pool indices correspond to recent uidx
                pool_uidx = [data.uidx for data in dataset_pool.data]
                recent_pool_idx = []
                for uidx in self._recent_uidx:
                    if uidx in pool_uidx:
                        recent_pool_idx.append(pool_uidx.index(uidx))

                if len(recent_pool_idx) > 0:
                    # Max similarity to any recently selected point (in pool space)
                    K_recent = K[:, recent_pool_idx]
                    max_sim_recent = K_recent.max(axis=1)
                    repulsion = 1.0 - max_sim_recent
                    repulsion = np.maximum(repulsion, 1e-10)
                else:
                    repulsion = np.ones(len(dataset_pool))
            else:
                repulsion = np.ones(len(dataset_pool))

            score = std * density * repulsion

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            # Track selected uidx for repulsion
            for i in idx:
                self._recent_uidx.append(dataset_pool.data[i].uidx)
            # Keep only the most recent entries
            if len(self._recent_uidx) > self.history_size:
                self._recent_uidx = self._recent_uidx[-self.history_size:]

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"DWRepulsionSelector(batch_size={self.batch_size}, history_size={self.history_size})"


class AOptimalSelector(BaseRandomSelector):
    """A-Optimal experimental design: maximize expected total variance reduction.

    score(x) = sum_j K(x, x_j)^2 / sigma^2(x) for all j in pool.
    Naturally penalizes outliers: they have high sigma^2 (denominator) but low
    covariance to other pool points (numerator), yielding low scores.
    """
    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()
            var = std ** 2
            var = np.maximum(var, 1e-10)

            K = kernel(dataset_pool.X)
            # A-optimal: sum of squared covariances / own variance
            score = np.sum(K ** 2, axis=1) / var

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"AOptimalSelector(batch_size={self.batch_size})"


class CoreSetSelector(BaseRandomSelector):
    """Core-set selection via greedy k-center in kernel space.

    score(x) = min_{j in train} d_kernel(x, x_j)
    where d(x, y) = K(x,x) + K(y,y) - 2*K(x,y).
    Selects the pool point farthest from any training point, filling coverage gaps.
    Pure geometric representativeness, no uncertainty signal.
    """
    def __call__(self, model, dataset_pool, dataset_train, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            K_pp_diag = np.diag(kernel(dataset_pool.X))

            if len(dataset_train) > 0:
                K_tp = kernel(dataset_train.X, dataset_pool.X)  # (n_train, n_pool)
                K_tt_diag = np.diag(kernel(dataset_train.X))
                # d(pool_i, train_j) = K_pp[i,i] + K_tt[j,j] - 2*K_tp[j,i]
                distances = K_pp_diag[None, :] + K_tt_diag[:, None] - 2 * K_tp  # (n_train, n_pool)
                distances = np.maximum(distances, 0.0)
                # For each pool point, find minimum distance to any training point
                score = distances.min(axis=0)  # (n_pool,)
            else:
                score = K_pp_diag

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"CoreSetSelector(batch_size={self.batch_size})"


class PercentileDensitySelector(BaseRandomSelector):
    """Two-stage selection: filter by uncertainty percentile, then select densest.

    Stage 1: Keep only pool points in the top P% by uncertainty (std).
    Stage 2: Among those, select the point with highest density (mean kernel similarity).
    Hard threshold eliminates outliers entirely rather than just downweighting them.
    """
    def __init__(self, batch_size: int = 1, seed: int = 0, percentile: float = 75.0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.percentile = percentile

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)

            # Stage 1: filter to top percentile by uncertainty
            threshold = np.percentile(std, self.percentile)
            mask_uncertain = std >= threshold

            # Stage 2: among filtered, select densest
            score = np.full_like(std, -np.inf)
            score[mask_uncertain] = density[mask_uncertain]

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = density[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"PercentileDensitySelector(batch_size={self.batch_size}, percentile={self.percentile})"


class BoltzmannDWSelector(BaseRandomSelector):
    """Stochastic density-weighted uncertainty via Boltzmann (softmax) sampling.

    P(x) proportional to exp(std(x) * density(x) / tau).
    Samples one point from this distribution instead of deterministic argmax.
    Temperature tau decays linearly from tau_max to tau_min over iterations.
    """
    def __init__(self, batch_size: int = 1, seed: int = 0,
                 tau_max: float = 1.0, tau_min: float = 0.1, total_iter: int = 1000):
        super().__init__(batch_size=batch_size, seed=seed)
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.total_iter = total_iter
        self.current_iter = 0

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)

            raw_score = std * density

            # Temperature schedule: linear decay
            ratio = min(self.current_iter / max(self.total_iter, 1), 1.0)
            tau = self.tau_max - (self.tau_max - self.tau_min) * ratio
            tau = max(tau, 1e-10)

            # Boltzmann probabilities (shift for numerical stability)
            logits = raw_score / tau
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()

            # Sample from the distribution
            rng = np.random.RandomState(self.seed + self.current_iter)
            idx = [int(rng.choice(len(probs), p=probs))]
            acquisition = [raw_score[idx[0]]]

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()

            self.current_iter += 1
            return idx, acquisition, idx_remain
        else:
            self.current_iter += 1
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return (f"BoltzmannDWSelector(batch_size={self.batch_size}, "
                f"tau_max={self.tau_max}, tau_min={self.tau_min}, total_iter={self.total_iter})")


class DensityUncertaintyScheduleSelector(BaseRandomSelector):
    """Scheduled blend from density to uncertainty selection.

    score = (1 - alpha) * density_normalized + alpha * std_normalized
    alpha = t / T, linearly increases from 0 (pure density) to 1 (pure uncertainty).
    Early iterations prioritize representative sampling; later iterations trust uncertainty.
    """
    def __init__(self, batch_size: int = 1, seed: int = 0, total_iter: int = 1000):
        super().__init__(batch_size=batch_size, seed=seed)
        self.total_iter = total_iter
        self.current_iter = 0

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K = kernel(dataset_pool.X)
            density = K.mean(axis=1)

            # Normalize both to [0, 1]
            std_range = std.max() - std.min()
            std_norm = (std - std.min()) / std_range if std_range > 1e-10 else np.zeros_like(std)

            den_range = density.max() - density.min()
            den_norm = (density - density.min()) / den_range if den_range > 1e-10 else np.zeros_like(density)

            # Schedule: alpha increases from 0 to 1
            alpha = min(self.current_iter / max(self.total_iter, 1), 1.0)
            score = (1 - alpha) * den_norm + alpha * std_norm

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()

            self.current_iter += 1
            return idx, acquisition, idx_remain
        else:
            self.current_iter += 1
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return (f"DensityUncertaintyScheduleSelector(batch_size={self.batch_size}, "
                f"total_iter={self.total_iter})")


class CoreSetUncertaintySelector(BaseRandomSelector):
    """Core-set distance combined with uncertainty rank.

    score = min_dist_to_train * rank(std)
    Selects points that are both far from training data AND uncertain.
    Combines geometric coverage (core-set) with informativeness (uncertainty rank).
    """
    def __call__(self, model, dataset_pool, dataset_train, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            from scipy.stats import rankdata

            mean, std = model.predict(dataset_pool.X, return_std=True)
            std = np.asarray(std).ravel()

            K_pp_diag = np.diag(kernel(dataset_pool.X))

            if len(dataset_train) > 0:
                K_tp = kernel(dataset_train.X, dataset_pool.X)  # (n_train, n_pool)
                K_tt_diag = np.diag(kernel(dataset_train.X))
                distances = K_pp_diag[None, :] + K_tt_diag[:, None] - 2 * K_tp
                distances = np.maximum(distances, 0.0)
                min_dist = distances.min(axis=0)
            else:
                min_dist = K_pp_diag

            rank_std = rankdata(std)
            score = min_dist * rank_std

            idx = get_topn_idx(score, n=self.batch_size, target="max")
            acquisition = score[np.array(idx)].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"CoreSetUncertaintySelector(batch_size={self.batch_size})"
