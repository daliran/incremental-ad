import math
import torch

from typing import Dict
from ties_merging_utility import ties_merging


def add_merging_args(parser):
    parser.add_argument('--merging', type=str, default='ta',
                        choices=["ta", "dare", "iso", "ties", 'tsv'])
    parser.add_argument('--alpha_merging', type=float, default=1.0)


def get_merging_function(command_args, device):
    if command_args.merging == 'ta':
        return TaskArithmetic(device, alpha=command_args.alpha_merging)
    elif command_args.merging == 'dare':
        return DARE(device, alpha=command_args.alpha_merging)
    elif command_args.merging == 'iso':
        return ISO(device, alpha=command_args.alpha_merging)
    elif command_args.merging == 'ties':
        return TIES(device, alpha=command_args.alpha_merging)
    elif command_args.merging == 'tsv':
        return TSV(device, alpha=command_args.alpha_merging)
    else:
        raise ValueError


class AbstractMerging:

    def merge(self):
        raise NotImplementedError

    def add(self: Dict):
        raise NotImplementedError


class TaskArithmetic(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict = None
        self.scaled_sum: Dict = None

    @torch.no_grad()
    def merge(self, names=None):
        assert self._running_sum.keys() == self.scaled_sum.keys()
        for k in self._running_sum.keys():
            self.scaled_sum[k].copy_(self._running_sum[k])
            self.scaled_sum[k].mul_(self.alpha)
        if names is None:
            return self.scaled_sum
        assert self.scaled_sum.keys() == set(names)
        return [self.scaled_sum[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self.scaled_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
        assert param_dict.keys() == self._running_sum.keys()
        for k, v in param_dict.items():
            self._running_sum[k].add_(v)
        self.num_tasks += 1


class DARE(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0, p: float = 0.7):
        self.device = device
        self.alpha = alpha
        self.p = p
        self.num_tasks = 0
        self._running_sum: Dict = None
        self.scaled_sum: Dict = None

    def randbin(self, M, N):
        return torch.randint(2, size=(M, N), dtype=torch.float32).\
            bernoulli(1 - self.p).to(self.device)

    @torch.no_grad()
    def merge(self, names=None):
        assert self._running_sum.keys() == self.scaled_sum.keys()
        for k, v in self._running_sum.items():
            self.scaled_sum[k].copy_(v)
            self.scaled_sum[k].mul_(self.alpha)
        if names is None:
            return self.scaled_sum
        assert self.scaled_sum.keys() == set(names)
        return [self.scaled_sum[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self.scaled_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
        assert param_dict.keys() == self._running_sum.keys()
        for k, v in param_dict.items():
            if len(v.shape) != 2:
                self._running_sum[k].add_(v)
            else:
                mask_ = self.randbin(v.shape[0], v.shape[1])
                self._running_sum[k].add_(v * mask_ * (1 / (1 - self.p)))
        self.num_tasks += 1


class ISO(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._separated_task_vectors: Dict = None
        self.per_task_weights = []

    @torch.no_grad()
    def merge(self, names=None, pruning_rank=None):
        """
        Args:
            pruning_rank (int, optional): If provided, the percentage of singular values to keep. Defaults to None.
        """
        for k, v in self._separated_task_vectors.items():
            iso_sum = torch.zeros_like(v[0])
            for i, tv in enumerate(v):
                if len(self.per_task_weights) == 0:
                    iso_sum.add_(tv * (1 / self.num_tasks))
                else:
                    iso_sum.add_(tv * self.per_task_weights[i])
            self.merged_model[k].copy_(iso_sum)
            if len(v[0].shape) == 2:
                U, S, V = torch.linalg.svd(self.merged_model[k], full_matrices=False)
                if pruning_rank is not None:
                    pruning_rank_k = math.ceil(pruning_rank * S.shape[0])
                    S = S[:pruning_rank_k]
                    U = U[:, :pruning_rank_k]
                    V = V[:pruning_rank_k, :]
                s_mean = S.mean()
                if len(self.per_task_weights) > 0:
                    s_mean /= sum(self.per_task_weights)
                S_iso = torch.diag(torch.ones_like(S) * s_mean)
                self.merged_model[k].copy_(torch.linalg.multi_dot((U, S_iso, V)))
            self.merged_model[k].mul_(self.alpha)
        if names is None:
            return self.merged_model
        assert self.merged_model.keys() == set(names)
        return [self.merged_model[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._separated_task_vectors is None:
            self.merged_model = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self._separated_task_vectors = {k: [] for k, v in param_dict.items()}
        for k, v in param_dict.items():
            self._separated_task_vectors[k].append(torch.clone(v))
        self.num_tasks += 1

    def set_per_task_weights(self, weights):
        self.per_task_weights = weights

    def apply_ta(self, v):
        if len(v.shape) == 2:
            return False
        return True


class TIES(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict = None
        self._separated_task_vectors: Dict = None
        self.merged_model: Dict = None

    def apply_ta(self, v):
        if len(v.shape) == 2:
            return False
        return True

    @torch.no_grad()
    def merge(self, names=None):
        for k, v in self._running_sum.items():
            self.merged_model[k].copy_(v)
            self.merged_model[k].mul_(self.alpha / self.num_tasks)
        for k, v in self._separated_task_vectors.items():
            merged_tv, _, _ = ties_merging(v)
            self.merged_model[k].copy_((self.alpha / self.num_tasks) * merged_tv)
        if names is None:
            return self.merged_model
        assert self.merged_model.keys() == set(names)
        return [self.merged_model[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self.merged_model = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items() if self.apply_ta(v)}
            self._separated_task_vectors = {k: [] for k, v in param_dict.items() if not self.apply_ta(v)}
        for k, v in param_dict.items():
            if self.apply_ta(v):
                self._running_sum[k].add_(v)
            else:
                self._separated_task_vectors[k].append(torch.clone(v))
        self.num_tasks += 1


class TSV(AbstractMerging):

    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict = None
        self._separated_task_vectors: Dict = None
        self.merged_model: Dict = None
        self.per_task_weights = []

    def apply_ta(self, v):
        if len(v.shape) == 2:
            return False
        return True

    @staticmethod
    @torch.no_grad()
    def get_tsv_delta_w(ftms_task_dirs, per_task_weights):
        sv_reduction = 1 / len(ftms_task_dirs)
        for i, vec in enumerate(ftms_task_dirs):
            u, s, v = torch.linalg.svd(vec.to(torch.float64), full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(v)
            reduced_index_s = int(s.shape[0] * sv_reduction)
            # select only the first reduced_index_s columns of u and place them
            sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[
                :, :reduced_index_s
            ]
            sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[
                :reduced_index_s
            ] * per_task_weights[i]

            # select only the first reduced_index_s rows of v and place them
            sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[
                :reduced_index_s, :
            ]
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).type_as(ftms_task_dirs[0])

    @torch.no_grad()
    def merge(self, names=None):
        assert len(self.per_task_weights) == self.num_tasks, "Wrong number of per-task weights."
        for k, v in self._running_sum.items():
            self.merged_model[k].copy_(v)
            self.merged_model[k].div_(self.num_tasks)
            self.merged_model[k].mul_(self.alpha)
        for k, v in self._separated_task_vectors.items():
            merged_tv = self.get_tsv_delta_w(v, self.per_task_weights)
            merged_tv = merged_tv.type_as(v[0]) if \
                hasattr(merged_tv, 'type_as') else merged_tv
            self.merged_model[k].copy_(self.alpha * merged_tv)
        if names is None:
            return self.merged_model
        assert self.merged_model.keys() == set(names)
        return [self.merged_model[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self.merged_model = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items() if self.apply_ta(v)}
            self._separated_task_vectors = {k: [] for k, v in param_dict.items() if not self.apply_ta(v)}
        for k, v in param_dict.items():
            if self.apply_ta(v):
                self._running_sum[k].add_(v)
            else:
                self._separated_task_vectors[k].append(torch.clone(v))
        self.num_tasks += 1
        self.per_task_weights.append(1.0)

    def set_per_task_weights(self, weights):
        self.per_task_weights = weights
