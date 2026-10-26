"""Loss functions: differentiable gamma passing rate, weighted MSE and their combination (Sec. 3 of the paper)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GammaIndexLoss(nn.Module):
    """Differentiable gamma passing rate loss (Eq. 5-8), after Martinot et al., IPMI 2023.

    Both doses are resampled to 1 mm^3 with trilinear interpolation. The gamma index is the minimum
    over a spherical neighbourhood of offsets |delta| <= DTA, evaluated in parallel as a 3D convolution
    with one-hot kernels. Dose difference and cutoff are global, relative to the target maximum.
    """

    def __init__(self, dose_percent=2.0, dta_mm=2.0, voxel_size_mm=(2.0, 2.0, 2.0), dose_cutoff=0.2,
                 beta_init=0.1, max_gamma=10.0):
        super().__init__()
        self.dose_threshold = dose_percent / 100.0
        self.dta_mm = dta_mm
        self.voxel_size = (tuple(voxel_size_mm) if isinstance(voxel_size_mm, (list, tuple))
                           else (voxel_size_mm,) * 3)
        self.dose_cutoff = dose_cutoff
        self.max_gamma = max_gamma
        self.register_buffer("beta", torch.tensor(beta_init, dtype=torch.float32))
        self._build_search_kernels()

    def _build_search_kernels(self):
        """One-hot kernels for every 1 mm offset within the DTA sphere, and the matching |delta|^2 / DTA^2."""
        r = int(math.ceil(self.dta_mm))
        k = 2 * r + 1
        dta_sq = self.dta_mm ** 2
        offsets = [(oz, oy, ox)
                   for oz in range(-r, r + 1) for oy in range(-r, r + 1) for ox in range(-r, r + 1)
                   if oz * oz + oy * oy + ox * ox <= dta_sq]

        kernels = torch.zeros(len(offsets), 1, k, k, k, dtype=torch.float32)
        for i, (oz, oy, ox) in enumerate(offsets):
            kernels[i, 0, oz + r, oy + r, ox + r] = 1.0
        distance_term = [(oz * oz + oy * oy + ox * ox) / dta_sq for oz, oy, ox in offsets]

        self.register_buffer("kernels", kernels)
        self.register_buffer("dta_sq", torch.tensor(distance_term, dtype=torch.float32))
        self.pad_size = r

    def _resample_to_1mm(self, x):
        if all(abs(s - 1.0) < 1e-6 for s in self.voxel_size):
            return x
        _, _, d, h, w = x.shape
        size = (int(d * self.voxel_size[0]), int(h * self.voxel_size[1]), int(w * self.voxel_size[2]))
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)

    def compute_gamma(self, pred, target):
        """Gamma map (clamped at ``max_gamma``) and the mask of voxels above the dose cutoff, both at 1 mm^3.

        Gamma(r) = min_delta sqrt(|delta|^2 / DTA^2 + (pred(r + delta) - target(r))^2 / Delta^2)
        """
        if pred.dim() == 3:
            pred, target = pred[None, None], target[None, None]
        elif pred.dim() == 4:
            pred, target = pred.unsqueeze(1), target.unsqueeze(1)

        pred = self._resample_to_1mm(pred)
        target = self._resample_to_1mm(target)

        # Global criteria, relative to the maximum of the target batch (targets are max-normalized to 1)
        delta = self.dose_threshold * target.max()
        valid_mask = target >= self.dose_cutoff * target.max()

        kernels = self.kernels.to(dtype=pred.dtype)  # match autocast dtype
        dta_sq = self.dta_sq.to(dtype=pred.dtype)

        pred_neighbors = F.conv3d(pred, kernels, padding=self.pad_size)  # [B, n_offsets, D, H, W]
        gamma_sq = ((pred_neighbors - target) / (delta + 1e-10)) ** 2 + dta_sq.view(1, -1, 1, 1, 1)
        gamma = torch.sqrt(gamma_sq.min(dim=1, keepdim=True).values + 1e-10)
        return torch.clamp(gamma, max=self.max_gamma), valid_mask

    def forward(self, pred, target):
        """Eq. 8: 1 - soft GPR, with the soft pass sigmoid(beta * (1 - Gamma)), averaged over the batch."""
        gamma, valid_mask = self.compute_gamma(pred, target)
        soft_pass = torch.sigmoid(self.beta * (1.0 - gamma))
        valid = valid_mask.float()
        dims = (1, 2, 3, 4)
        soft_gpr = (soft_pass * valid).sum(dim=dims) / (valid.sum(dim=dims) + 1e-8)
        return (1.0 - soft_gpr).mean()

    @torch.no_grad()
    def compute_pass_rate(self, pred, target):
        """Hard gamma passing rate (%) over the whole batch, used for monitoring and model selection."""
        gamma, valid_mask = self.compute_gamma(pred, target)
        n_valid = valid_mask.float().sum()
        if n_valid == 0:
            return 100.0
        return (((gamma < 1.0) & valid_mask).float().sum() / n_valid).item() * 100

    def set_beta(self, value):
        self.beta.fill_(value)


class BetaScheduler:
    """Piecewise linear annealing of the sigmoid sharpness beta, stepped once per epoch.

    beta = beta_start until ``warmup_epochs``, then rises linearly towards beta_mid until ``phase1_end``
    and towards beta_end until ``phase2_end``; it is updated only every ``update_interval`` epochs and is
    constant at beta_end afterwards.
    """

    def __init__(self, loss_fn, beta_start=0.1, beta_mid=2.0, beta_end=5.0, warmup_epochs=10, phase1_end=50,
                 phase2_end=100, update_interval=5):
        self.loss_fn = loss_fn
        self.beta_start = beta_start
        self.beta_mid = beta_mid
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.update_interval = update_interval
        self._last_update = 0
        self._current_beta = beta_start
        loss_fn.set_beta(beta_start)

    def _linear(self, epoch, start_epoch, end_epoch, start_beta, end_beta):
        if epoch - self._last_update >= self.update_interval:
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            self._current_beta = start_beta + (end_beta - start_beta) * progress
            self._last_update = epoch
        return self._current_beta

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            beta = self.beta_start
        elif epoch < self.phase1_end:
            beta = self._linear(epoch, self.warmup_epochs, self.phase1_end, self.beta_start, self.beta_mid)
        elif epoch < self.phase2_end:
            beta = self._linear(epoch, self.phase1_end, self.phase2_end, self.beta_mid, self.beta_end)
        else:
            beta = self.beta_end
        self.loss_fn.set_beta(beta)
        return beta


class WeightedMSE(nn.Module):
    """Eq. 3-4: per-sample MSE weighted by exp(alpha * clip(target, -10, 10)), averaged over the batch."""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        weights = torch.exp(self.alpha * torch.clamp(target, -10.0, 10.0))
        dims = tuple(range(1, pred.dim()))
        per_sample = (weights * (pred - target) ** 2).sum(dim=dims) / (weights.sum(dim=dims) + 1e-8)
        return per_sample.mean()


class CombinedWMSEGammaLoss(nn.Module):
    """Eq. 9-11: WMSE normalized by its exponential moving average, plus the gamma loss.

    Note: the moving average is updated on every call, including validation batches; this matches
    the training runs reported in the paper.
    """

    def __init__(self, wmse_loss, gamma_loss, wmse_weight=1.0, gamma_weight=1.0, momentum=0.99, eps=1e-8):
        super().__init__()
        self.wmse_loss = wmse_loss
        self.gamma_loss = gamma_loss
        self.wmse_weight = wmse_weight
        self.gamma_weight = gamma_weight
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("wmse_avg", torch.tensor(0.0))

    def forward(self, pred, target):
        loss_wmse = self.wmse_loss(pred, target)
        loss_gamma = self.gamma_loss(pred, target)

        with torch.no_grad():
            if self.wmse_avg == 0.0:  # first call
                self.wmse_avg = loss_wmse.detach().clone()
            else:
                self.wmse_avg = self.momentum * self.wmse_avg + (1 - self.momentum) * loss_wmse.detach()

        loss_wmse_norm = loss_wmse / (self.wmse_avg + self.eps)
        total = self.wmse_weight * loss_wmse_norm + self.gamma_weight * loss_gamma
        return total, {"wmse": loss_wmse.item(), "wmse_norm": loss_wmse_norm.item(), "gamma": loss_gamma.item()}

    def compute_pass_rate(self, pred, target):
        return self.gamma_loss.compute_pass_rate(pred, target)
