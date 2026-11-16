import torch


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, y):
        self.mean = torch.mean(y)
        self.std = torch.std(y) + 1e-4
        return (y - self.mean) / self.std, self.mean, self.std

    # def inverse_transform(self, y):
    #     return y * self.std + self.mean

    def inverse_transform(self, y, mean, std):
        self.mean = mean
        self.std = std
        return y * self.std + self.mean

    def transform(self, y, mean, std):
        self.mean = mean
        self.std = std
        return (y - self.mean) / self.std



class MaxScaler:
    def __init__(self):
        self.max = None

    def fit_transform(self, y):
        self.max = torch.max(y)
        return y / self.max, self.max

    def inverse_transform(self, y, max):
        self.max = max
        return y * self.max

    def transform(self, y, max):
        self.max = max
        return y / self.max


class MeanScaler:
    def __init__(self):
        self.mean = None

    def fit_transform(self, y):
        self.mean = torch.mean(y)
        return y / self.mean, self.mean

    def inverse_transform(self, y, mean):
        self.mean = mean
        return y * self.mean

    def transform(self, y, mean):
        self.mean = mean
        return y / self.mean


class LogScaler:

    def fit_transform(self, y):
        return torch.log1p(y)

    def inverse_transform(self, y):
        return torch.expm1(y)

    def transform(self, y):
        return torch.log1p(y)

def smooth(scalar, weight=0.5):
    smoothed = []
    last = scalar[0]
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


class ColumnStandardScaler:
    """
    Per-column standardisation: every column of y [B, S, C] gets its OWN mean and
    std, computed over the time axis of each sample separately (the statistics
    are [B, 1, C]). Unlike StandardScaler (one scalar for the whole tensor),
    columns on very different scales -- VIX ~15, a 0/1 flag, a price ~6800 --
    each end up with unit-scale values, and a sample's normalisation does not
    depend on which other samples share its batch.

    A column that is (almost) constant inside a window has no usable std; dividing
    by it would blow the next day's value up to ~1e4. For those columns the std
    is set to 1, so the column is simply centred on its window mean.
    """
    def __init__(self, rel_floor=1e-4):
        self.rel_floor = rel_floor
        self.mean = None
        self.std = None

    def _safe_std(self, std, mean):
        floor = self.rel_floor * torch.clamp(mean.abs(), min=1.0)
        return torch.where(std < floor, torch.ones_like(std), std)

    def fit_transform(self, y):
        mean = y.mean(dim=1, keepdim=True)
        std = ((y - mean) ** 2).mean(dim=1, keepdim=True).sqrt()
        std = self._safe_std(std, mean)
        self.mean, self.std = mean, std
        return (y - mean) / std, mean, std

    def transform(self, y, mean, std):
        return (y - mean) / std

    def inverse_transform(self, y, mean, std):
        # caller passes mean/std already shaped to broadcast against y
        return y * std + mean