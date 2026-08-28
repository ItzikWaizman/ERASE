import random

import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor
        # ----- Dynamic per-sample stop hook -----
        # When the ERASE trainer enables `dynamic_stop_loss_threshold > 0`, it
        # registers a SHARED set of "still-active" forget indices via
        # ``set_dyn_active_indices``. While that set is non-empty, every
        # ``__getitem__`` call whose requested forget index has been
        # marked done is transparently swapped for a random active forget
        # index. The dataloader/sampler does not need to know -- it keeps
        # emitting the full [0, N) permutation per epoch -- but the actual
        # batches contain only ACTIVE forget samples, so the resistant
        # tail gets near-100% sample-time once it's the only one left.
        # Mutated in place by the trainer; we read it lazily each call.
        # None = the legacy (non-dynamic) behaviour, bit-identical to all
        # pre-existing runs.
        self._dyn_active_forget_indices = None
        # Reduced-probability re-sampling for "done" samples. When > 0,
        # done samples are NOT fully excluded; they get drawn with this
        # probability each time a non-active index is requested. This
        # lets the band/MSE loss keep correcting drift on done samples.
        self._dyn_done_sample_prob = 0.0

    def set_dyn_active_indices(self, active_indices):
        """Register a SHARED mutable set of still-active forget indices.

        The trainer mutates this set in place when samples cross the
        dynamic-stop threshold. We hold a reference (not a copy), so each
        ``__getitem__`` always sees the latest membership.
        """
        self._dyn_active_forget_indices = active_indices

    def set_done_sample_prob(self, prob: float):
        """Set probability of keeping a done sample (instead of swapping)."""
        self._dyn_done_sample_prob = float(prob)

    def _maybe_swap_forget_idx(self, idx):
        active = self._dyn_active_forget_indices
        if active is None or len(active) == 0:
            return idx
        if idx in active:
            return idx
        # idx is "done". With `done_sample_prob` chance, keep it as-is
        # so the band/MSE loss can still correct drift on finished samples.
        if self._dyn_done_sample_prob > 0 and random.random() < self._dyn_done_sample_prob:
            return idx
        # Otherwise substitute with a uniformly random active one.
        active_tuple = tuple(active)
        pick = torch.randint(0, len(active_tuple), (1,)).item()
        return active_tuple[pick]

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            forget_idx = self._maybe_swap_forget_idx(idx)
            item["forget"] = self.forget[forget_idx]
            if self.retain:
                retain_idx = torch.randint(0, len(self.retain), (1,)).item()
                item["retain"] = self.retain[retain_idx]
        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            if self.forget:
                forget_idx = torch.randint(0, len(self.forget), (1,)).item()
                forget_idx = self._maybe_swap_forget_idx(forget_idx)
                item["forget"] = self.forget[forget_idx]
        return item
