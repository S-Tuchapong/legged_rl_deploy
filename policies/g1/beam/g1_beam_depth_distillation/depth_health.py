"""Depth validity policy; no simulator dependency and no actor feature changes."""
import torch
class DepthHealth:
    # Diagnostic only: 100 policy observations = 2 seconds at 50 Hz.
    ZERO_DIAGNOSTIC_TICKS = 100
    def __init__(self, n, device='cpu'):
        self.streak=torch.zeros(n,dtype=torch.long,device=device)
        self.counts=torch.zeros(5,dtype=torch.float64,device=device)
        self.max_streak=torch.zeros((),dtype=torch.long,device=device)
    @torch.no_grad()
    def observe(self, x, reset, fresh):
        if x.ndim!=4 or x.shape[0]!=len(self.streak) or x.shape[1]!=1:
            raise ValueError('Expected N1HW depth')
        if reset.shape!=self.streak.shape or reset.dtype!=torch.bool or fresh.shape!=reset.shape or fresh.dtype!=torch.bool:
            raise ValueError('Expected N bool reset/fresh')
        # Fail closed for ANY numerical fault, including mixed valid/invalid frames.
        # Finite zero is a renderer no-hit, negative/NaN/Inf are NOT no-hit.
        nonfinite=~torch.isfinite(x); negative=x<0
        if nonfinite.any() or negative.any():
            raise RuntimeError(f'Depth numerical fault: nonfinite_pixels={int(nonfinite.sum())}, negative_pixels={int(negative.sum())}')
        empty=~(x>0).flatten(1).any(1)
        self.streak[reset]=0
        self.streak=torch.where(empty,self.streak+1,torch.zeros_like(self.streak))
        self.max_streak=torch.maximum(self.max_streak,self.streak.max())
        self.counts+=torch.stack((empty.sum(),torch.tensor(len(x),device=x.device), (empty&fresh).sum(),fresh.sum(),(self.streak>=self.ZERO_DIAGNOSTIC_TICKS).sum())).to(torch.float64)
    @torch.no_grad()
    def flush(self):
        counts=self.counts.clone(); peak=self.max_streak.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
            torch.distributed.all_reduce(peak,op=torch.distributed.ReduceOp.MAX)
        self.counts.zero_();self.max_streak.zero_()
        names=['empty_observations','observations','empty_delivered_frames','delivered_frames','long_empty_observations']
        out={'DepthHealth/'+k:float(v) for k,v in zip(names,counts)}
        out.update({'DepthHealth/empty_observation_fraction':float(counts[0]/counts[1].clamp_min(1)),
                    'DepthHealth/empty_delivered_fraction':float(counts[2]/counts[3].clamp_min(1)),
                    'DepthHealth/max_consecutive_empty_ticks':int(peak),
                    'DepthHealth/max_consecutive_empty_seconds':int(peak)*.02})
        return out
