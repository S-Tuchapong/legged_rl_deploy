"""Isolated CPU-tested core. No environment registration or training entry point."""
import copy
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from depth_health import DepthHealth
CAMERA_WIDTH=424
CAMERA_HEIGHT=240
CAMERA_FOVY=58.310943190845826
CAMERA_FY=215.1071014404295
DIMS=(3,3,3,2,29,29,29)
SHA='bc8a0f4ba974a99cb10a0d6c129c7438e6be1506119dfdd9f91013896f641b3e'
def unpack(x):
    if x.shape[-1]!=5978: raise ValueError("Expected term-major H61 proprioception")
    return torch.cat([p.reshape(-1,61,d) for p,d in zip(x.split([61*d for d in DIMS],-1),DIMS)],-1)
def head():
    return nn.Sequential(nn.Linear(128,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,29))
class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        for k,v in {'_mean':torch.zeros(1,5978),'_var':torch.ones(1,5978),'_std':torch.ones(1,5978),'count':torch.tensor(0,dtype=torch.long)}.items(): self.register_buffer(k,v)
    def forward(self,x): return (x-self._mean)/(self._std+.01)
class Window(nn.Module):
    def __init__(self):
        super().__init__(); self.core=nn.GRU(98,64,batch_first=True)
    def forward(self,x): return self.core(x)[0][:,-1]
class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=Window()
        self.cnn=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ELU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.ELU(),nn.Flatten(),nn.Linear(2464,64),nn.ELU())
    def forward(self,x): return torch.cat((self.gru(unpack(x[:,:5978])),self.cnn(x[:,5978:].reshape(-1,1,13,21))),-1)
class Teacher(nn.Module):
    def __init__(self):
        super().__init__();self.obs_normalizer=Normalizer(); self.mlp=nn.Sequential(Encoder(),head())
        self.distribution=nn.Module();self.distribution.register_parameter('std_param',nn.Parameter(torch.ones(29)))
    def train(self,mode=True): return super().train(False)
    def forward(self,x):
        if x.shape[-1]!=6251: raise ValueError('Teacher requires 6251')
        return self.mlp(torch.cat((self.obs_normalizer(x[:,:5978]),x[:,5978:]),-1))
    @torch.no_grad()
    def labels(self,x,limits):
        return clip_action(self(x),limits)
def load_teacher(path):
    path=Path(path)
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=SHA: raise ValueError('Not the authorized teacher SHA256')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=Teacher(); model.load_state_dict(checkpoint['actor_state_dict'],strict=True)
    model.requires_grad_(False).eval();return model
class Student(nn.Module):
    def __init__(self,teacher):
        super().__init__();self.normalizer=copy.deepcopy(teacher.obs_normalizer)
        self.proprio=copy.deepcopy(teacher.mlp[0].gru)
        self.normalizer.requires_grad_(False);self.proprio.requires_grad_(False)
        self.visual=nn.Sequential(nn.Conv2d(1,16,5,stride=2,padding=2),nn.ELU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.ELU(),nn.AdaptiveAvgPool2d((3,5)),nn.Flatten(),nn.Linear(480,64),nn.ELU())
        self.terrain=nn.GRUCell(64+98+2,64);self.actor=head()
    def train(self,mode=True):
        super().train(mode);self.normalizer.eval();self.proprio.eval();return self
    def forward(self,history,depth,fresh,age,hidden,reset):
        # reset describes the current observation, not the result of this action.
        with torch.no_grad():
            window=unpack(self.normalizer(history)); prop=self.proprio(window)
        hidden=hidden*(~reset.bool()).to(hidden.dtype).unsqueeze(-1)
        features=torch.cat((self.visual(depth),window[:,-1],fresh[:,None],age[:,None]),-1)
        hidden=self.terrain(features,hidden)
        return self.actor(torch.cat((prop,hidden),-1)),hidden
def corrupt_axial(depth, *, multiplier=None, gaussian_std=0., pixel_dropout=0., quantize=False):
    """Optional Go2-inspired subset; invalid pixels remain far. No GT inputs."""
    if gaussian_std<0 or not 0<=pixel_dropout<=1: raise ValueError('Invalid corruption')
    if multiplier is not None and not 0<multiplier[0]<=multiplier[1]: raise ValueError('Invalid multiplier')
    valid=torch.isfinite(depth)&(depth>0)
    z=torch.where(valid,depth,torch.full_like(depth,3.)).clamp(.3,3.)
    if multiplier is not None:
        z=z*torch.empty((len(z),1,1,1),device=z.device,dtype=z.dtype).uniform_(*multiplier)
    if gaussian_std: z=z+torch.randn_like(z)*gaussian_std
    if pixel_dropout: valid=valid&(torch.rand_like(z)>=pixel_dropout)
    z=torch.where(valid,z.clamp(.3,3.),torch.full_like(z,3.))
    if quantize: z=(z/.001).round()*.001
    return ((z-.3)/2.7).clamp(0,1)

class DepthCache:
    """50Hz calls, 25Hz delivered frames; forced fresh on per-env reset.
    Range input N1HW. No corruption enabled unless explicitly passed.
    """
    def __init__(self,n,device='cpu',width=CAMERA_WIDTH,height=CAMERA_HEIGHT,fovy=CAMERA_FOVY,corruption=None):
        if width<=0 or height<=0 or not 0<fovy<180: raise ValueError('Invalid camera')
        self.image=torch.ones(n,1,height,width,device=device);self.age=torch.zeros(n,device=device)
        self.ticks=torch.zeros(n,dtype=torch.long,device=device)
        focal=height/(2*math.tan(math.radians(fovy)/2))
        yy=(torch.arange(height,device=device)+.5-height/2)/focal
        xx=(torch.arange(width,device=device)+.5-width/2)/focal
        self.cosine=torch.rsqrt(1+yy[:,None].square()+xx[None,:].square())
        self.corruption=dict(corruption or {})
        self.health=DepthHealth(n,device)
    @torch.no_grad()
    def update(self,ranges,reset):
        if ranges.shape!=self.image.shape: raise ValueError('Expected N1HW range')
        if reset.shape!=self.ticks.shape or reset.dtype!=torch.bool: raise ValueError('Expected N bool reset')
        self.ticks[reset]=0
        fresh=(self.ticks%2==0)
        self.health.observe(ranges,reset,fresh)
        normalized=corrupt_axial(ranges[fresh]*self.cosine,**self.corruption)
        self.image[fresh]=normalized;self.age+=.02;self.age[fresh]=0;self.ticks+=1
        return self.image.clone(),fresh.float(),self.age.clone()
def sequence_loss(model,batch,initial_hidden,burn_in=4,clip=1.,loss_mode="raw",limits=None):
    """Contiguous T,B sequence only; train hidden never aliases collection hidden.
    First burn_in frames warm recurrent state without gradient, remaining train.
    Call backward per chunk then explicitly detach state at TBPTT boundaries.
    """
    if not 0<=burn_in<len(batch): raise ValueError('Invalid burn-in')
    h=initial_hidden.detach().clone(); losses=[]
    for t,step in enumerate(batch):
        with torch.set_grad_enabled(t>=burn_in):
            action,h=model(step['history'],step['depth'],step['fresh'],step['age'],h,step['reset'])
            if t>=burn_in:
                if loss_mode=='processed': action=clip_action(action,limits)
                losses.append((action-step['label']).square().mean())
        if t<burn_in: h=h.detach()
    return torch.stack(losses).mean(),h.detach()
@torch.no_grad()
def synchronize_state(model):
    """Call after construction/load on both ranks; includes frozen normalizer."""
    if torch.distributed.is_initialized():
        for value in model.state_dict().values(): torch.distributed.broadcast(value,src=0)
def synchronize_gradients(model):
    # Every rank must execute the same full sequence and backward before this.
    if torch.distributed.is_initialized():
        for p in model.parameters():
            if p.requires_grad:
                if p.grad is None: raise RuntimeError('Missing trainable gradient')
                torch.distributed.all_reduce(p.grad);p.grad.div_(torch.distributed.get_world_size())
def mixture_action(teacher_action,student_action,update,mix_updates):
    if mix_updates<=0: raise ValueError('mix_updates must be configured explicitly')
    beta=0.0  # Matched ablation: teacher labels retained, never execute teacher action.
    choose=torch.rand((len(teacher_action),1),device=teacher_action.device)<beta
    return torch.where(choose,teacher_action,student_action)

def clip_action(action,limits):
    lower,upper=limits
    if not torch.all(lower<=upper): raise ValueError("Invalid joint limits")
    return action.clamp(lower,upper)
