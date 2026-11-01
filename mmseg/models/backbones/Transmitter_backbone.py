import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import BACKBONES
from mmseg.registry import MODELS
from mmengine.model import BaseModule

class RMSNorm(nn.Module):
    __constants__ = ['bias']
    bias: bool
    def __init__(self, dim, p=-1., eps=1e-8, bias=False, scale_bool=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super().__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias
        self.scale_bool = scale_bool
        self.register_parameter("scale", nn.Parameter(torch.ones(dim)))

        if self.bias:
            self.register_parameter("offset", nn.Parameter(torch.zeros(dim)))

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale.view(1, 1, -1) * x_normed + self.offset

        if self.scale_bool:
            return self.scale.view(1, 1, -1) * x_normed
        return x_normed

class SelfTransduction(nn.Module):
    def __init__(self,code_size, code_dim, code_num, signal_split, image_size_h, image_size_w, init_limt=0.05, device='cpu'):
        super().__init__()
        self.code_dim = code_dim
        self.code_size = code_size
        self.code_num = code_num
        self.signal_split = signal_split
        self.device = device
        #init cell
        self.register_parameter('key_book', nn.Parameter(torch.nn.init.orthogonal_(torch.FloatTensor(code_size, code_dim), gain=1)))
        self.register_parameter('value_book', nn.Parameter(torch.nn.init.orthogonal_(torch.FloatTensor(code_size, code_dim), gain=1)))

        #loss
        self.commit_loss = nn.L1Loss()
        self.vq_loss = nn.L1Loss()
        #strenghth layer
        self.strength_layer = nn.Conv1d(in_channels=signal_split * (image_size_h * image_size_w), out_channels=signal_split * (image_size_h * image_size_w), kernel_size=code_dim, stride=1, groups= signal_split*(image_size_h * image_size_w))

        #conv
        self.conv = nn.Conv1d(in_channels=(image_size_h * image_size_w), out_channels=signal_split * (image_size_h * image_size_w), kernel_size=code_dim, stride=1, padding='same', groups=(image_size_h * image_size_w))

        self.fatigue = nn.Conv1d(in_channels=signal_split * (image_size_h * image_size_w), out_channels=1, kernel_size=1, stride=1)

        # tmp map
        self.one_tensor = torch.ones(self.code_size * self.code_dim, dtype=torch.long).to(device)

    def _sensory_fatigue(self, tensor):
        # tensor_concept = self.fatigue(tensor.permute(0, 2, 1)).permute(0, 2, 1)
        # tensor_median = torch.median(tensor, dim=1, keepdim=True)
        tensor_concept = self.fatigue(tensor)
        fatigue = (F.cosine_similarity(tensor, tensor_concept, dim=2) + 1) * 0.5
        return fatigue

    @staticmethod
    def _sensory_fatigue_activation(tensor, cof=1):
        return cof/ (cof + torch.exp(-tensor))

    @staticmethod
    def statistics_key_book(codebook):
        flatten_mean = torch.mean(codebook.flatten())
        flatten_std = torch.std(codebook.flatten())
        codebook_dis = torch.pdist(codebook, p=2)
        flatten_dis_mean = torch.mean(codebook_dis.flatten())
        flatten_dis_std = torch.std(codebook_dis.flatten())
        return flatten_mean, flatten_std, flatten_dis_mean, flatten_dis_std

    def reward(self, sum_activated_hot):
        krewardup = torch.sum(sum_activated_hot >0)  #cnt of activated code
        krewardudown = torch.max(sum_activated_hot) * torch.std(sum_activated_hot)

        return krewardup / krewardudown
    def forward(self,tensor):
        #tensor(b, n, c)
        b, n, c= tensor.shape
        loss = 0.
        query = self.conv(tensor)
        strength = self.strength_layer(query)

        mixed_fatigue = self._sensory_fatigue(query)

        keybook = self.key_book
        valuebook = self.value_book

        # euclidean distance (p = 2) Default: use_mm_for_euclid_dist_if_necessary.
        distances = torch.cdist(query.reshape(-1, c), keybook, p=2.)
        activated_pos = distances.argmin(dim=1)
        activated_hot = torch.zeros_like(distances, memory_format=torch.legacy_contiguous_format).scatter_(-1, activated_pos.unsqueeze(1), 1.0)

        #get activate key and val

        key = torch.matmul(activated_hot, keybook).reshape(b,self.signal_split*n,c)
        value = torch.matmul(activated_hot, valuebook).reshape(b,self.signal_split*n,c)

        #calcul usage

        sensory = self._sensory_fatigue_activation(strength.squeeze(2), mixed_fatigue)
        value = torch.mul(value, sensory.unsqueeze(2))
        kreward = self.reward(torch.sum(activated_hot, dim=0))

        loss += self.commit_loss(query, key.detach()).item() * kreward
        loss += self.vq_loss(query.detach(), key).item() * kreward * 1e-4

        value = torch.sum(torch.stack(torch.chunk(value, self.signal_split, 1), 0), 0)
        output = value + tensor
        return output, loss

class TransmitterBlock(nn.Module):
    def __init__(self,book_size, embedding_dim, book_num, heads, splits, image_size_h, image_size_w, init_limt=0.05, device='cuda'):
        super().__init__()
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w
        self.linear = nn.ModuleList([nn.LazyLinear(embedding_dim) for _ in range(heads)])
        self.transduction = nn.ModuleList([SelfTransduction(book_size, embedding_dim, book_num, splits, image_size_h, image_size_w, init_limt=init_limt, device=device) for _ in range(heads)])
        self.Norm = RMSNorm(dim=embedding_dim*heads)
        self.DW_conv = nn.Conv2d(in_channels=embedding_dim*heads, out_channels=embedding_dim*heads, kernel_size=7, stride=1, padding='same', groups=embedding_dim*heads)
        self.heads = heads
        self.embedding_dim = embedding_dim
    def forward(self, z):
        signals = []
        losses = 0
        for mlp, transduction in zip(self.linear, self.transduction):
            signal, loss, usage = transduction(mlp(z))
            losses += loss
            signals.append(signal)
        signal = torch.cat(signals, dim=2)
        signal = self.Norm(signal)
        signal = self.DW_conv(signal.reshape(-1, self.image_size_h, self.image_size_w, self.embedding_dim*self.heads).permute(0, 3, 1, 2)).reshape(-1, self.embedding_dim*self.heads, self.image_size_h*self.image_size_w).permute(0, 2, 1)
        return signal, losses

@MODELS.register_module()
class Transmitter(BaseModule):
    def __init__(self, image_size_h, image_size_w, code_size, code_dim, code_num, signal_split, multiheads, num_classes, init_limt=0.05, device='cuda'):
        super().__init__()
        self.mutiheads = multiheads
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8*2, kernel_size=3, stride=2),
        )
        self.conv1 = nn.Conv2d(in_channels=8*2, out_channels=8*4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8*4, out_channels=8*8, kernel_size=3, stride=2, padding=1)
        self.cells_1 = TransmitterBlock(code_size, 64, code_num, multiheads, signal_split, image_size_h//16, image_size_w//16, init_limt=init_limt, device=device)

        self.cells_2 = TransmitterBlock(code_size, 64, code_num, multiheads, signal_split, image_size_h//8, image_size_w//8, init_limt=init_limt, device=device)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        signal1, loss1 = self.cells_1(x3.permute(0, 2, 3, 1).reshape(-1, 128*64, 64))
        signal2, loss2 = self.cells_2(signal1)
        return signal2, loss1+loss2