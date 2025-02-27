# Adapted from https://github.com/facebookresearch/moco-v3

import torch
import torch.nn as nn
from math import ceil

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, swin=False, CTP=False, resnext=False):
        super(MoCo, self).__init__()

        self.T = T

        if not swin and not resnext:
            self.base_encoder = base_encoder(num_classes=mlp_dim)
            self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        else:
            if CTP:
                self.base_encoder = base_encoder(pretrained=True)
                self.momentum_encoder = base_encoder(pretrained=True)
            else:
                self.base_encoder = base_encoder()
                self.momentum_encoder = base_encoder()

        self._build_projector_and_predictor_mlps(dim, mlp_dim, resnext=resnext)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
    
    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
    
    def contrastive_loss_lesion(self, q, k, lesion_mask, sample_weight):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]
        offset = N * torch.distributed.get_rank()
        lesion_idxs = lesion_mask.nonzero()
        if len(lesion_idxs)>0:
            lesion_idxs = lesion_idxs.squeeze(1).unsqueeze(0).repeat(N,1)
        diag = logits.diag(offset)
        if len(lesion_idxs)>0:
            return -(diag.view(-1,1).double().logsumexp(1)-torch.concat(((sample_weight*logits).diagonal_scatter(-torch.ones(N)*float("Inf"),offset).gather(1,lesion_idxs),logits),
                                                             dim=1).double().logsumexp(1)).double().mean() * (2 * self.T)
        else:
            return -(diag.view(-1,1).double().logsumexp(1)-logits.double().logsumexp(1)).double().mean() * (2 * self.T)
        
    def forward(self, x1, x2, m, lesion_mask=None, sample_weight=10, sampling="None"):
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():
            self._update_momentum_encoder(m)
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
                
            if sampling=="Lesion" and sample_weight>0:
                lesion_mask = concat_all_gather(lesion_mask)
            
        if sampling=="Lesion":      
            if sample_weight>0:
                return self.contrastive_loss_lesion(q1,k2,lesion_mask,sample_weight) + self.contrastive_loss_lesion(q2,k1,lesion_mask,sample_weight)
            else:
                return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        else:
            return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, resnext):
        if resnext:
            hidden_dim = self.base_encoder.head.fc.weight.shape[1]
            del self.base_encoder.head.fc, self.momentum_encoder.head.fc
            self.base_encoder.head.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
            self.momentum_encoder.head.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        
        else:
            hidden_dim = self.base_encoder.fc.weight.shape[1]
            del self.base_encoder.fc, self.momentum_encoder.fc
            self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
            self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim,resnext):
        if resnext:
            try:
                hidden_dim = self.base_encoder.head.fc.weight.shape[1]
            except Exception as exc:
                print(self.base_encoder)
                print(exc)
            del self.base_encoder.head.fc, self.momentum_encoder.head.fc
            self.base_encoder.head.fc = self._build_mlp(3, hidden_dim, mlp_dim, dim)
            self.momentum_encoder.head.fc = self._build_mlp(3, hidden_dim, mlp_dim, dim)
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim)
        
        else:
            try:
                hidden_dim = self.base_encoder.head.weight.shape[1]
                del self.base_encoder.head, self.momentum_encoder.head
                self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
                self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
            except Exception as exc:
                hidden_dim = 768
                self.base_encoder.classifier.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
                self.momentum_encoder.classifier.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

@torch.no_grad()
def concat_all_gather(tensor,dim=0):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output
