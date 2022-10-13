import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import accuracy, calc_loss, get_cossim


# TODO: this class is already deprecated
class GE2ELoss(nn.Module):
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.device = device

    def forward(self, embeddings: Tensor) -> Tensor:
        torch.clamp(self.w, 1e-6)
        centroids = torch.mean(embeddings, dim=1)
        cossim = get_cossim(embeddings, centroids, self.cos)
        sim_matrix = self.w * cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss


class GE2ELoss_(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(GE2ELoss_, self).__init__()

        self.test_normalize = True

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Initialised GE2E')

    def forward(self, x: Tensor, label=None):

        assert x.size()[1] >= 2

        gsize = x.size()[1]
        centroids = torch.mean(x, 1)
        stepsize = x.size()[0]

        cos_sim_matrix = []

        for ii in range(0, gsize):
            idx = [*range(0, gsize)]
            idx.remove(ii)
            exc_centroids = torch.mean(x[:, idx, :], 1)
            cos_sim_diag = F.cosine_similarity(x[:, ii, :], exc_centroids)
            cos_sim = F.cosine_similarity(x[:, ii, :].unsqueeze(-1), centroids.unsqueeze(-1).transpose(0, 2))
            cos_sim[range(0, stepsize), range(0, stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim, 1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix, dim=1)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(np.asarray(range(0, stepsize))).cuda()
        nloss = self.criterion(cos_sim_matrix.view(-1, stepsize),
                               torch.repeat_interleave(label, repeats=gsize, dim=0).cuda())
        prec1 = accuracy(cos_sim_matrix.view(-1, stepsize).detach(),
                         torch.repeat_interleave(label, repeats=gsize, dim=0).detach(), topk=(1,))[0]
        return nloss, prec1

    def get_confidence(self, x: Tensor):
        assert x.size()[1] >= 2

        stepsize = x.size()[0]
        gsize = x.size()[1]  # in our case, gsize is number of  utterance
        centroids = torch.mean(x, 1)  # self-included centroids

        cos_sim_matrix = []

        for ii in range(0, gsize):
            idx = [*range(0, gsize)]
            idx.remove(ii)
            # Calculate the self-excluded centroids.
            exc_centroids = torch.mean(x[:, idx, :], 1)
            cos_sim_diag = F.cosine_similarity(x[:, ii, :], exc_centroids)
            cos_sim = F.cosine_similarity(x[:, ii, :].unsqueeze(-1), centroids.unsqueeze(-1).transpose(0, 2))
            cos_sim[range(0, stepsize), range(0, stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim, 1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix, dim=1)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(np.asarray(range(0, stepsize))).cuda()
        nloss = self.criterion(cos_sim_matrix.view(-1, stepsize),
                               torch.repeat_interleave(label, repeats=gsize, dim=0).cuda())
        prec1 = accuracy(cos_sim_matrix.view(-1, stepsize).detach(),
                         torch.repeat_interleave(label, repeats=gsize, dim=0).detach(), topk=(1,))[0]
        return nloss, prec1


class AAMSoftmax(nn.Module):
    """AAM Loss Criterion
    """

    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(AAMSoftmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f' % (self.m, self.s))

    # TODO: no one is calling this func? What is `label`?
    def predict(self, x: torch.Tensor):
        # assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        output = cosine * self.s
        return output

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


# FIXME: this is deprecated
class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 15.0 if not s else s
            self.m = 0.3 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * \
                torch.cos(
                    torch.acos(
                        torch.clamp(
                            torch.diagonal(
                                wf.transpose(
                                    0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * \
                torch.cos(
                    self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class SubcenterArcMarginProduct(nn.Module):
    r"""Modified implementation from
    https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
    """

    def __init__(self, in_features, out_features, K=3, s=30.0, m=0.50, easy_margin=False):
        super(SubcenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(out_features * self.K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.ce_loss = nn.CrossEntropyLoss()

    def predict(self, input: torch.Tensor):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            cosine, _ = torch.max(cosine, axis=2)

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # cos(phi+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > 0, phi, cosine)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            cosine, _ = torch.max(cosine, axis=2)

        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        # cos(phi+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)
        loss = self.ce_loss(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
