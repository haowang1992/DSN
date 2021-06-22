from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from scipy.spatial.distance import cdist

import time

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def ITQ(V, n_iter):
    # Main function for  ITQ which finds a rotation of the PCA embedded data
    # Input:
    #     V: nxc PCA embedded data, n is the number of images and c is the code length
    #     n_iter: max number of iterations, 50 is usually enough
    # Output:
    #     B: nxc binary matrix
    #     R: the ccc rotation matrix found by ITQ
    # Publications:
    #     Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A
    #     Procrustes Approach to Learning Binary Codes. In CVPR 2011.
    # Initialize with a orthogonal random rotation initialize with a orthogonal random rotation

    bit = V.shape[1]
    np.random.seed(n_iter)
    R = np.random.randn(bit, bit)
    U11, S2, V2 = np.linalg.svd(R)
    R = U11[:, :bit]

    # ITQ to find optimal rotation
    for iter in range(n_iter):
        Z = np.matmul(V, R)
        UX = np.ones((Z.shape[0], Z.shape[1])) * -1
        UX[Z >= 0] = 1
        C = np.matmul(np.transpose(UX), V)
        UB, sigma, UA = np.linalg.svd(C)
        R = np.matmul(UA, np.transpose(UB))

    # Make B binary
    B = UX
    B[B < 0] = 0
    return B, R

def compressITQ(Xtrain, Xtest, n_iter=50):
    # compressITQ runs ITQ
    # Center the data, VERY IMPORTANT
    Xtrain = Xtrain - np.mean(Xtrain, axis=0, keepdims=True)
    Xtest = Xtest - np.mean(Xtest, axis=0, keepdims=True)

    # PCA
    C = np.cov(Xtrain, rowvar=False)
    l, pc = np.linalg.eigh(C, 'U')
    idx = l.argsort()[::-1]
    pc = pc[:, idx]
    XXtrain = np.matmul(Xtrain, pc)
    XXtest = np.matmul(Xtest, pc)

    # ITQ
    _, R = ITQ(XXtrain, n_iter)

    Ctrain = np.matmul(XXtrain, R)
    Ctest = np.matmul(XXtest, R)

    Ctrain = Ctrain > 0
    Ctest = Ctest > 0

    return Ctrain, Ctest

def validate_paired(val_loader, model, criterion, criterion_kd, model_t, test_num=300):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kd = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_t.eval()
    end = time.time()
    sketch_hash_all = []
    image_hash_all = []
    target_all = []

    for i, (sketch, image, target) in enumerate(val_loader):
        if i == test_num:
            break
        sketch = torch.autograd.Variable(sketch, requires_grad=False).cuda()
        image = torch.autograd.Variable(image, requires_grad=False).cuda()
        target = target.type(torch.LongTensor).view(-1, )
        target = torch.autograd.Variable(target).cuda()

        # compute output
        with torch.no_grad():
            output, output_kd, hash_code, _ = model(torch.cat([sketch, image], 0),
                                         torch.cat([torch.zeros(sketch.size()[0], 1), torch.ones(image.size()[0], 1)], 0).cuda())

        loss = criterion(output, torch.cat([target, target]))

        # measure accuracy and record loss
        hash_code = F.normalize(hash_code).cpu().detach().numpy()
        sketch_hash = hash_code[:sketch.size(0)].reshape(sketch.size(0), -1)
        image_hash = hash_code[sketch.size(0):].reshape(image.size(0), -1)
        sketch_hash_all.append(sketch_hash)
        image_hash_all.append(image_hash)
        target_all.append(target.cpu().detach().numpy())

        losses.update(loss.item(), sketch.size(0) * 2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0 or i == len(val_loader) - 1:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f}({loss.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

    # compute mAP
    sketch_hash_all = np.concatenate(sketch_hash_all)
    image_hash_all = np.concatenate(image_hash_all)
    target_all = np.concatenate(target_all)
    binary_sketch_hash, binary_image_hash = compressITQ(sketch_hash_all, image_hash_all)
    scores = -cdist(binary_sketch_hash, binary_image_hash)
    mAP_ls = [[] for _ in range(np.unique(target_all).max() + 1)]
    for fi in range(sketch_hash_all.shape[0]):
        mapi = eval_AP_inner(target_all[fi], scores[fi], target_all)
        mAP_ls[target_all[fi]].append(mapi)

    mAP = np.array([np.nanmean(maps) for maps in mAP_ls if len(maps) != 0]).mean()
    print(' * mAP@all {mAP:.4f}'
          .format(mAP=mAP))

    return mAP

def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap


def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top


class MemoryStore(object):
    def __init__(self, n_classes, topk, embed_dim):
        self.n_classes = n_classes
        self.topk = topk
        self.embed_dim = embed_dim
        self.device = torch.device('cuda')

        # Initializing memory to blank embeddings and "n_classes = not seen" labels
        self.memory_embeds = torch.zeros(
            (self.n_classes, self.topk, self.embed_dim)).to(self.device)
        self.memory_full_flag = torch.zeros(self.n_classes).to(self.device).type(torch.int8)
        # self.image_num_flag = torch.zeros(self.n_classes).to(self.device)


    def get_memory_used_percent(self):
        return 100 * (torch.sum(self.memory_full_flag).float() / (self.topk * self.n_classes)).item()


    def get_class_center(self, label):
        if self.memory_full_flag[label] == 0:
            return
        else:
            return torch.mean(self.memory_embeds[label][:self.memory_full_flag[label], :], 0)

    def add_entries(self, embeds, labels, queries, query_labels):
        """
        Args:
            embed: a torch tensor with (batch, embed_dim) size
            label: a torch tensor with (batch) size
        Returns:
            None
        """
        for idx, embed in enumerate(embeds):
            saved_embed_nums = self.memory_full_flag[labels[idx]].type(torch.int32)
            if saved_embed_nums < self.topk:
                self.memory_embeds[labels[idx]][saved_embed_nums] = embed
                self.memory_full_flag[labels[idx]] += 1
            else:
                corresponding_idx = torch.nonzero(query_labels == labels[idx]).view(-1)
                if corresponding_idx.size() == 0:
                    # if self.image_num_flag[labels[idx]] != 0:
                    #     query = self.memory_embeds[labels[idx]][self.topk]
                    # else:
                    #     continue
                    continue
                else:
                    query = torch.mean(queries[corresponding_idx], 0)

                score, furthest_idx = self.get_furthest_entry(query, labels[idx])

                if F.cosine_similarity(query.view(1, -1), embed.view(1, -1))  > score:
                    self.memory_embeds[furthest_idx] = embed


    def get_furthest_entry(self, query, label):
        """
        Reference code for pairwise distance computation:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
        Args:
            queries: a torch array with (n_queries, embed_dim) size
        Returns:
            embeds: a torch array with (n_queries, max_neighbours, embed_dim) size
            labels: a torch array with (n_queries, max_neighbours) size
        """
        scores = F.cosine_similarity(self.memory_embeds[label], query.view(1, -1))
        score, idx = torch.min(scores, 0)
        return score, idx

    def memory_loss(self, queries, labels):
        loss = []
        for idx, query in enumerate(queries):
            center = self.get_class_center(labels[idx])
            if center is not None:
                loss.append(F.cosine_similarity(query.view(1, -1), center.view(1, -1)))
        if len(loss) == 0:
            return torch.Tensor([1.0]).cuda() # to make the loss invalid
        return sum(loss) / len(loss)

    def flush(self):
        self.memory_embeds = torch.zeros(
            (self.n_classes, self.topk, self.embed_dim)).to(self.device)
        self.memory_full_flag = torch.zeros(self.n_classes).to(self.device).type(torch.int8)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count