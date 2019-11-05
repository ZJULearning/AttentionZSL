import sys
import torch
import numpy as np
import torch.nn.functional as F

_eps = 1e-15

class OnlineZeroShotLearningLoss(torch.nn.Module):
    def __init__(self, all_attr):
        super(OnlineZeroShotLearningLoss, self).__init__()
        self.zsl_loss = ZeroShotLearningLoss(all_attr)

    def forward(self, embeddings, mask, selection):
        """
        Input:  
            embeddings: N * (k + 2) * d
        Output:
            loss: scalar
        """
        anchor = embeddings[:, 0, :]
        positive = embeddings[:, 1, :]
        negative = embeddings[:, 2:, :].contiguous().view(-1, embeddings.size(2))[selection, :]

        mask_anchor = mask[:, 0, :]
        mask_positive = mask[:, 1, :]
        mask_negative = mask[:, 2:, :].contiguous().view(-1, mask.size(2))[selection, :]

        final_loss = self.zsl_loss(anchor, mask_anchor) + \
                     self.zsl_loss(positive, mask_positive) + \
                     self.zsl_loss(negative, mask_negative)
        return final_loss

class ZeroShotLearningLoss(torch.nn.Module):
    def __init__(self, all_attr, temperature=1):
        super(ZeroShotLearningLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(1)
        self.all_attr = all_attr
        self.temperature = temperature

    def forward(self, x, attr_mask):
        """
        Input:  
            x:          N * d
            attr_mask   N * #cls 
            self.all_attr:   #cls * d
        Output:
            loss: scalar
        """
        batch_size = x.size(0)
        feature_dim = x.size(1)
        all_similarity = torch.matmul(x, self.all_attr.transpose(0, 1))

        softmax_like_loss = (self.log_softmax(all_similarity * self.temperature) * attr_mask).sum()
        final_loss = -softmax_like_loss / batch_size
        return final_loss

class OnlineTripletLoss(torch.nn.Module):
    def __init__(self, device, preset_margin=1.0):
        super(OnlineTripletLoss, self).__init__()
        self.device = device
        self.margin = preset_margin
        self.triplet_loss = TripletLoss(self.device, self.margin)

    def forward(self, embeddings):
        """
        Input:
            embeddings: N * (2 + k) * d
        Output:
            loss: scalar
        """
        anchor = embeddings[:, 0, :]
        positive = embeddings[:, 1, :]
        negative = embeddings[:, 2:, :].contiguous().view(-1, embeddings.size(2))
        
        final_loss, neg_selection = self.triplet_loss(anchor, positive, negative)
        return final_loss, neg_selection

def triplet_combination_list(p, k):
    anchor_y = [[i] * (k - 1) * (p * k - k) for i in range(k)]
    anchor_y = torch.LongTensor(sum(anchor_y, []) * p)
    positive_y = sum([sum([[i] * (p - 1) * k for i in range(k) if i != j], []) for j in range(k)], [])
    positive_y = torch.LongTensor(positive_y * p)

    anchor_x = [[i] * k * (k - 1) * (p * k - k) for i in range(p)]
    anchor_x = torch.LongTensor(sum(anchor_x, []))
    positive_x = [[i] * k * (k - 1) * (p * k - k) for i in range(p)]
    positive_x = torch.LongTensor(sum(positive_x, []))
   
    negative_y = torch.LongTensor(list(range(k)) * ((p - 1) * k * p * (k - 1)))
    negative_x = [[i for i in range(p) if i != j] * k * (k - 1) for j in range(p)]
    negative_x = torch.LongTensor(np.repeat(np.array(negative_x), k).tolist()) 
    
    return anchor_x, anchor_y, positive_x, positive_y, negative_x, negative_y

class BatchAllTripletLoss(torch.nn.Module):
    def __init__(self, device, p, k, preset_margin=1.0):
        super(BatchAllTripletLoss, self).__init__()
        self.device = device
        self.margin = preset_margin
        self.ax, self.ay, self.px, self.py, self.nx, self.ny = \
                triplet_combination_list(p, k)
        self.ax = self.ax.to(self.device)
        self.ay = self.ay.to(self.device)
        self.px = self.px.to(self.device)
        self.py = self.py.to(self.device)
        self.nx = self.nx.to(self.device)
        self.ny = self.ny.to(self.device)

    def forward(self, embeddings):
        """Input:
            embeddings: P x K x #embedding_dims
            P classes and K different images within each class
        """
        ap_loss = torch.sqrt(((embeddings[self.ax, self.ay, :] - \
                embeddings[self.px, self.py, :]) ** 2).sum(1) + _eps)
        an_loss = torch.sqrt(((embeddings[self.ax, self.ay, :] - \
                embeddings[self.nx, self.ny, :]) ** 2).sum(1) + _eps)
        loss = F.relu(ap_loss - an_loss + self.margin)
        return loss.sum() / (loss.nonzero().size(0) + _eps)

class BatchHardTripletLoss(torch.nn.Module):
    def __init__(self, device, p, k, preset_margin=1.0):
        super(BatchHardTripletLoss, self).__init__()
        self.device = device
        self.margin = preset_margin
        self.ax, self.ay, self.px, self.py, self.nx, self.ny = \
                triplet_combination_list(p, k)
        self.ax = self.ax.to(self.device)
        self.ay = self.ay.to(self.device)
        self.px = self.px.to(self.device)
        self.py = self.py.to(self.device)
        self.nx = self.nx.to(self.device)
        self.ny = self.ny.to(self.device)

        self.p = p
        self.k = k

    def forward(self, embeddings):
        """Input:
            embeddings: P x K x #embedding_dims
            P classes and K different images within each class
        """
        ap_loss = torch.sqrt(((embeddings[self.ax, self.ay, :] - \
                embeddings[self.px, self.py, :]) ** 2).sum(1) + _eps)
        an_loss = torch.sqrt(((embeddings[self.ax, self.ay, :] - \
                embeddings[self.nx, self.ny, :]) ** 2).sum(1) + _eps)
        ap_loss = ap_loss.view(-1, (self.p - 1) * self.k * (self.k - 1))
        an_loss = an_loss.view(-1, (self.p - 1) * self.k * (self.k - 1))
        ap_loss, _ = ap_loss.max(1)
        an_loss, _ = an_loss.min(1)

        loss = F.relu(ap_loss - an_loss + self.margin)
        return loss.sum() / (self.p * self.k)

class TripletLoss(torch.nn.Module):
    def __init__(self, device, preset_margin=1.0):
        super(TripletLoss, self).__init__()
        self.device = device
        self.margin = preset_margin

    def forward(self, anchor, pos, neg):
        """
        Input:
            anchor: N * d
            pos: N * d
            neg: kN * d (employ argmin)
        Output:
            loss: scalar
        """
        d = anchor.size(1)
        batch_size = anchor.size(0)
        
        neg_selection_k = int(neg.size(0) / batch_size)
        neg2 = neg.view(batch_size, -1, d)
        anchor_repeat = anchor.unsqueeze(1).repeat(1, neg_selection_k, 1)

        neg_selection_loss = (anchor_repeat - neg2) ** 2
        neg_selection_loss = torch.sqrt(neg_selection_loss.sum(2) + _eps) # N * k
        
        an_loss, n_indices = neg_selection_loss.min(1)
        base = (torch.ones(batch_size, dtype=torch.long).cumsum(0) - 1) * neg_selection_k
        base = base.to(self.device)
        n_indices = base + n_indices
        ap_loss = torch.sqrt(((anchor - pos) ** 2).sum(1) + _eps)

        loss = F.relu(ap_loss - an_loss + self.margin).mean()
        return loss, n_indices

if __name__ == "__main__":
    p = 3
    k = 4
    ax, ay, px, py, nx, ny = triplet_combination_list(p, k)
    array = np.arange(p * k)
    array.resize(p, k)

    print(array)

    for i in range(len(ax)):
        anchor = array[ax[i], ay[i]]
        positive = array[px[i], py[i]]
        negative = array[nx[i], ny[i]]
        print(anchor, positive, negative)

