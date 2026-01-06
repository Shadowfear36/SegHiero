import torch.nn as nn
import torch
import torch.nn.functional as F

class TreeTripletLoss(nn.Module):
    def __init__(self, num_classes, upper_ids, lower_ids, ignore_index=255):
        super(TreeTripletLoss, self).__init__()

        self.ignore_label = ignore_index
        self.num_classes = num_classes
        self.upper_ids = upper_ids
        self.lower_ids = lower_ids

    def forward(self, feats, labels=None, max_triplet=200):
        batch_size = feats.shape[0]
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        labels = labels.view(-1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(-1, feats.shape[-1])
        
        triplet_loss=0
        exist_classes = torch.unique(labels)
        exist_classes = [x for x in exist_classes if x != 255 and x!=0]
        class_count=0
        
        for ii in exist_classes:
            index_anchor = labels==ii
            if ii in self.upper_ids:
                label_pos = self.upper_ids.copy()
                label_neg = self.lower_ids.copy()
            else:
                label_pos = self.lower_ids.copy()
                label_neg = self.upper_ids.copy()
            label_pos.remove(ii)
            index_pos = torch.zeros_like(index_anchor)
            index_neg = torch.zeros_like(index_anchor)
            for pos_l in label_pos:
                index_pos += labels==pos_l
            for neg_l in label_neg:
                index_neg += labels==neg_l
            
            min_size = min(torch.sum(index_anchor), torch.sum(index_pos), torch.sum(index_neg), max_triplet)
            
            feats_anchor = feats[index_anchor][:min_size]
            feats_pos = feats[index_pos][:min_size]
            feats_neg = feats[index_neg][:min_size]
            
            distance = torch.zeros(min_size,2).cuda()
            distance[:,0:1] = 1-(feats_anchor*feats_pos).sum(1, True) 
            distance[:,1:2] = 1-(feats_anchor*feats_neg).sum(1, True) 
            
            # margin always 0.1 + (4-2)/4 since the hierarchy is three level
            # TODO: should include label of pos is the same as anchor, i.e. margin=0.1
            margin = 0.6*torch.ones(min_size).cuda()
            
            tl = distance[:,0] - distance[:,1] + margin
            tl = F.relu(tl)

            if tl.size(0)>0:
                triplet_loss += tl.mean()
                class_count+=1
        if class_count==0:
            return None, torch.tensor([0]).cuda()
        triplet_loss /=class_count
        return triplet_loss, torch.tensor([class_count]).cuda()
