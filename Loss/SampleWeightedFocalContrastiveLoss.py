import torch
import torch.nn as nn
from torch.nn.functional import normalize




'''
Sample-Weighted Focal Contrastive (SWFC) Loss:
1. Divide training samples into positive and negative pairs to maximize 
inter-class distances while minimizing intra-class distances;
2. Assign more importance to hard-to-classify positive pairs;
3. Assign more importance to minority classes. 
'''
class SampleWeightedFocalContrastiveLoss(nn.Module):

    def __init__(self, temp_param, focus_param, sample_weight_param, dataset, class_counts, device):
        '''
        temp_param: control the strength of penalty on hard negative samples;
        focus_param: forces the model to concentrate on hard-to-classify samples;
        sample_weight_param: control the strength of penalty on minority classes;
        dataset: MELD or IEMOCAP.
        device: cpu or cuda. 
        '''
        super().__init__()
        
        self.temp_param = temp_param
        self.focus_param = focus_param
        self.sample_weight_param = sample_weight_param
        self.dataset = dataset
        self.class_counts = class_counts
        self.device = device

        if self.dataset == 'MELD':
            self.num_classes = 7
        elif self.dataset == 'IEMOCAP':
            self.num_classes = 6
        else:
            raise ValueError('Please choose either MELD or IEMOCAP')
        
        self.class_weights = self.get_sample_weights()
    

    '''
    Use dot-product to measure the similarity between feature pairs.
    '''
    def dot_product_similarity(self, current_features, feature_sets):
        similarity = torch.sum(current_features * feature_sets, dim = -1)
        similarity_probs = torch.softmax(similarity / self.temp_param, dim = 0)

        return similarity_probs
    

    '''
    Calculate the loss contributed from positive pairs.
    '''
    def positive_pairs_loss(self, similarity_probs):
        pos_pairs_loss = torch.mean(torch.log(similarity_probs) * ((1 - similarity_probs)**self.focus_param), dim = 0)

        return pos_pairs_loss


    '''
    Assign more importance to minority classes. 
    '''
    def get_sample_weights(self):
        total_counts = torch.sum(self.class_counts, dim = -1)
        class_weights = (total_counts / self.class_counts)**self.sample_weight_param
        class_weights = normalize(class_weights, dim = -1, p = 1.0)

        return class_weights
        

    def forward(self, features, labels):
        self.num_samples = labels.shape[0]
        self.feature_dim = features.shape[-1]

        features = normalize(features, dim = -1)  # normalization helps smooth the learning process

        batch_sample_weights = torch.FloatTensor([self.class_weights[label] for label in labels]).to(self.device)

        total_loss = 0.0
        for i in range(self.num_samples):
            current_feature = features[i]
            current_label = labels[i]
            feature_sets = torch.cat((features[:i], features[i + 1:]), dim = 0)
            label_sets = torch.cat((labels[:i], labels[i + 1:]), dim = 0)
            expand_current_features = current_feature.expand(self.num_samples - 1, self.feature_dim).to(self.device)
            similarity_probs = self.dot_product_similarity(expand_current_features, feature_sets)
            pos_similarity_probs = similarity_probs[label_sets == current_label]  # positive pairs with the same label
            if len(pos_similarity_probs) > 0:
                pos_pairs_loss = self.positive_pairs_loss(pos_similarity_probs)
                weighted_pos_pairs_loss = pos_pairs_loss * batch_sample_weights[i]
                total_loss += weighted_pos_pairs_loss
        
        loss = - total_loss / self.num_samples

        return loss