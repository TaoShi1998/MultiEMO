import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np




'''
label index mapping = {'happiness': 0, 'sadness': 1, 'neutral': 2, 'anger': 3, 'excitement': 4, 'frustration': 5}
'''
class IEMOCAPDataset(Dataset):

    def __init__(self, train = True):
        _, self.videoSpeakers, self.videoLabels, _, _, _, _, self.trainVid,\
        self.testVid = pickle.load(open('Data/IEMOCAP/Speakers.pkl', 'rb'), encoding='latin1')

        '''
        Textual features are extracted using pre-trained EmoBERTa. If you want to extract textual
        features on your own, please visit https://github.com/tae898/erc
        '''
        self.videoText = pickle.load(open('Data/IEMOCAP/TextFeatures.pkl', 'rb'))
        self.videoAudio = pickle.load(open('Data/IEMOCAP/AudioFeatures.pkl', 'rb'))
        self.videoVisual = pickle.load(open('Data/IEMOCAP/VisualFeatures.pkl', 'rb'))

        self.trainVid = sorted(self.trainVid)
        self.testVid = sorted(self.testVid)

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)


    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(np.array(self.videoText[vid])),\
            torch.FloatTensor(np.array(self.videoAudio[vid])),\
                torch.FloatTensor(np.array(self.videoVisual[vid])),\
                    torch.FloatTensor(np.array([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]])),\
                        torch.FloatTensor(np.array([1]*len(self.videoLabels[vid]))),\
                            torch.LongTensor(np.array(self.videoLabels[vid]))


    def __len__(self):
        return self.len


    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        output = []
        for i in dat:
            temp = dat[i].values
            if i <= 3:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], padding_value = 0)) 
            elif i <= 4:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], True, padding_value = 0))
            elif i <= 5:
                output.append(pad_sequence([temp[i] for i in range(len(temp))], True, padding_value = -1))

        return output