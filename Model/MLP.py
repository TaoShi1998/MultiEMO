import torch.nn as nn




'''
2-layer MLP with ReLU activation.
'''
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x):
        return self.dropout(self.linear_2(self.relu(self.linear_1(x))))