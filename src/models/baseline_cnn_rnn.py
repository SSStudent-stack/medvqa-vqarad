import torch
import torch.nn as nn
import torchvision.models as tvm

class BaselineVQA(nn.Module):
    def __init__(self, vocab_size: int, open_num_classes: int, q_emb_dim: int = 256, q_hidden: int = 256):
        super().__init__()
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # (B,512,1,1)
        self.img_dim = 512

        self.emb = nn.Embedding(vocab_size, q_emb_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=q_emb_dim, hidden_size=q_hidden, batch_first=True)
        self.q_dim = q_hidden

        self.fusion = nn.Sequential(
            nn.Linear(self.img_dim + self.q_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.closed_head = nn.Linear(512, 2)
        self.open_head = nn.Linear(512, open_num_classes)

    def forward(self, image, q_ids):
        img_feat = self.cnn(image).flatten(1)
        q_emb = self.emb(q_ids)
        _, h = self.gru(q_emb)
        q_feat = h.squeeze(0)
        fused = self.fusion(torch.cat([img_feat, q_feat], dim=1))
        return self.closed_head(fused), self.open_head(fused)
