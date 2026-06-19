import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()    
        self.features = models.efficientnet_b0(models.EfficientNet_B0_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280                     # effecientnet feature output dimension

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)                    # (B, out_dim, 1, 1)
        x = x.flatten(1)                    # (B, out_dim)
        return x

class AgeNet(nn.Module):
    def __init__(self, d_model=1280, nhead=8):
        super().__init__()
        # positional encodings 
        self.pos_embedding = nn.Parameter(torch.zeros(1, 8, d_model))      # learnable positional encodings
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)                # initialize them using truncated normal distribution.
        # encoders
        self.backbone = CNNEncoder()
        self.transformerEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=2
        )
        # projection layer to match dimensions if needed
        out_dim = self.backbone.out_dim
        self.proj = nn.Linear(out_dim, d_model) if out_dim != d_model else nn.Identity()
        # output heads
        self.person_head = nn.Linear(d_model, 1)
        self.age_head    = nn.Linear(d_model, 1)
    
    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.flatten(0, 1)                 # (B*T, C, H, W)
        x = self.backbone(x)                # (B*T, dim_out)
        x = self.proj(x)                    # (B*T, d)
        x = x.view(B, T, -1)                # (B, T, d)
        x = x + self.pos_embedding          # (B, T, d)
        x = self.transformerEncoder(x)      # (B, T, d)
        x = x.mean(dim=1)                   # (B, d)

        person = self.person_head(x)        # (B, 1)
        age = self.age_head(x)              # (B, 1)
        
        return (person, age)
    
    def freeze_backbone(self, isFreeze=True):
        if isFreeze:
            self.backbone.eval()
        else:
            self.backbone.train()

        for param in self.backbone.parameters():
            param.requires_grad = not isFreeze
    

class MultitaskLoss(nn.Module):
    def __init__(self, lambda1=1., lambda2=1.):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, y_pred, y_true):
        person_pred, age_pred = y_pred
        person_true, age_true = y_true
        # mask for true persons
        isPerson = person_true == 1
        # compute loss
        person_loss = self.bce(person_pred, person_true.unsqueeze(1))
        age_loss    = None
        # saftey check incase all is non-person so MSELoss don't crash.
        if isPerson.sum() > 0:
            age_loss = self.mse(age_pred[isPerson], age_true[isPerson].unsqueeze(1))
        else:
            torch.tensor(0.0, device=person_pred.device)

        return (self.lambda1 * person_loss) + (self.lambda2 * age_loss)

