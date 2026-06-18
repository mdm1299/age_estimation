import torch
import torch.nn as nn

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

