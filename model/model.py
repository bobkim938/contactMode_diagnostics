import torch
from torch import nn

"""
input: x_t = normalize((qx,qy,vx,vy,Fx,Fy)), [batch, 6]
output: delta_x = [delta_qx, delta_qy, delta_vx, delta_vy], [batch, 4]
"""

class dynamicsMLP(nn.Module):
    def __init__(self, inputDim=6, hiddenDim=128, outputDim=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.SiLU(),

            nn.Linear(hiddenDim, hiddenDim),
            nn.SiLU(),

            nn.Linear(hiddenDim, hiddenDim),
            nn.SiLU(),

            nn.Linear(hiddenDim, outputDim),
        )

    def forward(self, inputs):
        return self.network(inputs)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import E1Dataset

    torch.manual_seed(42)

    train_dataset = E1Dataset(
        roots="run_001",
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    model = dynamicsMLP()

    inputs, targets = next(iter(train_loader))

    model.eval()
    with torch.no_grad():
        predictions = model(inputs)

    print(model)
    print("Inputs:", inputs.shape)
    print("Targets:", targets.shape)
    print("Predictions:", predictions.shape)

    parameter_count = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print("Trainable parameters:", parameter_count)

    assert predictions.shape == targets.shape
    assert torch.isfinite(predictions).all()
        


