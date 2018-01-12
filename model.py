import torch


class ModelRaw(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(ModelRaw, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv1d(1, 32, 5, stride=1),  # 44096
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.AvgPool1d(4, 4),  # 11024
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv1d(32, 64, 5, stride=1),  # 11020
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.AvgPool1d(4, 4),  # 2756
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv1d(64, 128, 100, stride=20),  # 133
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.AvgPool1d(40, 30)  # 4
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(512, 70),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(70),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(70, 30),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(30),
            torch.nn.Linear(30, 16),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1))
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
