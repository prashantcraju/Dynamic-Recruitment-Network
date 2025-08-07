import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedDRNLayer(nn.Module):
    def __init__(self, input_size, output_size, num_pops=4, k_active=1):
        super().__init__()
        self.num_pops = num_pops
        self.k_active = k_active  # Number of active populations during inference

        self.pops = nn.ModuleList([
            nn.Linear(input_size, output_size) for _ in range(num_pops)
        ])
        self.selector = nn.Linear(input_size, num_pops)

    def forward(self, x):
        # Population selection
        logits = self.selector(x)
        probs = F.softmax(logits, dim=-1)

        if self.training:
            # Soft selection: weighted sum of all populations
            outputs = torch.stack([pop(x) for pop in self.pops], dim=1)
            probs_expanded = probs.unsqueeze(-1)
            out = (outputs * probs_expanded).sum(dim=1)
        else:
            # Hard selection: top-k populations
            topk = probs.topk(self.k_active, dim=-1).indices
            batch_outs = []
            for i in range(x.size(0)):
                x_i = x[i]
                selected_outs = [self.pops[j](x_i) for j in topk[i]]
                avg_out = torch.stack(selected_outs).mean(dim=0)
                batch_outs.append(avg_out)
            out = torch.stack(batch_outs)

        return out


class SimplifiedDRN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_pops=4, k_active=1):
        super().__init__()
        layers = []
        sizes = [input_size] + hidden_sizes

        for i in range(len(hidden_sizes)):
            layers.append(
                SimplifiedDRNLayer(sizes[i], sizes[i+1], num_pops, k_active)
            )

        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.classifier(x)
