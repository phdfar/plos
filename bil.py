import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

class DavisBouldinLoss(nn.Module):
    def __init__(self):
        super(DavisBouldinLoss, self).__init__()

    def forward(self, tensor, KK):
        B, C, T, H, W = tensor.shape
        total_loss = 0.0

        for i in range(B):
            # Flatten T, H, W dimensions, keeping C as the feature dimension
            flattened = tensor[i].permute(1, 2, 3, 0).reshape(-1, C).detach().cpu().numpy()  # Convert to numpy array
            try:
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=KK[i], random_state=0).fit(flattened)
                # Compute Davis-Bouldin Index
                labels = kmeans.labels_
                if len(set(labels)) > 1:  # Check to avoid davies_bouldin_score error for single cluster
                    score = davies_bouldin_score(flattened, labels)
                else:
                    score = float(1.0)  # Worst case if only one cluster found
            except:
                score = float(1.0)
                
            # Calculate loss as Davis-Bouldin Index
            loss = score
            total_loss += loss

        # Average the loss over the batch
        avg_loss = total_loss / B
        return torch.tensor(avg_loss, requires_grad=True, device=tensor.device)
