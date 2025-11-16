import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    This loss encourages samples from the same class to be closer in the embedding space,
    while pushing samples from different classes apart. It is commonly used in supervised
    contrastive learning for representation learning tasks.

    Args:
        temperature (float, optional): Temperature scaling factor for the similarity scores. Default: 0.07.

    Forward Args:
        features (Tensor): Feature representations of shape (batch_size, feature_dim).
        labels (Tensor): Ground truth labels of shape (batch_size,).

    Returns:
        Tensor: Scalar loss value.

    Usage:
        >>> criterion = SupConLoss(temperature=0.07)
        >>> features = torch.randn(32, 128)  # batch of 32 samples, 128-dim features
        >>> labels = torch.randint(0, 10, (32,))  # batch of 32 labels (10 classes)
        >>> loss = criterion(features, labels)
    """
    def __init__(self, temperature=0.07):

        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = features.device
        batch_size = features.shape[0]

        # 1) L2 归一化
        features = F.normalize(features, p=2, dim=1)

        # 2) 计算相似度矩阵：batch × batch
        sim_matrix = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # 3) 构造 mask，标记正样本对（同类别且非自身）
        labels = labels.contiguous().view(-1,1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # 4) 对每一个样本 i，logits_i_j = exp(sim_ij) / sum_k≠i exp(sim_ik)
        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(batch_size, device=device))
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # 5) 只保留正对，计算 loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = - mean_log_prob_pos
        return loss.mean()