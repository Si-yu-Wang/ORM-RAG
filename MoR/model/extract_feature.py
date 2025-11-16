import torch

def get_embeddings(model, device, masks, images):
    """
    Extracts and aggregates feature embeddings from a batch of images using the given model.

    Args:
        model (torch.nn.Module): The neural network model used to extract features from images.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        masks (torch.Tensor): A boolean or float tensor of shape [B, N_max] indicating valid (non-padding) images in each batch.
        images (torch.Tensor): A tensor of shape [B, N_max, C, H, W] containing image data.

    Returns:
        torch.Tensor: Aggregated feature embeddings of shape [B, dim] for each sample in the batch.

    Usage:
        >>> agg_feats = get_embeddings(model, device, masks, images)
        # agg_feats: [B, dim], where each row is the mean feature embedding of valid images in the batch.
    """
    B, N_max, C, H, W = images.shape
    images = images.view(B * N_max, C, H, W).to(device)
    # labels = labels.to(device)
    masks  = masks.to(device)

    # ---- 前向：提取特征 ----
    outputs = model(images)                     # [B*N_max, seq, dim]
    feats   = outputs.last_hidden_state.mean(1) # [B*N_max, dim]

    # 恢复成 [B, N_max, dim]
    feats = feats.view(B, N_max, -1)

    # 用 mask 把 padding 的槽位去掉
    masks = masks.unsqueeze(-1)                 # [B, N_max, 1]
    feats = feats * masks                       # padding 部分会被置 0

    # 聚合：对每个样本真实的图像平均
    valid_counts = masks.sum(1)                 # [B, 1]
    agg_feats    = feats.sum(1) / valid_counts  # [B, dim]
    return agg_feats

def extract_embeddings(model, images):
    """
    Extracts and aggregates feature embeddings from a batch of images using the provided model.

    The function reshapes the input images, passes them through the model to obtain feature representations,
    performs pooling over the sequence dimension, and aggregates features across multiple images per sample.

    Args:
        model (torch.nn.Module): The neural network model used to extract features. 
            The model is expected to return an object with a `last_hidden_state` attribute of shape 
            [batch_size * num_images, seq_len, feature_dim].
        images (torch.Tensor): A 5D tensor of shape [batch_size, num_images, channels, height, width] 
            representing a batch of image sets.

    Returns:
        torch.Tensor: A 2D tensor of shape [batch_size, feature_dim] containing the aggregated feature 
            embeddings for each sample in the batch.

    Usage:
        >>> features = extract_embeddings(model, images)
        >>> # features.shape == (batch_size, feature_dim)
    """
    reshaped_images = images.view(-1, images.size(2), images.size(3), images.size(4))

    # output [batch_size * num_images, feature_dim]
    with torch.no_grad():
        outputs = model(reshaped_images)

    # 假设模型返回的结果中包含 `last_hidden_state`，这个张量的形状为 [batch_size * num_images, seq_len, feature_dim]
    image_features = outputs.last_hidden_state
    # 对每个图像的特征进行池化
    # 假设我们要在序列维度 (dim=1) 上进行池化（可以选择平均池化）
    image_features = image_features.mean(dim=1)  # [batch_size * num_images, feature_dim]
    # 将所有图像的特征堆叠成一个张量，形状为 [batch_size, num_images, feature_dim]
    image_features = image_features.view(images.size(0), images.size(1), -1)  # 重新调整维度为 [batch_size, num_images, feature_dim]
    # 对 num_images 维度进行聚合，得到每个样本的最终特征
    aggregated_features = image_features.mean(dim=1)  # [batch_size, feature_dim]

    return aggregated_features