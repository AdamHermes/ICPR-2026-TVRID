import torch
import torch.nn as nn
import torchvision.models as models
import lightning as L


class TripletLoss(nn.Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_ap = (anchor - positive).pow(2).sum(dim=1)
        d_an = (anchor - negative).pow(2).sum(dim=1)
        return torch.relu(d_ap - d_an + self.margin).mean()

# Add to utils/models.py

class HardTripletLoss(nn.Module):
    """Mining hard triplets for better training"""
    def __init__(self, margin: float, mining: str = "batch_hard"):
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute pairwise distances
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        loss = 0
        for i, label in enumerate(labels):
            # Positive mask: same identity, different sample
            pos_mask = (labels == label) & (torch.arange(len(labels), device=labels.device) != i)
            # Negative mask: different identity
            neg_mask = labels != label
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            # Hard positive: furthest positive
            hardest_pos_dist = dist_mat[i][pos_mask].max()
            # Hard negative: closest negative
            hardest_neg_dist = dist_mat[i][neg_mask].min()
            
            loss += torch.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        
        return loss / len(labels)


class CombinedLoss(nn.Module):
    """Combine multiple losses for better training"""
    def __init__(self, margin: float = 0.3, use_center: bool = True):
        super().__init__()
        self.triplet = HardTripletLoss(margin)
        self.use_center = use_center
        if use_center:
            self.center_loss = nn.Module()  # Implement center loss
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        triplet_loss = self.triplet(embeddings, labels)
        # Add center loss, contrastive loss, etc.
        return triplet_loss
    
def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(1)  # B,1,C,H,W
    if x.ndim == 5:
        return x
    raise ValueError(f"Unsupported input shape {x.shape}")


class DepthEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 3 * 3, out_features=embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)  # B,S,1,H,W
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.encoder(x)
        x = x.view(B, S, -1).mean(dim=1)
        return x


class RGBEncoder(nn.Module):
    def __init__(self, embedding_size: int, layers_not_frozen: int = 4):
        super().__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        last_hidden = resnet50.fc.in_features
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        self.embedding_layer = nn.Sequential(
            nn.Linear(last_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size),
            nn.ReLU(),
        )
        self._freeze_feature_extractor(layers_not_frozen)

    def _freeze_feature_extractor(self, layers_not_frozen: int):
        layers = list(self.feature_extractor.children())
        to_freeze = layers[:-layers_not_frozen] if layers_not_frozen > 0 else layers
        for layer in to_freeze:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)  # B,S,C,H,W
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        emb = self.embedding_layer(feats)
        emb = emb.view(B, S, -1).mean(dim=1)
        return emb


# Add to utils/models.py

class ImprovedRGBEncoder(nn.Module):
    """Better RGB encoder with attention and deeper embeddings"""
    def __init__(self, embedding_size: int = 128, backbone: str = "resnet50", 
                 layers_not_frozen: int = 4, use_attention: bool = True):
        super().__init__()
        
        # Try different backbones
        if backbone == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feat_dim = 2048
        elif backbone == "resnet101":
            base_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            feat_dim = 2048
        elif backbone == "efficientnet_b3":
            base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            feat_dim = 1536
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 16),
                nn.ReLU(),
                nn.Linear(feat_dim // 16, feat_dim),
                nn.Sigmoid()
            )
        
        # Improved embedding layers with batch norm and dropout
        self.embedding_layer = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        self._freeze_layers(layers_not_frozen)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        
        # Extract features
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        
        # Apply attention
        if self.use_attention:
            attn = self.attention(feats)
            feats = feats * attn
        
        # Get embeddings
        emb = self.embedding_layer(feats)
        
        # L2 normalize embeddings (important for re-id!)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        
        # Aggregate sequence
        emb = emb.view(B, S, -1).mean(dim=1)
        return emb


class ImprovedDepthEncoder(nn.Module):
    """Better depth encoder with residual connections"""
    def __init__(self, embedding_size: int = 128):
        super().__init__()
        
        # Use a deeper network with residual connections
        self.conv1 = self._make_layer(1, 64)
        self.conv2 = self._make_layer(64, 128)
        self.conv3 = self._make_layer(128, 256)
        self.conv4 = self._make_layer(256, 512)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
    
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        emb = self.embedding_layer(x)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        
        emb = emb.view(B, S, -1).mean(dim=1)
        return emb

class ReIDLightning(L.LightningModule):
    def __init__(self, embedding_size: int = 256, lr: float = 3e-4, 
                 margin: float = 0.3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Use improved encoders
        self.depth_encoder = ImprovedDepthEncoder(embedding_size)
        self.rgb_encoder = ImprovedRGBEncoder(
            embedding_size, 
            backbone=kwargs.get('backbone', 'resnet50'),
            use_attention=kwargs.get('use_attention', True)
        )
        
        # Use hard mining loss
        if kwargs.get('use_hard_mining', False):
            self.loss_fn = HardTripletLoss(margin)
        else:
            self.loss_fn = TripletLoss(margin)

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "rgb":
            return self.rgb_encoder(x)
        if modality == "depth":
            return self.depth_encoder(x)
        raise ValueError(f"Unknown modality {modality}")

    def training_step(self, batch, batch_idx):
        anchor = batch["anchor"]
        positive = batch["positive"]
        negative = batch["negative"]

        anchor_x = anchor[self.hparams.anchor_modality]
        positive_x = positive[self.hparams.positive_modality]
        negative_x = negative[self.hparams.negative_modality]

        anchor_out = self.encode(anchor_x, self.hparams.anchor_modality)
        positive_out = self.encode(positive_x, self.hparams.positive_modality)
        negative_out = self.encode(negative_x, self.hparams.negative_modality)

        loss = self.loss_fn(anchor_out, positive_out, negative_out)

        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)
        correct = (d_ap < d_an).float().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", correct, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if not {"anchor", "positive", "negative"} <= set(batch.keys()):
            return

        anchor_x = batch["anchor"][self.hparams.anchor_modality]
        positive_x = batch["positive"][self.hparams.positive_modality]
        negative_x = batch["negative"][self.hparams.negative_modality]

        anchor_out = self.encode(anchor_x, self.hparams.anchor_modality)
        positive_out = self.encode(positive_x, self.hparams.positive_modality)
        negative_out = self.encode(negative_x, self.hparams.negative_modality)

        loss = self.loss_fn(anchor_out, positive_out, negative_out)
        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)
        correct = (d_ap < d_an).float().mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/accuracy", correct, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Use AdamW with weight decay
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=1e-4
        )
        
        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
