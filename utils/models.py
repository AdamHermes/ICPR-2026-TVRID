import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lightning as L


def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(1)  # B,1,C,H,W
    if x.ndim == 5:
        return x
    raise ValueError(f"Unsupported input shape {x.shape}")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ImprovedTripletLoss(nn.Module):
    """Improved triplet loss with hard negative mining within batch"""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        batch_size = anchor.size(0)
        
        # Basic triplet distances
        pos_dist = (anchor - positive).pow(2).sum(dim=1)
        
        # Hard negative mining
        anchor_expanded = anchor.unsqueeze(1)
        negative_expanded = negative.unsqueeze(0)
        all_neg_dists = (anchor_expanded - negative_expanded).pow(2).sum(dim=2)
        hard_neg_dist, _ = all_neg_dists.min(dim=1)
        
        loss = torch.relu(pos_dist - hard_neg_dist + self.margin).mean()
        return loss


class ModifiedCentroidTripletLoss(nn.Module):
    """
    Modified Centroid Triplet Loss (MCTL) from paper:
    "Modified centroid triplet loss for person re-identification"
    
    L_mctl = ω₁ · d(f(A), c_p)² + ω₂ · [m - min d(f(A), c_N)]₊²
    """
    def __init__(self, margin: float = 0.3, w1: float = 0.1, w2: float = 0.9):
        super().__init__()
        self.margin = margin
        self.w1 = w1
        self.w2 = w2
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] - batch of embeddings
            labels: [B] - person IDs for each embedding
        """
        batch_size = embeddings.size(0)
        unique_labels = labels.unique()
        
        # Compute centroids for each class in the batch
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label.item()] = embeddings[mask].mean(dim=0)
        
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            anchor_emb = embeddings[i]
            anchor_label = labels[i].item()
            
            # Positive centroid (anchor's class)
            c_p = centroids[anchor_label]
            
            # Distance to positive centroid (intra-class term)
            dist_pos = (anchor_emb - c_p).pow(2).sum()
            
            # Distances to negative centroids (inter-class term)
            neg_dists = []
            for label, centroid in centroids.items():
                if label != anchor_label:
                    dist = (anchor_emb - centroid).pow(2).sum().sqrt()
                    neg_dists.append(dist)
            
            if len(neg_dists) > 0:
                min_neg_dist = torch.stack(neg_dists).min()
                
                # MCTL formula
                intra_term = self.w1 * dist_pos
                inter_term = self.w2 * torch.relu(self.margin - min_neg_dist).pow(2)
                
                loss += intra_term + inter_term
                count += 1
        
        return loss / max(count, 1)


class CenterLoss(nn.Module):
    """Center Loss for intra-class compactness"""
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # Initialize centers randomly
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] - raw features (NOT normalized)
            labels: [B] - person IDs
        """
        batch_size = embeddings.size(0)
        
        # Expand centers
        centers_batch = self.centers[labels]  # [B, D]
        
        # Compute L2 distance
        loss = (embeddings - centers_batch).pow(2).sum(dim=1).mean()
        
        return loss


# ============================================================================
# ENCODERS WITH PAPER ARCHITECTURE
# ============================================================================

class PaperRGBEncoder(nn.Module):
    """
    RGB Encoder following paper architecture:
    - ResNet50 backbone
    - Modified last stride (2 -> 1) for larger feature maps
    - Output: 2048-dim raw features (before BNNeck)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # CRITICAL: Modify last stride from 2 to 1 (as per paper)
        # This is in layer4's first block
        resnet50.layer4[0].conv2.stride = (1, 1)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        
        # Remove the final FC layer and avg pool (we'll use our own)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] or [B, S, 3, H, W]
        Returns:
            features: [B, 2048] - raw features before BNNeck
        """
        x = _ensure_sequence(x)  # Ensure [B, S, C, H, W]
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        
        # ResNet forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.gap(x)  # [B*S, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B*S, 2048]
        
        # Aggregate sequence (if S > 1)
        x = x.view(B, S, -1).mean(dim=1)  # [B, 2048]
        
        return x


class PaperDepthEncoder(nn.Module):
    """
    Depth encoder with architecture similar to RGB encoder
    Since paper doesn't specify depth encoder details, we use a similar structure
    """
    def __init__(self):
        super().__init__()
        
        # Custom CNN for depth (since ResNet is designed for RGB)
        self.conv1 = self._make_layer(1, 64)      # 1 -> 64
        self.conv2 = self._make_layer(64, 128)    # 64 -> 128
        self.conv3 = self._make_layer(128, 256)   # 128 -> 256
        self.conv4 = self._make_layer(256, 512)   # 256 -> 512
        self.conv5 = self._make_layer(512, 1024)  # 512 -> 1024
        self.conv6 = self._make_layer(1024, 2048) # 1024 -> 2048 (match RGB)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] or [B, S, 1, H, W]
        Returns:
            features: [B, 2048]
        """
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Aggregate sequence
        x = x.view(B, S, -1).mean(dim=1)
        
        return x


# ============================================================================
# MAIN MODEL WITH PAPER ARCHITECTURE
# ============================================================================

class ReIDLightning(L.LightningModule):
    """
    Person Re-ID model following paper architecture:
    - ResNet50 encoder with modified stride
    - BNNeck before classifier
    - Multiple losses: CrossEntropy + Triplet + Center + MCTL
    """
    def __init__(
        self,
        num_classes: int = 62,
        embedding_size: int = 2048,  # Paper uses 2048
        lr: float = 0.00035,
        margin: float = 0.3,
        anchor_modality: str = "rgb",
        positive_modality: str = "rgb",
        negative_modality: str = "rgb",
        use_mctl: bool = True,
        use_center_loss: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoders (output 2048-dim features)
        self.rgb_encoder = PaperRGBEncoder(pretrained=True)
        self.depth_encoder = PaperDepthEncoder()
        
        # BNNeck: Batch Normalization before classifier (critical in paper)
        self.bn_neck = nn.BatchNorm1d(2048)
        self.bn_neck.bias.requires_grad_(False)  # No bias in BN
        
        # Classifier head (for cross-entropy loss)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        
        # Loss functions
        self.triplet_loss = ImprovedTripletLoss(margin)
        
        if use_mctl:
            self.mctl_loss = ModifiedCentroidTripletLoss(margin=margin, w1=0.1, w2=0.9)
        
        if use_center_loss:
            self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=2048)
        
        # Initialize weights
        self._init_params()
    
    def _init_params(self):
        """Initialize classifier weights"""
        nn.init.normal_(self.classifier.weight, std=0.001)
    
    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Get raw 2048-dim features (before BNNeck)
        """
        if modality == "rgb":
            return self.rgb_encoder(x)
        elif modality == "depth":
            return self.depth_encoder(x)
        raise ValueError(f"Unknown modality: {modality}")
    
    def forward(self, x: torch.Tensor, modality: str) -> tuple:
        """
        Forward pass returning both raw features and BN features
        
        Returns:
            raw_features: [B, 2048] - for center loss
            bn_features: [B, 2048] - for triplet/MCTL loss and inference
            logits: [B, num_classes] - for classification loss
        """
        raw_features = self.encode(x, modality)
        bn_features = self.bn_neck(raw_features)
        logits = self.classifier(bn_features)
        
        return raw_features, bn_features, logits

    def training_step(self, batch, batch_idx):
        anchor = batch["anchor"]
        positive = batch["positive"]
        negative = batch["negative"]
        person_ids = batch["person_id"]

        anchor_x = anchor[self.hparams.anchor_modality]
        positive_x = positive[self.hparams.positive_modality]
        negative_x = negative[self.hparams.negative_modality]

        # Forward pass
        anchor_raw, anchor_bn, anchor_logits = self.forward(anchor_x, self.hparams.anchor_modality)
        positive_raw, positive_bn, _ = self.forward(positive_x, self.hparams.positive_modality)
        negative_raw, negative_bn, _ = self.forward(negative_x, self.hparams.negative_modality)

        # ===== LOSS 1: Classification Loss (CrossEntropy) =====
        loss_cls = F.cross_entropy(anchor_logits, person_ids)
        
        # ===== LOSS 2: Triplet Loss =====
        # Use BN features for metric learning (as per paper)
        loss_triplet = self.triplet_loss(anchor_bn, positive_bn, negative_bn)
        
        # ===== LOSS 3: Center Loss =====
        loss_center = torch.tensor(0.0, device=self.device)
        if self.hparams.use_center_loss:
            # Use RAW features (before BN) for center loss
            loss_center = self.center_loss(anchor_raw, person_ids)
        
        # ===== LOSS 4: MCTL Loss =====
        loss_mctl = torch.tensor(0.0, device=self.device)
        if self.hparams.use_mctl:
            # Use BN features
            all_embeddings = torch.cat([anchor_bn, positive_bn, negative_bn], dim=0)
            all_labels = torch.cat([person_ids, person_ids, person_ids], dim=0)
            loss_mctl = self.mctl_loss(all_embeddings, all_labels)
        
        # ===== TOTAL LOSS (weighted as per paper) =====
        total_loss = (
            1.0 * loss_cls +
            1.0 * loss_triplet +
            5e-4 * loss_center +  # Paper weights center loss by 5e-4
            1.0 * loss_mctl
        )

        # Compute accuracy
        d_ap = (anchor_bn - positive_bn).pow(2).sum(1)
        d_an = (anchor_bn - negative_bn).pow(2).sum(1)
        correct = (d_ap < d_an).float().mean()

        # Logging
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/loss_cls", loss_cls)
        self.log("train/loss_triplet", loss_triplet)
        self.log("train/loss_center", loss_center)
        self.log("train/loss_mctl", loss_mctl)
        self.log("train/accuracy", correct, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        if not {"anchor", "positive", "negative"} <= set(batch.keys()):
            return

        anchor_x = batch["anchor"][self.hparams.anchor_modality]
        positive_x = batch["positive"][self.hparams.positive_modality]
        negative_x = batch["negative"][self.hparams.negative_modality]

        _, anchor_bn, _ = self.forward(anchor_x, self.hparams.anchor_modality)
        _, positive_bn, _ = self.forward(positive_x, self.hparams.positive_modality)
        _, negative_bn, _ = self.forward(negative_x, self.hparams.negative_modality)

        loss = self.triplet_loss(anchor_bn, positive_bn, negative_bn)
        d_ap = (anchor_bn - positive_bn).pow(2).sum(1)
        d_an = (anchor_bn - negative_bn).pow(2).sum(1)
        correct = (d_ap < d_an).float().mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/accuracy", correct, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Split parameters: main network vs center loss
        main_params = [
            p for name, p in self.named_parameters()
            if 'center_loss' not in name and p.requires_grad
        ]
        
        # Adam optimizer for main network (as per paper)
        optimizer_main = torch.optim.Adam(
            main_params,
            lr=self.hparams.lr,
            weight_decay=5e-4
        )
        
        optimizers = [optimizer_main]
        
        # MultiStep scheduler (reduce at epochs 40, 70 as per paper)
        scheduler_main = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_main,
            milestones=[40, 70],
            gamma=0.1
        )
        
        schedulers = [{
            "scheduler": scheduler_main,
            "interval": "epoch",
            "frequency": 1
        }]
        
        # Separate SGD optimizer for center loss (as per paper)
        if self.hparams.use_center_loss:
            optimizer_center = torch.optim.SGD(
                self.center_loss.parameters(),
                lr=0.5  # Paper uses LR=0.5 for center loss
            )
            optimizers.append(optimizer_center)
        
        return optimizers, schedulers