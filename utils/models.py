import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lightning as L


class BatchHardTripletLoss(nn.Module):
    """Online hard triplet mining - much better than random sampling"""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Stack all embeddings
        embeddings = torch.cat([anchor, positive, negative], dim=0)
        batch_size = anchor.size(0)
        
        # Create labels (0, 0, 1, 1, 2, 2, ...) for anchor-positive pairs
        labels = torch.arange(batch_size, device=anchor.device).repeat(2)
        
        # Compute pairwise distance matrix
        dist_mat = torch.cdist(embeddings[:batch_size*2], embeddings, p=2)
        
        loss = 0
        count = 0
        for i in range(batch_size):
            anchor_idx = i
            # Hard positive: furthest positive (same identity)
            pos_mask = labels[:batch_size*2] == labels[anchor_idx]
            pos_mask[anchor_idx] = False  # Exclude self
            
            if pos_mask.any():
                hardest_pos_dist = dist_mat[anchor_idx][pos_mask].max()
                
                # Hard negative: closest negative (different identity)
                neg_mask = labels != labels[anchor_idx]
                if neg_mask.any():
                    hardest_neg_dist = dist_mat[anchor_idx][neg_mask].min()
                    loss += torch.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
                    count += 1
        
        return loss / max(count, 1)


def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(1)  # B,1,C,H,W
    if x.ndim == 5:
        return x
    raise ValueError(f"Unsupported input shape {x.shape}")


class ImprovedDepthEncoder(nn.Module):
    """Much deeper depth encoder with residual-style blocks and attention"""
    def __init__(self, embedding_size: int = 256):
        super().__init__()
        
        # Deeper convolutional blocks
        self.conv1 = self._make_block(1, 64)
        self.conv2 = self._make_block(64, 128)
        self.conv3 = self._make_block(128, 256)
        self.conv4 = self._make_block(256, 512)
        
        # Channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512 // 16, 1),
            nn.ReLU(),
            nn.Conv2d(512 // 16, 512, 1),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Deeper embedding layer with batch norm and dropout
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
    
    def _make_block(self, in_ch, out_ch):
        """ResNet-style conv block"""
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
        x = _ensure_sequence(x)  # B,S,1,H,W
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Channel attention
        att = self.attention(x)
        x = x * att
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Embedding
        emb = self.embedding_layer(x)
        
        # L2 normalize (CRITICAL for re-ID)
        emb = F.normalize(emb, p=2, dim=1)
        
        # Aggregate sequence
        emb = emb.view(B, S, -1).mean(dim=1)
        return emb


class ImprovedRGBEncoder(nn.Module):
    """Improved RGB encoder with attention and better embedding layers"""
    def __init__(self, embedding_size: int = 256, layers_not_frozen: int = 4):
        super().__init__()
        
        # Use pretrained ResNet50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        last_hidden = resnet50.fc.in_features  # 2048
        
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Channel attention
        self.attention = nn.Sequential(
            nn.Linear(last_hidden, last_hidden // 16),
            nn.ReLU(),
            nn.Linear(last_hidden // 16, last_hidden),
            nn.Sigmoid()
        )
        
        # Improved embedding layer with batch norm and dropout
        self.embedding_layer = nn.Sequential(
            nn.Linear(last_hidden, 2048),
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
        
        # Extract features
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        
        # Apply attention
        att = self.attention(feats)
        feats = feats * att
        
        # Get embeddings
        emb = self.embedding_layer(feats)
        
        # L2 normalize (CRITICAL for re-ID)
        emb = F.normalize(emb, p=2, dim=1)
        
        # Aggregate sequence
        emb = emb.view(B, S, -1).mean(dim=1)
        return emb


class ReIDLightning(L.LightningModule):
    def __init__(
        self,
        embedding_size: int = 256,
        lr: float = 3e-4,
        margin: float = 0.3,
        anchor_modality: str = "rgb",
        positive_modality: str = "rgb",
        negative_modality: str = "rgb",
        rgb_layers_not_frozen: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Use improved encoders
        self.depth_encoder = ImprovedDepthEncoder(embedding_size)
        self.rgb_encoder = ImprovedRGBEncoder(embedding_size, rgb_layers_not_frozen)
        
        # Use batch hard triplet loss instead of simple triplet loss
        self.loss_fn = BatchHardTripletLoss(margin)

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

        # Compute accuracy
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
            weight_decay=5e-4  # Important for generalization
        )
        
        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step, not every epoch
                "frequency": 1
            }
        }