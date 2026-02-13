import torch 
from torch.utils.data import DataLoader
from datetime import datetime
import os
import json

from models.VideoViT2 import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg
from models.ViTMAEMotion2 import VideoMotionMAE, SimpleConvDecoder
from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
from datahandling.augmentations.get_augmentations import get_pretrain_augmentations
from datahandling.collate import pretrain_collate
from utils.trainer import MAETrainer, TrainerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
date = datetime.now().strftime("%Y_%m_%d")
timestamp = datetime.now().strftime("%H_%M")
output_dir = f"results/{date}/{timestamp}_VMAE"
os.makedirs(output_dir, exist_ok=True)

# Training Parameters
epochs = 600
batch_size = 64
learning_rate = 2e-4
weight_decay = 1e-2
max_frames = 64

torch.set_float32_matmul_precision('medium')
autocast = True
torch_compile = True
workers = 32

# Model
config = json.load(open("config/VMAE.json", "r"))
enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
dec = VideoViTDecoder(enc_dim=config["encoder"]["dim"], patch=config["encoder"]["patch"], 
                      in_chans=config["encoder"]["in_chans"], cfg=VideoViTDecCfg(**config["decoder"]))
frame_dec = SimpleConvDecoder(latent=config["encoder"]["dim"], out_dim=config["encoder"]["in_chans"], base=config["decoder"]["dec_dim"])
mae = VideoMotionMAE(enc, dec, frame_dec, motion_dim=2, norm_pix_loss=False, mask_ratio=0.75)
mae = mae.to(device)
print(f"Initialized VMAE with {sum(p.numel() for p in mae.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")
optimizer = torch.optim.AdamW(mae.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Dataset
aug = get_pretrain_augmentations()
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(get_mask=False)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, 
    pin_memory=True, collate_fn=lambda x: pretrain_collate(x, max_frames=max_frames, augmentations=aug))
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers//2, 
    pin_memory=True, collate_fn=lambda x: pretrain_collate(x, max_frames=max_frames, augmentations=None))

# Trainer
cfg = TrainerConfig(
    output_dir=output_dir,
    epochs=epochs,
    autocast=autocast,
    amp_dtype=torch.bfloat16,
    torch_compile=torch_compile,
    grad_clip_max_norm=1.0,
)

trainer = MAETrainer(
    model=mae,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    train_dl=train_dl,
    val_dl=val_dl,
    val_ds=val_ds,
    augmentations=True,
    config=cfg,
)

history = trainer.train(run_val=False)
