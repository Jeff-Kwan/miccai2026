import torch 
from datetime import datetime
import os
import json


def main(paradigm: str, workers: int):
    # Directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date = datetime.now().strftime("%Y_%m_%d")
    timestamp = datetime.now().strftime("%H_%M")
    output_dir = f"results/{date}/{timestamp}_V{paradigm.upper()}"
    os.makedirs(output_dir, exist_ok=True)

    # Model
    config = json.load(open(f"config/V{paradigm.upper()}.json", "r"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    from models.VideoViT import VideoViTEncoder, VideoViTDecoder, VideoViTCfg, VideoViTDecCfg
    enc = VideoViTEncoder(VideoViTCfg(**config["encoder"]))
    dec = VideoViTDecoder(enc_dim=config["encoder"]["dim"], patch=config["encoder"]["patch"], 
                        in_chans=config["encoder"]["in_chans"], cfg=VideoViTDecCfg(**config["decoder"]))
    
    if paradigm.lower() == 'jepa':
        from models.VJEPA import VideoJEPA
        model = VideoJEPA(enc, dec, momentum=config["jepa"]["momentum"], mask_ratio=config["jepa"]["mask_ratio"])

    elif paradigm.lower() == 'mae':
        from models.VMAE import VideoMAE
        model = VideoMAE(enc, dec, norm_pix_loss=config["mae"]["norm_pix_loss"], mask_ratio=config["mae"]["mask_ratio"])

    else:
        raise ValueError(f"Unsupported paradigm: {paradigm}")
    model = model.to(device)
    print(f"Initialized V{paradigm.upper()} with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    # Optimizer & Scheduler
    torch.set_float32_matmul_precision(config["training"]["matmul_precision"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # Data
    from torch.utils.data import DataLoader
    from datahandling.EchoDynaDatasetShard import load_echonet_dynamic_datasets
    from datahandling.augmentations.get_augmentations import get_pretrain_augmentations
    from datahandling.collate import pretrain_collate
    aug = get_pretrain_augmentations()
    train_ds, _, _ = load_echonet_dynamic_datasets(get_mask=False)
    train_dl = DataLoader(train_ds, 
        batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=True, persistent_workers=True,
        collate_fn=lambda x: pretrain_collate(x, max_frames=config["training"]["max_frames"], augmentations=aug))

    # Trainer
    from utils.pretrainer import PreTrainer, TrainerConfig
    cfg = TrainerConfig(
        output_dir=output_dir,
        epochs=config["training"]["epochs"],
        autocast=config["training"]["autocast"],
        amp_dtype=torch.bfloat16,
        torch_compile=config["training"]["compile"],
        grad_clip_max_norm=1.0,
    )

    trainer = PreTrainer(
        paradigm=paradigm,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_dl=train_dl,
        augmented_input=True, # does ["aug_videos"] exist
        config=cfg,
    )

    history = trainer.train()



if __name__ == "__main__":
    paradigm = 'mae'  # or mae | jepa
    workers = 24
    main(paradigm=paradigm, workers=workers)