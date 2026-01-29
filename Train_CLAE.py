import torch 
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datahandling.EchoDynaDataset import load_echonet_dynamic_datasets
from models.CircularLatentAE import CircularLatentAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training Parameters
epochs = 100
batch_size = 8
learning_rate = 3e-4
weight_decay = 1e-4
deformLAMBDA = 1e-3


# Dataset
train_ds, val_ds, test_ds = load_echonet_dynamic_datasets(
    "data/echodyna/FileList.csv",
    "data/echodyna/Videos",
    "data/echodyna/VolumeTracings.csv",
    load_video=True)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)


# Model
model = CircularLatentAE(in_c=3, latent=512, layers=4, levels=6)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training 
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    p_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in p_bar:
        videos = batch['video'].to(device)  # [B, C, T, H, W]
        
        optimizer.zero_grad()
        recon_videos = model(videos)
        
        mse_loss = criterion(recon_videos, videos)
        loss = mse_loss + deformLAMBDA * model.deformationL2
        
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += mse_loss.item() * videos.size(0)
        p_bar.set_postfix({'MSE Loss': mse_loss.item(), 'Deform L2': model.deformationL2.item(), 'Grad Norm': norm})
    
    train_loss /= len(train_dl.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dl:
            videos = batch['video'].to(device)
            recon_videos = model(videos)
            
            mse_loss = criterion(recon_videos, videos)
            val_loss += mse_loss.item() * videos.size(0)
    
    val_loss /= len(val_dl.dataset)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")