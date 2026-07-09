import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Package level imports
from src.config import load_config
from src.data.processor import PyTorchForecastDataset
from src.models.patchtst import PatchTST
from src.models.nbeats import NBeats
from src.models.tft import TemporalFusionTransformer
from src.utils.logging import setup_logger

logger = setup_logger("train", "train.log")

def train_forecaster(
    model: nn.Module,
    device: torch.device,
    csv_path: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    amp: bool = True,
    checkpoint_dir: str = "checkpoints",
    wandb_logging: bool = False,
    wandb_project: str = "forecastiq"
):
    # Initialize datasets
    train_dataset = PyTorchForecastDataset(
        csv_path=csv_path,
        seq_len=model.seq_len,
        pred_len=model.pred_len,
        is_train=True
    )
    val_dataset = PyTorchForecastDataset(
        csv_path=csv_path,
        seq_len=model.seq_len,
        pred_len=model.pred_len,
        is_train=False,
        processor=train_dataset.processor
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    experiment = None
    if wandb_logging:
        try:
            import wandb
            experiment = wandb.init(project=wandb_project, resume='allow')
            experiment.config.update({
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "amp": amp,
                "weight_decay": weight_decay
            })
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}. Running training without WandB logging.")
            wandb_logging = False

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training windows: {len(train_dataset)}
        Validation windows: {len(val_dataset)}
        Mixed Precision: {amp}
        Device:          {device.type}
    ''')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}") as pbar:
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                with torch.autocast(device.type if device.type != 'mps' and device.type != 'cpu' else 'cpu', enabled=amp):
                    pred_y = model(batch_x)
                    loss = criterion(pred_y, batch_y)
                    
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                
                if wandb_logging and experiment:
                    experiment.log({"train_loss": loss.item()})
                    
        # Run validation
        model.eval()
        val_mse = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred_y = model(batch_x)
                
                val_mse += criterion(pred_y, batch_y).item()
                val_mae += mae_metric(pred_y, batch_y).item()
                
        avg_val_mse = val_mse / max(len(val_loader), 1)
        avg_val_mae = val_mae / max(len(val_loader), 1)
        
        logger.info(f"Epoch {epoch}: Val MSE: {avg_val_mse:.5f} | Val MAE: {avg_val_mae:.5f}")
        
        if wandb_logging and experiment:
            experiment.log({
                "val_mse": avg_val_mse,
                "val_mae": avg_val_mae,
                "epoch": epoch
            })
            
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mse": avg_val_mse
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at {checkpoint_path}")
            
    if wandb_logging and experiment:
        experiment.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ForecastIQ Time Series Models")
    parser.add_argument("--model", type=str, default="patchtst", choices=["patchtst", "nbeats", "tft"], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    
    args = parser.parse_args()
    
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    epochs = args.epochs if args.epochs is not None else config.epochs
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    lr = args.lr if args.lr is not None else config.learning_rate
    
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample", "time_series.csv"))
    
    # Model instantiation
    if args.model == "patchtst":
        model = PatchTST(
            seq_len=config.ptst_seq_len,
            pred_len=config.ptst_pred_len,
            patch_len=config.ptst_patch_len,
            stride=config.ptst_stride,
            d_model=config.ptst_d_model,
            n_heads=config.ptst_n_heads,
            d_ff=config.ptst_d_ff,
            n_layers=config.ptst_n_layers
        )
    elif args.model == "nbeats":
        model = NBeats(
            seq_len=config.nb_seq_len,
            pred_len=config.nb_pred_len,
            num_stacks=config.nb_num_stacks,
            num_blocks=config.nb_num_blocks,
            width=config.nb_width
        )
    else:
        model = TemporalFusionTransformer(
            seq_len=config.tft_seq_len,
            pred_len=config.tft_pred_len,
            hidden_size=config.tft_hidden_size,
            num_heads=config.tft_num_heads
        )
        
    model.to(device)
    
    train_forecaster(
        model=model,
        device=device,
        csv_path=csv_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        amp=config.mixed_precision,
        checkpoint_dir=config.checkpoint_dir,
        wandb_logging=config.wandb_logging,
        wandb_project=config.wandb_project
    )
