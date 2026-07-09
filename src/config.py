import os
import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        
        # Model default type
        model = config_dict.get("model", {})
        self.default_model_type: str = model.get("default_type", "patchtst")
        
        # PatchTST config
        ptst = model.get("patchtst", {})
        self.ptst_seq_len: int = ptst.get("seq_len", 96)
        self.ptst_pred_len: int = ptst.get("pred_len", 24)
        self.ptst_patch_len: int = ptst.get("patch_len", 16)
        self.ptst_stride: int = ptst.get("stride", 8)
        self.ptst_d_model: int = ptst.get("d_model", 128)
        self.ptst_n_heads: int = ptst.get("n_heads", 4)
        self.ptst_d_ff: int = ptst.get("d_ff", 256)
        self.ptst_n_layers: int = ptst.get("n_layers", 3)
        
        # N-BEATS config
        nb = model.get("nbeats", {})
        self.nb_seq_len: int = nb.get("seq_len", 96)
        self.nb_pred_len: int = nb.get("pred_len", 24)
        self.nb_num_stacks: int = nb.get("num_stacks", 4)
        self.nb_num_blocks: int = nb.get("num_blocks", 3)
        self.nb_width: int = nb.get("width", 128)
        
        # TFT config
        tft = model.get("tft", {})
        self.tft_seq_len: int = tft.get("seq_len", 96)
        self.tft_pred_len: int = tft.get("pred_len", 24)
        self.tft_hidden_size: int = tft.get("hidden_size", 64)
        self.tft_num_heads: int = tft.get("num_heads", 4)
        
        # Training config
        training = config_dict.get("training", {})
        self.epochs: int = training.get("epochs", 5)
        self.batch_size: int = training.get("batch_size", 16)
        self.learning_rate: float = training.get("learning_rate", 0.001)
        self.weight_decay: float = training.get("weight_decay", 0.0001)
        self.mixed_precision: bool = training.get("mixed_precision", True)
        self.checkpoint_dir: str = training.get("checkpoint_dir", "checkpoints")
        self.wandb_logging: bool = training.get("wandb_logging", False)
        self.wandb_project: str = training.get("wandb_project", "forecastiq")
        
        # API config
        api = config_dict.get("api", {})
        self.api_host: str = api.get("host", "0.0.0.0")
        self.api_port: int = api.get("port", 8002)

def load_config(config_path: str = "") -> Config:
    if not config_path:
        current_dir = Path(__file__).resolve().parent
        config_path = str(current_dir.parent / "configs" / "config.yaml")
        
    if not os.path.exists(config_path):
        return Config({})
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    return Config(config_dict)
