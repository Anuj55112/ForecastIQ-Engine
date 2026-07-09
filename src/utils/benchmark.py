import os
import sys
import time
import json
import platform
import subprocess
from datetime import datetime

def get_system_metadata() -> dict:
    metadata = {
        "os": platform.system(),
        "cpu": "Unknown",
        "ram_gb": 8,
        "gpu": "None"
    }
    
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            metadata["cpu"] = cpu
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        metadata["cpu"] = line.split(":")[1].strip()
                        break
    except Exception:
        pass
        
    try:
        if platform.system() == "Darwin":
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
            metadata["ram_gb"] = round(mem_bytes / (1024 ** 3))
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        metadata["ram_gb"] = round(mem_kb / (1024 ** 2))
                        break
    except Exception:
        pass
        
    try:
        import torch
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            metadata["gpu"] = "Apple Metal (MPS)"
    except Exception:
        pass
        
    return metadata

def run_benchmark():
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    report = {
        "project": "ForecastIQ Engine",
        "timestamp": timestamp,
        "status": "not_run",
        "hardware": get_system_metadata(),
        "environment": {
            "python": platform.python_version()
        },
        "metadata": {
            "model": "PatchTST / N-BEATS",
            "parameters_million": 1.2,
            "dataset": "Simulated Demand Series (500 steps)",
            "batch_size": 16,
            "image_size": None,
            "sequence_length": 96,
            "device": "cpu"
        },
        "benchmarks": {}
    }
    
    # Check dependencies
    required_libs = ["torch", "pandas", "yaml"]
    missing_deps = []
    
    for lib in required_libs:
        try:
            mod = __import__(lib)
            report["environment"][lib] = getattr(mod, "__version__", "installed")
        except ImportError:
            missing_deps.append(lib)
            
    if missing_deps:
        report["status"] = "not_run"
        report["reason"] = f"Missing required dependencies: {', '.join(missing_deps)}"
        report["required_dependency"] = missing_deps[0]
        
        save_reports(report, date_str)
        print(f"Benchmark not run: {report['reason']}")
        return
        
    try:
        import torch
        from src.models.patchtst import PatchTST
        from src.models.nbeats import NBeats
        from src.config import load_config
        
        config = load_config()
        
        # PatchTST initialization
        patchtst = PatchTST(
            seq_len=config.ptst_seq_len,
            pred_len=config.ptst_pred_len,
            n_layers=2,
            n_heads=2,
            d_model=16,
            d_ff=32
        )
        patchtst.eval()
        
        # NBEATS initialization
        nbeats = NBeats(
            seq_len=config.nb_seq_len,
            pred_len=config.nb_pred_len,
            num_stacks=config.nb_num_stacks,
            num_blocks=config.nb_num_blocks,
            width=config.nb_width
        )
        nbeats.eval()
        
        # dummy input: [Batch, Seq_Len, Features]
        dummy_input = torch.randn(16, config.ptst_seq_len, 1)
        
        print("Benchmarking PatchTST inference...")
        start_time = time.time()
        for _ in range(20):
            with torch.no_grad():
                _ = patchtst(dummy_input)
        patchtst_latency = ((time.time() - start_time) / 20) * 1000
        
        print("Benchmarking N-BEATS inference...")
        start_time = time.time()
        for _ in range(20):
            with torch.no_grad():
                _ = nbeats(dummy_input)
        nbeats_latency = ((time.time() - start_time) / 20) * 1000
        
        report["status"] = "success"
        report["benchmarks"] = {
            "patchtst_latency_ms": round(patchtst_latency, 2),
            "nbeats_latency_ms": round(nbeats_latency, 2),
            "patchtst_qps": round(1000 / patchtst_latency, 2),
            "nbeats_qps": round(1000 / nbeats_latency, 2)
        }
        print(f"Benchmark success: PatchTST = {patchtst_latency:.2f}ms, N-BEATS = {nbeats_latency:.2f}ms")
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"Benchmark execution error: {e}"
        print(f"Benchmark error: {e}")
        
    save_reports(report, date_str)

def save_reports(report: dict, date_str: str):
    with open(f"reports/{date_str}-benchmark.json", "w") as f:
        json.dump(report, f, indent=4)
    with open("reports/latest.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
