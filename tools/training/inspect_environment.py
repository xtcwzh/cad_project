"""Read-only server inventory, no model weight load."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from training_core import config, offline, write_json, memory_snapshot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qlora_probe.json')
    args = ap.parse_args()
    c = config(args.config)
    offline(c['gpu'])
    import torch
    model = json.loads((Path(c['model_path']) / 'config.json').read_text())
    text = model.get('text_config', model)
    versions = {}
    for name in ('torch', 'transformers', 'unsloth', 'unsloth_zoo', 'peft', 'trl', 'bitsandbytes', 'triton', 'accelerate'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    p = Path(c['output_root'])
    while not p.exists(): p = p.parent
    result = {'platform': platform.platform(), 'python': platform.python_version(),
        'versions': versions, 'torch_cuda': torch.version.cuda, 'selected_gpu': c['gpu'],
        'output_free_gib': shutil.disk_usage(p).free / 2**30,
        'model_type': model.get('model_type'), 'architecture': model.get('architectures'),
        'quantization_config': model.get('quantization_config'),
        'context': text.get('max_position_embeddings'), 'cuda_available': torch.cuda.is_available()}
    if torch.cuda.is_available():
        result['selected_device_memory'] = memory_snapshot()
        result.update(gpu_name=torch.cuda.get_device_name(0),
                      capability=torch.cuda.get_device_capability(0),
                      memory_free_total_gib=[x / 2**30 for x in torch.cuda.mem_get_info(0)])
    try:
        result['nvidia_smi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired) as exc: result['nvidia_smi_error'] = str(exc)
    write_json(Path(c['output_root']) / 'environment.json', result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
