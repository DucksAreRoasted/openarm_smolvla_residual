#!/usr/bin/env python3
"""Run the approved 10,000-step fine-tuning budget with persistent diagnostics."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "logs/正式微调_s10000.run.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    candidate = json.loads((ROOT / "configs/smolvla_rebot_train_s10000.json").read_text())
    cmd = candidate["argv"]
    assert "--steps=10000" in cmd and "--batch_size=8" in cmd
    features = json.loads(next(x.split("=", 1)[1] for x in cmd if x.startswith("--policy.input_features=")))
    assert features["observation.state"]["shape"] == [7]
    output = ROOT / next(x.split("=", 1)[1] for x in cmd if x.startswith("--output_dir="))
    if output.exists() or RECORD.exists():
        raise SystemExit("Output or run record already exists; refusing duplicate training.")
    stats = json.loads((ROOT / "data/lerobot/rebot_b601_pickcube_mp_yaw_n100/统计量_stats.json").read_text())
    assert stats["stats_id"] == candidate["stats_id"]
    (ROOT / "logs").mkdir(exist_ok=True)
    if args.detach:
        with (ROOT / "logs/正式微调_s10000.launch.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({"runner_pid": child.pid, "record": str(RECORD)}, ensure_ascii=False), flush=True)
        return
    start = time.time()
    record = {"status": "starting", "runner_pid": os.getpid(), "argv": cmd,
              "started_at": datetime.datetime.now().astimezone().isoformat(),
              "stats_id": stats["stats_id"], "dataset": "data/lerobot/rebot_b601_pickcube_mp_yaw_n100",
              "steps": 10000, "batch_size": 8, "seed": 1000,
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "lerobot", "numpy")},
              "converter_md5": hashlib.md5((ROOT / "scripts/转换LeRobot数据集_rebot.py").read_bytes()).hexdigest()}
    with RECORD.open("x") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    env = dict(os.environ, MPLCONFIGDIR="/tmp/rebot-mpl", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    monitor = None
    try:
        with (ROOT / "logs/正式微调_s10000.gpu.csv").open("w") as gpu, (ROOT / "logs/正式微调_s10000.log").open("w") as log:
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu",
                                        "--format=csv", "-l", "5"], stdout=gpu, stderr=subprocess.STDOUT)
            train = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            record.update(status="running", training_pid=train.pid, gpu_monitor_pid=monitor.pid)
            RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            code = train.wait()
        record.update(returncode=code, elapsed_s=time.time()-start, status="completed" if code == 0 else "failed")
        text = (ROOT / "logs/正式微调_s10000.log").read_text().replace("\r", "\n")
        losses = re.findall(r"\bloss:([\d.e+-]+)", text)
        if losses:
            record["final_loss"] = float(losses[-1])
        if code == 0:
            assert (output / "checkpoints/last/pretrained_model/model.safetensors").is_file()
            (output / "训练记录.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    except BaseException as exc:
        record.update(status="failed", error=repr(exc), elapsed_s=time.time()-start)
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait()
        RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    sys.exit(record["returncode"])


if __name__ == "__main__":
    main()
