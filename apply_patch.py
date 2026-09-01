#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Install the RLSFA+BHLR final version into:
    /mnt/sda/taochangyong/Projects/Model/YOLO-tcy

Legacy FBAM files are NOT deleted or overwritten.
Existing destination files are backed up with suffix .before_rlsfa.
"""

from pathlib import Path
import shutil

PROJECT = Path("/mnt/sda/taochangyong/Projects/Model/YOLO-tcy").resolve()
PATCH = Path(__file__).resolve().parent

FILES = [
    "models/modules/alignment/rlsfa.py",
    "models/modules/alignment/builder.py",
    "models/rgbt_rlsfa_bhlr_model.py",
    "train_rlsfa_bhlr.py",
    "val_rlsfa_bhlr.py",
    "visualize_rlsfa_bhlr_internal.py",
    "diagnose_rlsfa_alignment.py",
    "smoke_test_rlsfa_module.py",
    "configs/experiments/rlsfa/uavcb_rlsfa_bhlr_yolo26n_1280_640.yaml",
    "configs/experiments/rlsfa/uavcb_rlsfa_bhlr_yolo26n_1280_640_smoke.yaml",
]

if not PROJECT.exists():
    raise FileNotFoundError(f"Project not found: {PROJECT}")

for rel in FILES:
    src = PATCH / rel
    dst = PROJECT / rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        backup = dst.with_name(dst.name + ".before_rlsfa")
        shutil.copy2(dst, backup)
        print("BACKUP:", backup)
    shutil.copy2(src, dst)
    print("COPY  :", dst)

print("\nRLSFA patch installed.")
print("Legacy FBAM files remain unchanged.")
print("\nRecommended syntax/module check:")
print("  python smoke_test_rlsfa_module.py --device cpu")
print("\nTraining config:")
print("  configs/experiments/rlsfa/uavcb_rlsfa_bhlr_yolo26n_1280_640.yaml")
