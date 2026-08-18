"""
同步
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
git status
git add .
git commit -m "Update RGB-T training and validation pipeline"
git push
指定文件
git add train_rgbt.py val_rgbt.py models/rgbt_model.py
git commit -m "Improve RGB-T training and validation"
git push
不同分支
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy
git status
git checkout -b RSDT&Alignment
git add .
git commit -m "backup: save current RSD-T version"
git push -u origin backup/rsdt-current-20260818
"""