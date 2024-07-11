#!/bin/bash

### transfer 
# 1. dataset_stats.pkl
# 2. various ckpt

# 获取用户输入的文件夹名称和ckpt名称
echo "请输入目标文件夹名称:"
read folder_name
echo "请输入ckpt名称:"
read ckpt_name

source_file="/home/jyxc/GitHub/act-plus-plus/$folder_name/$ckpt_name"
target_folder="s3://lanzihan/ckpt/$folder_name/"

# oss upload
echo "开始上传文件..."
oss cp "$source_file" "$target_folder"
echo "文件上传完成"

