import os

# 指定包含episode文件的文件夹路径
folder_path = "/media/jyxc/T5 EVO/yun_clothes-post/"

# 获取文件夹中所有文件的名称
files = os.listdir(folder_path)

# 对文件进行排序
files.sort()

# 重命名文件
i = 0
for _, file_name in enumerate(files):
    # 获取文件的扩展名
    # _, ext = os.path.splitext(file_name)
    
    # 构建新的文件名
    new_file_name = f"episode_{i}.hdf5"
    
    # 重命名文件
    os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
    i += 1

print("Files renamed successfully.")