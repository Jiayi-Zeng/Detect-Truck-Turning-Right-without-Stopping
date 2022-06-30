# 抽数据集
import os
import random
import shutil

ns = 7481

file_list = []
for filename in os.listdir(r'C:\Users\Zeng\Desktop\新建文件夹\training-image_2a_8050398126542848\training-image_2a'):
    (name, extension) = os.path.splitext(filename)
    file_list.append(name)
samples = random.sample(file_list, ns)

i = 1
src = r'C:\Users\Zeng\Desktop\新建文件夹\training-image_2a_8050398126542848\training-image_2a'  # 原文件夹路径
des = r'C:\Users\Zeng\Desktop\新建文件夹\DAIR-V2X\image_2'  # 目标文件夹路径
for file in samples:
    # 遍历原文件夹中的文件
    file += ".jpg"
    full_file_name = os.path.join(src, file)  # 把文件的完整路径得到
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, des)
        print(i, "要被复制的全文件路径全名:", full_file_name)
        i = i + 1

i = 1
src = r'C:\Users\Zeng\Desktop\新建文件夹\label_2'  # 原文件夹路径
des = r'C:\Users\Zeng\Desktop\新建文件夹\DAIR-V2X\label_2'  # 目标文件夹路径
for file in samples:
    # 遍历原文件夹中的文件
    file += ".txt"
    full_file_name = os.path.join(src, file)  # 把文件的完整路径得到
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, des)
        print(i, "要被复制的全文件路径全名:", full_file_name)
        i = i + 1


i = 1
src = r'C:\Users\Zeng\Desktop\新建文件夹\calib'  # 原文件夹路径
des = r'C:\Users\Zeng\Desktop\新建文件夹\DAIR-V2X\calib'  # 目标文件夹路径
for file in samples:
    # 遍历原文件夹中的文件
    file += ".txt"
    full_file_name = os.path.join(src, file)  # 把文件的完整路径得到
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, des)
        print(i, "要被复制的全文件路径全名:", full_file_name)
        i = i + 1

