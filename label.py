# 将无标签数据打标为doubt，固定标签‘5’
import os  # 导入os模块用于文件和目录操作
import random  # 导入random模块用于随机打乱数据
import shutil  # 导入shutil模块用于文件操作

main_path = "dataset/"  # 定义主数据集路径
data_label_path = "dataset/"  # 定义数据标签路径
data_label = ['tri', 'T', 'sin', 'for', 'dou']  # 定义标签列表

def get_all_files(dir, label=None):
    files_list = []  # 初始化文件列表
    for file in os.listdir(dir):  # 遍历目录中的所有文件
        if label:  # 如果有标签
            files_list.append(dir + file + ' ' + str(data_label.index(label)))  # 将文件路径和标签索引添加到列表中
        else:  # 如果没有标签
            files_list.append(dir + file)  # 仅将文件路径添加到列表中
    return files_list  # 返回文件列表

data_label_files = []  # 初始化带标签的文件列表

for label in data_label:  # 遍历所有标签
    single_label_path = data_label_path + label + '/'  # 构建单个标签的路径
    single_label_files = get_all_files(single_label_path, label)  # 获取该标签下的所有文件
    data_label_files += single_label_files  # 将该标签的文件添加到总文件列表中

random.shuffle(data_label_files)  # 随机打乱文件列表

with open(main_path + 'label.txt', 'w') as f:  # 打开标签文件以写入
    for line in data_label_files:  # 遍历文件列表中的每一行
        f.write(line + '\n')  # 将文件路径和标签写入文件
    f.close()  # 关闭文件