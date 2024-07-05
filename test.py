import os  # 导入os模块，用于文件和目录操作
import random  # 导入random模块，用于随机操作
import shutil  # 导入shutil模块，用于高级文件操作

create_txt = "test.txt"  # 定义未标注文件的路径
maned_txt = "data_sort/test/label.txt"  # 定义已标注文件的路径

create_dic = {}  # 初始化未标注文件的字典
with open(create_txt, 'r') as f:  # 打开未标注文件
    lines = f.readlines()  # 读取所有行
    for line in lines:  # 遍历每一行
        line = line.split('/')[-1]  # 获取文件名
        k_v = line.split(' ')  # 分割文件名和标签
        create_dic[k_v[0]] = k_v[1]  # 将文件名和标签存入字典
    f.close()  # 关闭文件

maned_dic = {}  # 初始化已标注文件的字典
with open(maned_txt, 'r') as f:  # 打开已标注文件
    lines = f.readlines()  # 读取所有行
    for line in lines:  # 遍历每一行
        line = line.split('/')[-1]  # 获取文件名
        k_v = line.split(' ')  # 分割文件名和标签
        maned_dic[k_v[0]] = k_v[1]  # 将文件名和标签存入字典
    f.close()  # 关闭文件

with open('correct.txt', 'w') as f:  # 打开文件以写入纠正后的标签
    for k, v in create_dic.items():  # 遍历未标注字典中的每一项
        if k not in maned_dic or maned_dic[k] != v:  # 如果文件名不在已标注字典中或标签不匹配
            f.write(k + ' ' + v)  # 将文件名和标签写入文件
    f.close()  # 关闭文件