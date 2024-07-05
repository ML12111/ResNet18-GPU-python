# ResNet18---用ResNet18实现图像分类
# 1.代码运行环境：
- Python 3x
- PyTorch 2.3.0
- torchvision 0.10.0
- CUDA 12.1
- Python packages:torch,torchvision,tqdm,PIL,time,matplotlib

# 2.ResNet18运行方法
- 更改label.py中文件路径'main_path','data_label_path'，分两次分别对训练集和测试集进行打标，并存放在列表中，方便后续图像分类。
- 更改ResNet18.py中的训练集路径和测试集路径（改成你需要进行训练与分类的数据集路径），更改图像类别数--ResNet类中初始化函数'init'的形参'num_classes'。
- 开始训练ResNet18模型。代码将模型数据存放在'model_path'中，同时绘制训练过程中的训练loss、训练集准确率以及测试集准确率。
- 更改test.py中的'create_txt'路径和'maned_txt'路径，其中'maned_txt'是测试集标签列表文件。该文件的输出结果是ResNet18在对测试集进行分类时的错误样本。

# 3.applyResNet18的应用
- applyResNet18.py用于应用ResNet18网络对单张目标图像进行分类。
- 应用方法：'model_path'用于上传模型路径，'image_path'用于上传需要分类的目标图像路径。更改这两个路径以及class_names，即可用训练好的模型对目标图像进行分类，输出为目标图片属于各类别的概率。