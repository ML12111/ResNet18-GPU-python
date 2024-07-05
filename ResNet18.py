import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入常用函数模块
import torchvision.transforms as transforms  # 导入图像变换模块
from torch.utils.data import DataLoader  # 导入数据加载模块
from torch.cuda.amp import GradScaler, autocast  # 导入自动混合精度模块
from tqdm import tqdm  # 导入进度条模块
from PIL import Image  # 导入图像处理模块
import time  # 导入time模块
import matplotlib.pyplot as plt  # 导入matplotlib模块

# 定义数据集路径
train_label_path = "data_sort/train/label.txt"
test_label_path = "data_sort/test/label.txt"

# 读取标签和数据
labels = {}  # 创建标签字典
train_data = []  # 创建训练数据列表
test_data = []  # 创建测试数据列表

# 读取训练数据标签
with open(train_label_path, 'r') as f:
    lines = f.readlines()  # 读取所有行
    for line in lines:
        k_v = line.split(' ')  # 分割每行的文件名和标签
        train_data.append(k_v[0])  # 将文件名添加到训练数据列表
        labels[k_v[0]] = k_v[1]  # 将文件名和标签存入标签字典

# 定义训练数据加载类
class GetTrainLoader(torch.utils.data.Dataset):
    def __init__(self):
        self.data = train_data  # 获取训练数据列表
        self.label = labels  # 获取标签字典
        self.transform_train = transforms.Compose([
            transforms.Resize((135, 240)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
        ])

    def __getitem__(self, index):
        im_id = self.data[index]  # 获取图像ID
        image = Image.open(im_id)  # 打开图像
        image = image.crop([300, 80, 1620, 950])  # 裁剪图像
        image = self.transform_train(image)  # 进行图像变换
        label = int(self.label[im_id])  # 获取图像标签
        return image, label, im_id  # 返回图像、标签和图像ID

    def __len__(self):
        return len(self.data)  # 返回数据集大小

# 读取测试数据标签
with open(test_label_path, 'r') as f:
    lines = f.readlines()  # 读取所有行
    for line in lines:
        k_v = line.split(' ')  # 分割每行的文件名和标签
        test_data.append(k_v[0])  # 将文件名添加到测试数据列表
        labels[k_v[0]] = k_v[1]  # 将文件名和标签存入标签字典

# 定义测试数据加载类
class GetTestLoader(torch.utils.data.Dataset):
    def __init__(self):
        self.data = test_data  # 获取测试数据列表
        self.label = labels  # 获取标签字典
        self.transform_test = transforms.Compose([
            transforms.Resize((135, 240)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
        ])

    def __getitem__(self, index):
        im_id = self.data[index]  # 获取图像ID
        image = Image.open(im_id)  # 打开图像
        image = image.crop([300, 80, 1620, 950])  # 裁剪图像
        image = self.transform_test(image)  # 进行图像变换
        label = int(self.label[im_id])  # 获取图像标签
        return image, label, im_id  # 返回图像、标签和图像ID

    def __len__(self):
        return len(self.data)  # 返回数据集大小
    
# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),  # 卷积层
            nn.BatchNorm2d(outchannel),  # 批归一化层
            nn.ReLU(inplace=True),  # 激活层
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),  # 卷积层
            nn.BatchNorm2d(outchannel)  # 批归一化层
        )
        self.shortcut = nn.Sequential()  # 初始化捷径分支
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),  # 卷积层
                nn.BatchNorm2d(outchannel)  # 批归一化层
            )
            
    def forward(self, x):
        out = self.left(x)  # 前向传播通过主分支
        out = out + self.shortcut(x)  # 主分支与捷径分支相加
        out = F.relu(out)  # 激活
        return out  # 返回输出

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=5):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1, bias=False),  # 卷积层
            nn.BatchNorm2d(64),  # 批归一化层
            nn.ReLU()  # 激活层
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)  # 第一层
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)  # 第二层
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)  # 第三层
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)  # 第四层
        self.fc = nn.Linear(14336, num_classes)  # 全连接层
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 定义每层的步幅
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))  # 添加残差块
            self.inchannel = channels  # 更新输入通道
        return nn.Sequential(*layers)  # 返回层序列
    
    def forward(self, x):
        out = self.conv1(x)  # 通过第一个卷积层
        out = self.layer1(out)  # 通过第一层
        out = self.layer2(out)  # 通过第二层
        out = self.layer3(out)  # 通过第三层
        out = self.layer4(out)  # 通过第四层
        out = F.avg_pool2d(out, 4)  # 平均池化
        out = out.view(out.size(0), -1)  # 展平成一维
        out = self.fc(out)  # 通过全连接层
        return out  # 返回输出
    
def ResNet18():
    return ResNet(ResidualBlock)  # 返回ResNet18模型

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
EPOCH = 25  # 训练轮数
pre_epoch = 0  # 起始轮数
BATCH_SIZE = 8  # 批次大小
LR = 0.0004  # 学习率

# 准备数据集
train_labeled_set = GetTrainLoader()  # 获取训练数据集
trainloader = DataLoader(train_labeled_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)  # 训练数据加载器

test_labeled_set = GetTestLoader()  # 获取测试数据集
testloader = DataLoader(test_labeled_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)  # 测试数据加载器

# 定义 ResNet18 模型
net = ResNet18().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.8, weight_decay=5e-4)  # 随机梯度下降优化器

# 创建 GradScaler 实例
scaler = GradScaler()

# 记录每个epoch的训练损失、训练准确率和测试准确率
train_losses = []  # 存储每个epoch的训练损失
train_accuracies = []  # 存储每个epoch的训练准确率
test_accuracies = []  # 存储每个epoch的测试准确率

# 训练过程
total_start_time = time.time()  # 记录训练开始时间
for epoch in range(pre_epoch, EPOCH):

    print('\nEpoch: %d' % (epoch + 1))  # 打印当前epoch
    epoch_start_time = time.time()  # 记录每个epoch的开始时间
    net.train()  # 设置模型为训练模式
    sum_loss = 0.0  # 初始化损失和
    correct = 0.0  # 初始化正确预测数
    total = 0.0  # 初始化样本总数

    # 使用tqdm显示训练进度
    with tqdm(total=len(trainloader), desc=f'Epoch {epoch + 1}/{EPOCH}', unit='batch') as pbar:
        for i, data in enumerate(trainloader, 0):
            # 准备数据集
            inputs, labels, im_idx = data  # 获取输入数据和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移至GPU
            optimizer.zero_grad()  # 清空梯度

            # 使用autocast进行前向传播
            with autocast():
                outputs = net(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失

            # 使用scaler进行反向传播
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 更新缩放器

            # 计算损失和准确率
            sum_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 累加样本总数
            correct += predicted.eq(labels.data).cpu().sum()  # 累加正确预测数

            # 更新进度条
            pbar.set_postfix(loss=sum_loss / (i + 1), accuracy=100. * correct / total)  # 更新损失和准确率
            pbar.update(1)  # 更新进度条

    epoch_end_time = time.time()  # 记录每个epoch的结束时间
    epoch_duration = epoch_end_time - epoch_start_time  # 计算每个epoch的持续时间
    epoch_accuracy = 100. * correct / total  # 计算训练准确度

    # 记录训练损失和准确率
    train_losses.append(sum_loss / len(trainloader))  # 记录平均损失
    train_accuracies.append(epoch_accuracy.cpu().numpy())  # 记录训练准确率

    # 打印训练结果
    print(
        f'Epoch [{epoch + 1}/{EPOCH}], Loss: {sum_loss / len(trainloader):.4f}, Accuracy: {epoch_accuracy:.2f}%, Duration: {epoch_duration:.2f}s')

    # 测试集上的准确率
    print('Waiting Test...')
    with torch.no_grad():  # 关闭梯度计算
        correct = 0  # 初始化正确预测数
        total = 0  # 初始化样本总数
        for data in testloader:
            net.eval()  # 设置模型为评估模式
            inputs, labels, _ = data  # 获取输入数据和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移至GPU
            outputs = net(inputs)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 累加样本总数
            correct += (predicted == labels).sum()  # 累加正确预测数
        test_accuracy = 100. * correct.float() / total  # 计算测试准确率
        test_accuracies.append(test_accuracy.cpu().numpy())  # 记录测试准确率
        print('Test\'s accuracy is: %.3f%%' % test_accuracy)  # 打印测试准确率

    # 每个 epoch 结束后清理缓存
    torch.cuda.empty_cache()  # 清空缓存

total_end_time = time.time()  # 记录总训练时间的结束时间
total_duration = total_end_time - total_start_time  # 计算总训练时间

# 打印总训练时间
print(f'Total Training Time: {total_duration:.2f}s')

# 训练结束后保存模型权重
model_path = './resnet18_weights_v1.pth'  # 定义模型权重保存路径
torch.save(net.state_dict(), model_path)  # 保存模型的状态字典
print(f'Model weights saved to {model_path}')  # 打印保存路径

print('Train has finished, total epoch is %d' % EPOCH)  # 打印训练完成信息和总epoch数

# 绘制训练损失和准确率变化曲线
epochs = range(1, EPOCH + 1)  # 生成从1到EPOCH的序列
plt.figure(figsize=(12, 5))  # 设置图像大小

# 训练损失曲线
plt.subplot(1, 2, 1)  # 创建子图1
plt.plot(epochs, train_losses, 'orange', label='Training Loss')  # 绘制训练损失曲线
# 每五个epoch标记一个点并显示数值
for i in range(0, EPOCH, 2):
    plt.scatter(epochs[i], train_losses[i], color='orange')  # 标记点
    plt.text(epochs[i], train_losses[i] + 0.02, f'{train_losses[i]:.2f}', ha='left', va='top', color='orange', fontsize=8)  # 显示数值
plt.title('Training Loss')  # 设置标题
plt.xlabel('Epochs')  # 设置X轴标签
plt.ylabel('Loss')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.savefig('training_loss1.png')  # 保存训练损失图像

# 训练和测试准确率曲线
plt.subplot(1, 2, 2)  # 创建子图2
plt.plot(epochs, train_accuracies, 'orange', label='Training Accuracy')  # 绘制训练准确率曲线
plt.plot(epochs, test_accuracies, 'green', label='Test Accuracy')  # 绘制测试准确率曲线
# 每五个epoch标记一个点并显示数值
for i in range(0, EPOCH, 2):
    plt.scatter(epochs[i], train_accuracies[i], color='orange')  # 标记训练准确率点
    plt.text(epochs[i], train_accuracies[i] + 2, f'{train_accuracies[i]:.2f}', ha='left', va='top', color='orange', fontsize=8)  # 显示训练准确率数值
    plt.scatter(epochs[i], test_accuracies[i], color='green')  # 标记测试准确率点
    plt.text(epochs[i], test_accuracies[i] - 2, f'{test_accuracies[i]:.2f}', ha='left', va='top', color='green', fontsize=8)  # 显示测试准确率数值
plt.title('Training and Test Accuracy')  # 设置标题
plt.xlabel('Epochs')  # 设置X轴标签
plt.ylabel('Accuracy (%)')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.savefig('training_and_test_accuracy1.png')  # 保存训练和测试准确率图像

plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图像

# 最终测试并保存结果
net.eval()  # 设置模型为评估模式
with torch.no_grad():  # 禁用梯度计算，节省内存和加快计算速度
    with open("test.txt", 'w') as f:  # 打开文件以写入测试结果
        for idx, (inputs, targets, im_ids) in enumerate(trainloader):  # 遍历训练数据加载器中的每个批次
            inputs, targets = inputs.to(device), targets.to(device)  # 将输入和目标移动到设备（GPU或CPU）
            outputs = net(inputs)  # 前向传播，获得模型输出
            _, pred = outputs.topk(1, 1, True, True)  # 获取每个样本的最高概率预测
            pred = pred.t()  # 转置预测结果
            pred = pred.cpu().numpy().tolist()  # 将预测结果转移到CPU并转换为列表
            for i in range(inputs.shape[0]):  # 遍历当前批次中的每个样本
                f.write(im_ids[i] + ' ' + str(pred[0][i]) + '\n')  # 将图像ID和预测结果写入文件
    f.close()  # 关闭文件