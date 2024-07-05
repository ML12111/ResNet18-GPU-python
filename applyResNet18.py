import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('agg')  # 设置matplotlib使用非交互式后端
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=5):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(14336, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(f"Output shape: {out.shape}")
        out = self.fc(out)
        # print(f"Output shape: {out.shape}")
        return out

def ResNet18():
    return ResNet(ResidualBlock)

class ResNet18ImageClassifier:
    def __init__(self, model_path, class_names, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path, num_classes)
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize((135, 240)),
            transforms.ToTensor(),
        ])

    def _load_model(self, model_path, num_classes):
        model = ResNet(ResidualBlock, num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def classify_image(self, image_path):
        image = Image.open(image_path)
        image = image.crop([300, 80, 1620, 950])
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

        # 获取最大概率的类别
        max_prob_index = probabilities.argmax()
        sort = {
            0: "三柱顺转",
            1: "T型顺转",
            2: "单柱顺转",
            3: "四柱顺转",
            4: "双柱顺转"
        }[max_prob_index]

        probabilities = {self.class_names[i]: probabilities[i] for i in [1, 2, 4, 0, 3]}
        return probabilities, sort

def plot_probabilities2(probabilities, save_path):
    class_names = ['T', 'sin', 'dou', 'tri', 'for']  # 修改这里的顺序
    probs = [probabilities[class_name] for class_name in class_names]

    plt.figure(figsize=(5, 4))
    plt.bar(class_names, probs, color='orange')
    plt.xlabel('Classes')
    plt.ylabel('Probabilities')
    plt.title('ResNet18-Classfier Probabilities')
    plt.savefig(save_path)
    plt.close()  # 关闭图表，释放内存

if __name__ == "__main__":
    model_path = "ResNet18_weights.pth"
    image_path = "tri.jpg"
    class_names = ['tri', 'T', 'sin', 'for', 'dou']
    classifier = ResNet18ImageClassifier(model_path, class_names)
    probabilities, sort = classifier.classify_image(image_path)

    for class_name, prob in probabilities.items():
        print(f"{class_name}: {prob:.7f}")

    # 输出最大概率的类别
    print(f"最大概率类别: {sort}")

    # 绘制并保存柱形图
    save_path = 'ResNet-Classfier Probabilities.png'
    plot_probabilities2(probabilities, save_path)
    print("柱形图已保存为 ResNet-Classfier Probabilities.png")