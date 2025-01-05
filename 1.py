import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 构建ResNet模型
class PneumoniaDetector(nn.Module):
    def __init__(self, model_name='resnet18'):
        super(PneumoniaDetector, self).__init__()
        self.base_model = getattr(models, model_name)(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 2)  # 二分类问题

    def forward(self, x):
        return self.base_model(x)

# 定义模型创建函数
def create_model(learning_rate=0.001, model_name='resnet18'):
    model = PneumoniaDetector(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

# 定义超参数网格
param_grid = {
    'learning_rate': [0.001, 0.0001],
    'model_name': ['resnet18', 'resnet50']
}

# 自定义评分函数
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return all_labels, all_preds

# 自定义评分器类
class CustomScorer:
    def __call__(self, model, X, y):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in X:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

# 初始化模型、损失函数和优化器
model, criterion, optimizer = create_model()

# 超参数优化过程
best_score = 0
best_params = None
for learning_rate in param_grid['learning_rate']:
    for model_name in param_grid['model_name']:
        model, criterion, optimizer = create_model(learning_rate, model_name)
        scorer = CustomScorer()
        score = scorer(model, train_loader, None)
        print(f"Model: {model_name}, Learning Rate: {learning_rate}, Score: {score}")
        if score > best_score:
            best_score = score
            best_params = {'learning_rate': learning_rate, 'model_name': model_name}

print(f"Best Model: {best_params}")

# 使用最佳超参数重新训练模型
best_model, best_criterion, best_optimizer = create_model(best_params['learning_rate'], best_params['model_name'])
num_epochs =8
scheduler = torch.optim.lr_scheduler.StepLR(best_optimizer, step_size=7, gamma=0.1)

for epoch in range(num_epochs):
    best_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = best_model(images)
        loss = best_criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 在测试集上评估模型性能
best_model.eval()
all_labels, all_preds = evaluate_model(best_model, test_loader)
cm = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:\n', cm)
roc_auc = roc_curve(all_labels, all_preds)
fpr, tpr, thresholds = roc_auc
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print(classification_report(all_labels, all_preds))
