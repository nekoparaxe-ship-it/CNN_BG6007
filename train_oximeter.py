# train_oximeter.py

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 第1步: 定义CNN模型架构
# 这个版本是修正后的，它能自动计算维度，避免了手动设置可能导致的错误。
class OximeterCNN(nn.Module):
    def __init__(self):
        super(OximeterCNN, self).__init__()
        # 论文描述了1个2D卷积层和2个1D卷积层
        # 输入是 (batch_size, 1, 3, 90) -> (批大小, 通道, RGB, 帧数)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 12)),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 12)),
            nn.ReLU()
        )

        # === 鲁棒性修正：自动计算全连接层的输入维度 ===
        # 1. 创建一个与真实输入形状相同的虚拟张量
        dummy_input = torch.randn(1, 1, 3, 90)
        # 2. 将虚拟张量通过卷积层
        dummy_output = self.conv_layers(dummy_input)
        # 3. 计算输出的扁平化尺寸
        #    dummy_output.shape[0]是批大小，这里是1，所以用numel()计算总元素数量
        n_features = dummy_output.numel()

        # 论文描述了2个全连接线性层
        self.fc_layers = nn.Sequential(
            nn.Linear(n_features, 10),
            nn.Linear(10, 1)  # 最终输出1个SpO2值
        )

    def forward(self, x):
        # 增加一个通道维度以适应Conv2d
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        # 扁平化以送入全连接层
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 第2步: 创建自定义数据集类
# 这个类用于加载.h5文件，并将其处理成模型可用的样本
class OximetryDataset(Dataset):
    def __init__(self, h5_path, subject_indices):
        self.h5_path = h5_path
        self.subject_indices = subject_indices

        self.data_samples = []
        self.ground_truths = []

        # 使用h5py加载数据
        with h5py.File(self.h5_path, 'r') as f:
            full_data = f['dataset'][:]
            full_gt = f['groundtruth'][:]

        # 论文使用3秒(90帧)的片段作为输入
        window_size = 90

        for subj_idx in self.subject_indices:
            # 每个受试者有左右手数据，我们这里将它们都视为独立样本
            for hand_offset in [0, 3]:  # 0-2是左手RGB, 3-5是右手RGB
                # 提取一个受试者一只手的数据
                subj_data = full_data[subj_idx, hand_offset:hand_offset + 3, :]
                # 使用Masimo Radical-7作为地面真实值 (论文中提到，索引为4)
                subj_gt = full_gt[subj_idx, 4, :]

                # 找到有效数据长度 (数据末尾可能以0填充)
                valid_gt_len = np.where(subj_gt == 0)[0]
                valid_gt_len = valid_gt_len[0] if len(valid_gt_len) > 0 else len(subj_gt)

                # 创建滑窗样本
                for i in range(valid_gt_len):
                    start_frame = i * 30
                    end_frame = start_frame + window_size
                    if end_frame > subj_data.shape[1]:
                        break

                    sample = subj_data[:, start_frame:end_frame]
                    # 论文排除了低于70% SpO2的数据
                    if 70 <= subj_gt[i] <= 100:
                        self.data_samples.append(torch.tensor(sample, dtype=torch.float32))
                        self.ground_truths.append(torch.tensor(subj_gt[i], dtype=torch.float32))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx], self.ground_truths[idx]


# 第3步: 训练和评估的主流程
def main():
    # 请确保这个路径与您存放数据的位置一致
    h5_file_path = './dataset/data/preprocessed/all_uw_data.h5'
    num_subjects = 6  # 论文研究了6名受试者
    all_subject_ids = list(range(num_subjects))

    # 存储每个LOOCV折叠的结果
    test_maes = []

    # 实施留一交叉验证 (LOOCV)
    for test_subject_id in all_subject_ids:
        print(f"\n--- 开始训练: 将受试者 {test_subject_id + 1} 作为测试集 ---")

        train_subject_ids = [i for i in all_subject_ids if i != test_subject_id]

        # 创建数据集和数据加载器
        train_dataset = OximetryDataset(h5_file_path, train_subject_ids)
        test_dataset = OximetryDataset(h5_file_path, [test_subject_id])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化模型、损失函数和优化器
        model = OximeterCNN()
        # 论文使用均方误差作为损失函数
        criterion = nn.MSELoss()
        # 论文使用Adam优化器和L2正则化。根据建议，我们将学习率提高到0.0001
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.1)

        # 开始训练循环，周期数增加到100
        num_epochs = 80
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # 每10个周期打印一次损失，避免刷屏
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, 训练损失: {running_loss / len(train_loader):.4f}")

        # 在测试集上评估
        model.eval()
        total_mae = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                # 评估指标使用平均绝对误差 (MAE)
                mae = torch.abs(outputs - labels).sum()
                total_mae += mae.item()

        avg_mae = total_mae / len(test_dataset)
        test_maes.append(avg_mae)
        print(f"受试者 {test_subject_id + 1} 的测试MAE: {avg_mae:.4f} %SpO2")

    # 打印最终结果
    final_avg_mae = np.mean(test_maes)
    print(f"\n--- 训练完成 ---")
    print(f"所有折叠的平均MAE: {final_avg_mae:.4f} %SpO2")

    # 提示：实际应用中，您可以在这里保存最优的模型
    torch.save(model.state_dict(), 'oximeter_model.pth')


if __name__ == '__main__':
    main()
