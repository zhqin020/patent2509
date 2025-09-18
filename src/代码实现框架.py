#!/usr/bin/env python3
"""
车辆左转轨迹预测系统
基于多模态深度学习的实现框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class MultiModalDataset(Dataset):
    """多模态数据集类"""
    
    def __init__(self, data_path: str, sequence_length: int = 8, prediction_length: int = 12):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            sequence_length: 历史轨迹长度
            prediction_length: 预测轨迹长度
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.data = self.load_data()
        
    def load_data(self):
        """加载和预处理数据"""
        # 加载NGSIM数据或自定义数据
        data = pd.read_csv(self.data_path)
        
        # 数据预处理
        processed_data = self.preprocess_data(data)
        
        return processed_data
    
    def preprocess_data(self, data):
        """数据预处理"""
        # 轨迹平滑
        data = self.smooth_trajectories(data)
        
        # 特征工程
        data = self.extract_features(data)
        
        # 标准化
        data = self.normalize_features(data)
        
        return data
    
    def smooth_trajectories(self, data):
        """轨迹平滑处理"""
        # 使用卡尔曼滤波或移动平均进行轨迹平滑
        return data
    
    def extract_features(self, data):
        """特征提取"""
        # 计算速度、加速度、航向角等特征
        data['velocity'] = np.sqrt(data['vx']**2 + data['vy']**2)
        data['acceleration'] = data['velocity'].diff()
        data['heading'] = np.arctan2(data['vy'], data['vx'])
        
        return data
    
    def normalize_features(self, data):
        """特征标准化"""
        # Z-score标准化
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()
        
        return data
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 历史轨迹
        history = self.data.iloc[idx:idx+self.sequence_length]
        
        # 未来轨迹
        future = self.data.iloc[idx+self.sequence_length:idx+self.sequence_length+self.prediction_length]
        
        # 提取多模态特征
        visual_features = self.extract_visual_features(history)
        motion_features = self.extract_motion_features(history)
        traffic_features = self.extract_traffic_features(history)
        
        # 左转意图标签
        left_turn_intent = self.get_left_turn_intent(future)
        
        # 目标轨迹
        target_trajectory = future[['x', 'y']].values
        
        return {
            'visual_features': torch.FloatTensor(visual_features),
            'motion_features': torch.FloatTensor(motion_features),
            'traffic_features': torch.FloatTensor(traffic_features),
            'left_turn_intent': torch.FloatTensor([left_turn_intent]),
            'target_trajectory': torch.FloatTensor(target_trajectory)
        }
    
    def extract_visual_features(self, history):
        """提取视觉特征"""
        # 这里应该从视频帧中提取视觉特征
        # 简化实现，返回随机特征
        return np.random.randn(64)
    
    def extract_motion_features(self, history):
        """提取运动特征"""
        features = []
        for col in ['x', 'y', 'velocity', 'acceleration', 'heading']:
            if col in history.columns:
                features.extend(history[col].values)
        return np.array(features)
    
    def extract_traffic_features(self, history):
        """提取交通环境特征"""
        # 简化实现，返回随机特征
        return np.random.randn(32)
    
    def get_left_turn_intent(self, future):
        """判断左转意图"""
        # 基于未来轨迹判断是否为左转
        start_heading = np.arctan2(future.iloc[0]['vy'], future.iloc[0]['vx'])
        end_heading = np.arctan2(future.iloc[-1]['vy'], future.iloc[-1]['vx'])
        
        heading_change = end_heading - start_heading
        
        # 如果航向角变化大于阈值，认为是左转
        return 1.0 if heading_change > np.pi/4 else 0.0

class VisualEncoder(nn.Module):
    """视觉特征编码器"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.encoder(x)

class MotionEncoder(nn.Module):
    """运动特征编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim // 8,  # 假设有8个时间步
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # 重塑输入为(batch_size, seq_len, feature_dim)
        batch_size = x.size(0)
        seq_len = 8
        feature_dim = x.size(1) // seq_len
        x = x.view(batch_size, seq_len, feature_dim)
        
        output, (hidden, cell) = self.lstm(x)
        # 使用最后一个时间步的输出
        return self.fc(output[:, -1, :])

class TrafficEncoder(nn.Module):
    """交通环境特征编码器"""
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.encoder(x)

class AttentionFusion(nn.Module):
    """注意力融合模块"""
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 自注意力机制
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # 输出投影
        self.output_proj = nn.Linear(feature_dim * 3, feature_dim)
    
    def forward(self, visual_feat, motion_feat, traffic_feat):
        # 堆叠特征
        features = torch.stack([visual_feat, motion_feat, traffic_feat], dim=1)
        
        # 自注意力
        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)
        
        attention_weights = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.feature_dim), 
            dim=-1
        )
        attended_features = torch.matmul(attention_weights, V)
        
        # 跨模态注意力
        attended_features = attended_features.transpose(0, 1)
        cross_attended, _ = self.cross_attention(
            attended_features, attended_features, attended_features
        )
        cross_attended = cross_attended.transpose(0, 1)
        
        # 融合特征
        fused_features = cross_attended.view(cross_attended.size(0), -1)
        output = self.output_proj(fused_features)
        
        return output

class IntentClassifier(nn.Module):
    """左转意图分类器"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x)

class TrajectoryDecoder(nn.Module):
    """轨迹预测解码器"""
    
    def __init__(self, input_dim: int = 129, hidden_dim: int = 128, output_dim: int = 2, seq_len: int = 12):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 初始隐藏状态
        self.init_hidden = nn.Linear(input_dim, hidden_dim * 2 * 2)  # 2 layers * 2 (h,c)
    
    def forward(self, fused_features, intent_prob):
        batch_size = fused_features.size(0)
        
        # 结合意图信息
        input_features = torch.cat([fused_features, intent_prob], dim=1)
        
        # 初始化隐藏状态
        init_states = self.init_hidden(input_features)
        h0 = init_states[:, :self.hidden_dim*2].view(2, batch_size, self.hidden_dim).contiguous()
        c0 = init_states[:, self.hidden_dim*2:].view(2, batch_size, self.hidden_dim).contiguous()
        
        # 解码预测轨迹
        outputs = []
        hidden = (h0, c0)
        
        # 第一步输入
        decoder_input = input_features.unsqueeze(1)
        
        for t in range(self.seq_len):
            output, hidden = self.lstm(decoder_input, hidden)
            trajectory_point = self.output_layer(output)
            outputs.append(trajectory_point)
            
            # 下一步的输入（教师强制或自回归）
            decoder_input = torch.cat([
                fused_features.unsqueeze(1), 
                intent_prob.unsqueeze(1),
                trajectory_point
            ], dim=2)
        
        # 拼接所有输出
        trajectory = torch.cat(outputs, dim=1)
        
        return trajectory

class LeftTurnPredictor(nn.Module):
    """左转轨迹预测主模型"""
    
    def __init__(self, visual_dim: int = 64, motion_dim: int = 40, traffic_dim: int = 32):
        super().__init__()
        
        # 特征编码器
        self.visual_encoder = VisualEncoder(visual_dim)
        self.motion_encoder = MotionEncoder(motion_dim)
        self.traffic_encoder = TrafficEncoder(traffic_dim)
        
        # 注意力融合
        self.attention_fusion = AttentionFusion()
        
        # 意图分类器
        self.intent_classifier = IntentClassifier()
        
        # 轨迹解码器
        self.trajectory_decoder = TrajectoryDecoder()
    
    def forward(self, visual_feat, motion_feat, traffic_feat):
        # 特征编码
        visual_encoded = self.visual_encoder(visual_feat)
        motion_encoded = self.motion_encoder(motion_feat)
        traffic_encoded = self.traffic_encoder(traffic_feat)
        
        # 多模态融合
        fused_features = self.attention_fusion(
            visual_encoded, motion_encoded, traffic_encoded
        )
        
        # 意图预测
        intent_prob = self.intent_classifier(fused_features)
        
        # 轨迹预测
        trajectory = self.trajectory_decoder(fused_features, intent_prob)
        
        return intent_prob, trajectory

class TrainingManager:
    """训练管理器"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # 损失函数
        self.intent_loss_fn = nn.BCELoss()
        self.trajectory_loss_fn = nn.MSELoss()
        
        # 训练历史
        self.train_history = {'loss': [], 'intent_acc': [], 'traj_error': []}
        self.val_history = {'loss': [], 'intent_acc': [], 'traj_error': []}
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        intent_correct = 0
        total_samples = 0
        traj_error = 0
        
        for batch in self.train_loader:
            # 数据移到设备
            visual_feat = batch['visual_features'].to(self.device)
            motion_feat = batch['motion_features'].to(self.device)
            traffic_feat = batch['traffic_features'].to(self.device)
            intent_target = batch['left_turn_intent'].to(self.device)
            traj_target = batch['target_trajectory'].to(self.device)
            
            # 前向传播
            intent_pred, traj_pred = self.model(visual_feat, motion_feat, traffic_feat)
            
            # 计算损失
            intent_loss = self.intent_loss_fn(intent_pred, intent_target)
            traj_loss = self.trajectory_loss_fn(traj_pred, traj_target)
            
            # 联合损失
            total_batch_loss = intent_loss + 0.5 * traj_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += total_batch_loss.item()
            intent_correct += ((intent_pred > 0.5) == (intent_target > 0.5)).sum().item()
            total_samples += intent_target.size(0)
            traj_error += torch.sqrt(torch.mean((traj_pred - traj_target) ** 2)).item()
        
        avg_loss = total_loss / len(self.train_loader)
        intent_acc = intent_correct / total_samples
        avg_traj_error = traj_error / len(self.train_loader)
        
        return avg_loss, intent_acc, avg_traj_error
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        intent_correct = 0
        total_samples = 0
        traj_error = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 数据移到设备
                visual_feat = batch['visual_features'].to(self.device)
                motion_feat = batch['motion_features'].to(self.device)
                traffic_feat = batch['traffic_features'].to(self.device)
                intent_target = batch['left_turn_intent'].to(self.device)
                traj_target = batch['target_trajectory'].to(self.device)
                
                # 前向传播
                intent_pred, traj_pred = self.model(visual_feat, motion_feat, traffic_feat)
                
                # 计算损失
                intent_loss = self.intent_loss_fn(intent_pred, intent_target)
                traj_loss = self.trajectory_loss_fn(traj_pred, traj_target)
                total_batch_loss = intent_loss + 0.5 * traj_loss
                
                # 统计
                total_loss += total_batch_loss.item()
                intent_correct += ((intent_pred > 0.5) == (intent_target > 0.5)).sum().item()
                total_samples += intent_target.size(0)
                traj_error += torch.sqrt(torch.mean((traj_pred - traj_target) ** 2)).item()
        
        avg_loss = total_loss / len(self.val_loader)
        intent_acc = intent_correct / total_samples
        avg_traj_error = traj_error / len(self.val_loader)
        
        return avg_loss, intent_acc, avg_traj_error
    
    def train(self, epochs: int = 100, early_stopping_patience: int = 15):
        """完整训练流程"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("开始训练...")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print("-" * 60)
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_intent_acc, train_traj_error = self.train_epoch()
            
            # 验证
            val_loss, val_intent_acc, val_traj_error = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['intent_acc'].append(train_intent_acc)
            self.train_history['traj_error'].append(train_traj_error)
            
            self.val_history['loss'].append(val_loss)
            self.val_history['intent_acc'].append(val_intent_acc)
            self.val_history['traj_error'].append(val_traj_error)
            
            # 打印进度
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Intent Acc: {val_intent_acc:.3f} | "
                  f"Traj Error: {val_traj_error:.3f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        print("训练完成！")
        return self.train_history, self.val_history
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.train_history['loss'], label='Train Loss')
        axes[0].plot(self.val_history['loss'], label='Val Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 意图识别准确率
        axes[1].plot(self.train_history['intent_acc'], label='Train Acc')
        axes[1].plot(self.val_history['intent_acc'], label='Val Acc')
        axes[1].set_title('Intent Classification Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # 轨迹预测误差
        axes[2].plot(self.train_history['traj_error'], label='Train Error')
        axes[2].plot(self.val_history['traj_error'], label='Val Error')
        axes[2].set_title('Trajectory Prediction Error')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('RMSE')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def evaluate_model(model, test_loader, device='cuda'):
    """模型评估"""
    model.eval()
    
    all_intent_preds = []
    all_intent_targets = []
    all_traj_preds = []
    all_traj_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            visual_feat = batch['visual_features'].to(device)
            motion_feat = batch['motion_features'].to(device)
            traffic_feat = batch['traffic_features'].to(device)
            intent_target = batch['left_turn_intent'].to(device)
            traj_target = batch['target_trajectory'].to(device)
            
            intent_pred, traj_pred = model(visual_feat, motion_feat, traffic_feat)
            
            all_intent_preds.append(intent_pred.cpu().numpy())
            all_intent_targets.append(intent_target.cpu().numpy())
            all_traj_preds.append(traj_pred.cpu().numpy())
            all_traj_targets.append(traj_target.cpu().numpy())
    
    # 合并结果
    intent_preds = np.concatenate(all_intent_preds)
    intent_targets = np.concatenate(all_intent_targets)
    traj_preds = np.concatenate(all_traj_preds)
    traj_targets = np.concatenate(all_traj_targets)
    
    # 计算评估指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # 意图识别指标
    intent_binary_preds = (intent_preds > 0.5).astype(int)
    intent_binary_targets = (intent_targets > 0.5).astype(int)
    
    intent_accuracy = accuracy_score(intent_binary_targets, intent_binary_preds)
    intent_precision = precision_score(intent_binary_targets, intent_binary_preds)
    intent_recall = recall_score(intent_binary_targets, intent_binary_preds)
    intent_f1 = f1_score(intent_binary_targets, intent_binary_preds)
    
    # 轨迹预测指标
    # ADE (Average Displacement Error)
    ade = np.mean(np.sqrt(np.sum((traj_preds - traj_targets) ** 2, axis=2)))
    
    # FDE (Final Displacement Error)
    fde = np.mean(np.sqrt(np.sum((traj_preds[:, -1, :] - traj_targets[:, -1, :]) ** 2, axis=1)))
    
    print("=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"意图识别准确率: {intent_accuracy:.4f}")
    print(f"意图识别精确率: {intent_precision:.4f}")
    print(f"意图识别召回率: {intent_recall:.4f}")
    print(f"意图识别F1分数: {intent_f1:.4f}")
    print("-" * 40)
    print(f"轨迹预测ADE: {ade:.4f} m")
    print(f"轨迹预测FDE: {fde:.4f} m")
    print("=" * 60)
    
    return {
        'intent_accuracy': intent_accuracy,
        'intent_precision': intent_precision,
        'intent_recall': intent_recall,
        'intent_f1': intent_f1,
        'trajectory_ade': ade,
        'trajectory_fde': fde
    }

def main():
    """主函数"""
    print("车辆左转轨迹预测系统")
    print("基于多模态深度学习")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模拟数据集（实际使用时替换为真实数据）
    print("创建数据集...")
    
    # 这里应该使用真实的数据路径
    # train_dataset = MultiModalDataset('train_data.csv')
    # val_dataset = MultiModalDataset('val_data.csv')
    # test_dataset = MultiModalDataset('test_data.csv')
    
    # 为演示创建模拟数据
    class MockDataset(Dataset):
        def __init__(self, size=1000):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'visual_features': torch.randn(64),
                'motion_features': torch.randn(40),
                'traffic_features': torch.randn(32),
                'left_turn_intent': torch.rand(1),
                'target_trajectory': torch.randn(12, 2)
            }
    
    train_dataset = MockDataset(800)
    val_dataset = MockDataset(200)
    test_dataset = MockDataset(200)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    print("创建模型...")
    model = LeftTurnPredictor()
    
    # 创建训练管理器
    trainer = TrainingManager(model, train_loader, val_loader, device)
    
    # 训练模型
    print("开始训练...")
    train_history, val_history = trainer.train(epochs=50)
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 评估模型
    print("评估模型...")
    results = evaluate_model(model, test_loader, device)
    
    print("训练和评估完成！")

if __name__ == "__main__":
    main()