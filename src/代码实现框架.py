#!/usr/bin/env python3
"""
车辆左转轨迹预测系统
基于历史轨迹的真正左转意图预测框架
重要更新：实现基于历史轨迹预测未来左转意图的真正预测任务
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 导入左转数据分析脚本
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from 左转数据分析脚本 import LeftTurnAnalyzer


# =============================
# 配置加载器
# =============================
def load_config(config_path=None):
    # 如果没有提供配置路径，使用脚本所在目录的config.yaml
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# =============================
# 数据管道类：结合 LeftTurnAnalyzer
# =============================
class DataPipeline:
    """数据处理管道类，用于统一管理从原始数据到数据集的转换过程"""
    def __init__(self, raw_path):
        self.raw_path = raw_path
    
    def build_dataset(self, int_id=None, approach=None, history_length=30, prediction_horizon=12, 
                     min_trajectory_length=100, use_prediction_mode=True, max_samples=None):
        """
        从原始数据构建数据集
        
        Args:
            int_id: 路口ID
            approach: 入口方向
            history_length: 历史轨迹长度
            prediction_horizon: 预测时间范围
            min_trajectory_length: 最小轨迹长度
            use_prediction_mode: 是否使用预测模式
            max_samples: 最大样本数限制
        
        Returns:
            MultiModalDataset: 构建好的数据集实例
        """
        print("🔄 开始构建数据集...")
        
        # 初始化LeftTurnAnalyzer时传入路口ID和入口方向
        analyzer = LeftTurnAnalyzer(self.raw_path, intersection_id=int_id, entrance_direction=approach)
        
        # 加载数据
        analyzer.load_data()
        
        # 如果是无特定路口ID的情况，确保selected_entrance为None以便筛选所有左转车辆
        if int_id is None:
            analyzer.selected_entrance = None
        
        # 筛选左转数据
        success = analyzer.filter_entrance_data()
        if not success:
            # 如果筛选失败但数据不为空，检查是否有左转车辆
            if analyzer.raw_data is not None and len(analyzer.raw_data) > 0:
                print("🔍 尝试筛选所有左转车辆...")
                if 'movement' in analyzer.raw_data.columns:
                    left_turn_data = analyzer.raw_data[analyzer.raw_data['movement'] == 2]
                    if len(left_turn_data) > 0:
                        analyzer.raw_data = left_turn_data
                        print(f"✅ 已筛选左转车辆数据: {len(analyzer.raw_data)} 条记录, {len(analyzer.raw_data['vehicle_id'].unique())} 辆车")
                    else:
                        raise ValueError("数据筛选失败：未找到左转车辆数据")
                else:
                    raise ValueError("数据筛选失败：数据中没有'movement'列")
            else:
                raise ValueError("数据筛选失败，请检查路口ID和方向是否正确")
        
        # 获取筛选后的左转数据
        filtered = analyzer.raw_data
        print(f"✅ 数据筛选完成: {len(filtered)} 条记录, {len(filtered['vehicle_id'].unique())} 辆车")
        
        # 直接使用DataFrame创建数据集，避免临时文件
        dataset = MultiModalDataset(
            data=filtered,  # 直接传递DataFrame
            history_length=history_length,
            prediction_horizon=prediction_horizon,
            min_trajectory_length=min_trajectory_length,
            use_prediction_mode=use_prediction_mode,
            max_samples=max_samples
        )
        
        return dataset
    
    def get_dataset_statistics(self, dataset):
        """获取数据集统计信息"""
        if hasattr(dataset, 'analyze_dataset'):
            return dataset.analyze_dataset()
        return {}
    
    def split_dataset(self, dataset, train_ratio=0.7, val_ratio=0.15):
        """
        将数据集划分为训练集、验证集和测试集
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        print(f"📊 数据集划分: 训练={train_size}, 验证={val_size}, 测试={test_size}")
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, dataset_size))
        
        return train_dataset, val_dataset, test_dataset
    

class MockDataset(Dataset):
    """模拟数据集类，用于演示和测试"""
    def __init__(self, size=1000, history_length=30):
        self.size = size
        self.history_length = history_length
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成与实际数据维度一致的模拟数据
        # visual_features: [history_length, 32]
        # motion_features: [history_length, 6]
        # traffic_features: [history_length, 4]
        # target_trajectory: [12, 2] (预测12个点的轨迹)
        return {
            'visual_features': torch.randn(self.history_length, 32),
            'motion_features': torch.randn(self.history_length, 6),
            'traffic_features': torch.randn(self.history_length, 4),
            'left_turn_intent': torch.rand(1),
            'target_trajectory': torch.randn(12, 2)
        }

class MultiModalDataset(Dataset):
    """多模态数据集类 - 支持真正的左转预测任务"""
    
    def __init__(self, data_path: str = None, data: pd.DataFrame = None, history_length: int = 30, prediction_horizon: int = 12, 
                 min_trajectory_length: int = 100, use_prediction_mode: bool = True, max_samples: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径 (与data参数二选一)
            data: 直接提供的DataFrame数据
            history_length: 历史轨迹长度 (帧数，默认30帧=3秒)
            prediction_horizon: 预测时间范围 (帧数，默认50帧=5秒)
            min_trajectory_length: 最小轨迹长度要求
            use_prediction_mode: 是否使用真正的预测模式
            max_samples: 最大样本数限制，用于缩减测试调试时间
        """
        # 参数检查
        if data is None and data_path is None:
            raise ValueError("必须提供data_path或data参数")
            
        self.data_path = data_path
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.min_trajectory_length = min_trajectory_length
        self.use_prediction_mode = use_prediction_mode
        self.max_samples = max_samples  
        
        # 为了兼容性，保留原有参数名
        self.sequence_length = history_length
        self.prediction_length = prediction_horizon
        
        print(f"🚀 初始化数据集 (预测模式: {'开启' if use_prediction_mode else '关闭'})")
        print(f"   历史长度: {history_length}帧 ({history_length*0.1:.1f}秒)")
        print(f"   预测范围: {prediction_horizon}帧 ({prediction_horizon*0.1:.1f}秒)")
        
        # 优先使用直接提供的DataFrame
        if data is not None:
            self.raw_data = data
            print(f"✅ 直接加载DataFrame数据: {len(data)} 条记录, {len(data['vehicle_id'].unique())} 辆车")
        else:
            self.raw_data = self.load_data()
        
        if use_prediction_mode:
            self.samples = self._build_prediction_samples()
            print(f"✅ 构建预测样本: {len(self.samples)} 个")
        else:
            # 兼容模式：使用原有逻辑
            self.data = self.raw_data
        
    def load_data(self):
        """加载预处理好的左转数据"""
        print(f"正在加载预处理数据: {self.data_path}")
        
        # 加载由数据预处理管道生成的数据
        data = pd.read_csv(self.data_path)
        
        print(f"数据加载成功: {len(data)} 条记录, {len(data['vehicle_id'].unique())} 辆车")
        
        # 检查是否包含必要的列
        required_columns = ['vehicle_id', 'frame_id', 'local_x', 'local_y']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必要列: {missing_columns}")
        
        # 如果有质量标记，优先使用高质量数据
        if 'is_high_quality' in data.columns:
            high_quality_data = data[data['is_high_quality'] == True]
            if len(high_quality_data) > 0:
                print(f"使用高质量数据: {len(high_quality_data)} 条记录, {len(high_quality_data['vehicle_id'].unique())} 辆车")
                data = high_quality_data
        
        # 按车辆ID和帧ID排序
        data = data.sort_values(['vehicle_id', 'frame_id'])
        
        return data
    
    def _build_prediction_samples(self):
        """构建真正的预测样本"""
        samples = []
        
        # 检查是否有movement字段
        if 'movement' not in self.raw_data.columns:
            print("⚠️  数据中没有movement字段，无法构建预测样本")
            return []
        
        # 按车辆分组
        vehicle_groups = self.raw_data.groupby('vehicle_id')
        
        print("🔄 构建预测样本...")
        valid_vehicles = 0
        total_vehicles = len(vehicle_groups)
        
        # 使用进度条显示处理进度
        with tqdm(total=total_vehicles, desc="处理车辆数据", unit="辆") as pbar:
            for vehicle_id, vehicle_data in vehicle_groups:
                # 按时间排序
                vehicle_data = vehicle_data.sort_values('frame_id').reset_index(drop=True)
                
                # 检查轨迹长度
                if len(vehicle_data) < self.min_trajectory_length:
                    pbar.update(1)
                    continue
                    
                valid_vehicles += 1
                
                # 滑动窗口构建样本
                total_length = self.history_length + self.prediction_horizon
                possible_windows = len(vehicle_data) - total_length + 1
                
                for i in range(possible_windows):
                    # 历史轨迹 (输入)
                    history_data = vehicle_data.iloc[i:i+self.history_length].copy()
                    
                    # 未来轨迹 (用于标签)
                    future_data = vehicle_data.iloc[i+self.history_length:i+total_length].copy()
                    
                    # 检查数据完整性
                    if len(history_data) != self.history_length or len(future_data) != self.prediction_horizon:
                        continue
                    
                    # 提取标签 (未来是否有左转)
                    future_movements = future_data['movement'].values
                    has_left_turn = np.any(future_movements == 2.0)  # 2.0 = 左转
                    
                    # 额外信息
                    sample_info = {
                        'vehicle_id': vehicle_id,
                        'start_frame': history_data['frame_id'].iloc[0],
                        'end_frame': future_data['frame_id'].iloc[-1],
                        'history_trajectory': history_data[['local_x', 'local_y']].values,
                        'future_trajectory': future_data[['local_x', 'local_y']].values,
                        'future_movements': future_movements  # 保留原始movement值用于验证
                    }
                    
                    samples.append({
                        'history_data': history_data,
                        'future_data': future_data,
                        'label': int(has_left_turn),
                        'info': sample_info
                    })
                    
                    # 如果达到最大样本数限制，提前结束
                    if self.max_samples and len(samples) >= self.max_samples:
                        print(f"🔄 已达到最大样本数限制 ({self.max_samples}个样本)")
                        pbar.update(total_vehicles - pbar.n)  # 更新进度条到100%
                        return samples[:self.max_samples]
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '有效车辆': valid_vehicles,
                    '样本数': len(samples)
                })
        
        print(f"✅ 构建完成，有效车辆: {valid_vehicles} 辆, 总样本数: {len(samples)} 个")
        return samples
    
    def __len__(self):
        if self.use_prediction_mode and hasattr(self, 'samples'):
            return len(self.samples)
        else:
            return len(self.data) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        """获取单个样本"""
        if self.use_prediction_mode and hasattr(self, 'samples'):
            # 真正的预测模式
            return self._get_prediction_sample(idx)
        else:
            # 兼容模式：使用原有逻辑
            return self._get_legacy_sample(idx)
    
    def _get_prediction_sample(self, idx):
        """获取预测样本"""
        if idx >= len(self.samples):
            idx = idx % len(self.samples)
        
        sample = self.samples[idx]
        history_data = sample['history_data']
        future_data = sample['future_data']
        label = sample['label']
        
        # 提取多模态特征（基于历史数据）
        visual_features = self.extract_visual_features(history_data)
        motion_features = self.extract_motion_features(history_data)
        traffic_features = self.extract_traffic_features(history_data)
        
        # 目标轨迹（未来轨迹）
        target_trajectory = future_data[['local_x', 'local_y']].values
        
        # 提取未来movement字段作为未来值
        future_movements = sample['info'].get('future_movements', np.array([]))
        
        return {
            'visual_features': torch.FloatTensor(visual_features),
            'motion_features': torch.FloatTensor(motion_features),
            'traffic_features': torch.FloatTensor(traffic_features),
            'left_turn_intent': torch.FloatTensor([float(label)]),  # 真正的预测标签
            'target_trajectory': torch.FloatTensor(target_trajectory),
            'future_movements': torch.FloatTensor(future_movements),  # 添加未来movement字段作为未来值
            'sample_info': sample['info']  # 额外信息
        }
    
    def _get_legacy_sample(self, idx):
        """获取传统样本（兼容模式）"""
        # 获取车辆序列数据
        vehicle_ids = self.data['vehicle_id'].unique()
        
        # 简化处理：按索引获取车辆数据
        if idx < len(vehicle_ids):
            vehicle_id = vehicle_ids[idx]
            vehicle_data = self.data[self.data['vehicle_id'] == vehicle_id]
            
            # 确保有足够的数据点
            if len(vehicle_data) < self.sequence_length + self.prediction_length:
                # 如果数据不足，使用填充或跳过
                vehicle_data = vehicle_data.iloc[:self.sequence_length + self.prediction_length]
            
            # 历史轨迹
            history = vehicle_data.iloc[:self.sequence_length]
            
            # 未来轨迹
            future = vehicle_data.iloc[self.sequence_length:self.sequence_length + self.prediction_length]
            
            # 提取多模态特征
            visual_features = self.extract_visual_features(history)
            motion_features = self.extract_motion_features(history)
            traffic_features = self.extract_traffic_features(history)
            
            # 左转意图标签（从预处理数据中获取）
            left_turn_intent = self.get_left_turn_intent(vehicle_data)
        else:
            # 索引超出范围，返回默认数据
            return self._get_legacy_sample(idx % len(vehicle_ids))
        
        # 目标轨迹（使用NGSIM数据的local_x, local_y列）
        if 'local_x' in future.columns and 'local_y' in future.columns:
            target_trajectory = future[['local_x', 'local_y']].values
        else:
            # 如果没有位置列，创建模拟数据
            target_trajectory = np.random.randn(len(future), 2)
        
        return {
            'visual_features': torch.FloatTensor(visual_features),
            'motion_features': torch.FloatTensor(motion_features),
            'traffic_features': torch.FloatTensor(traffic_features),
            'left_turn_intent': torch.FloatTensor([left_turn_intent]),
            'target_trajectory': torch.FloatTensor(target_trajectory)
        }
    
    def extract_visual_features(self, history):
        """
        提取视觉特征。由于NGSIM没有图像，这里是一个抽象实现。
        """
        # 这里的实现是抽象的，代表一个未来可以集成的视觉模型。
        # 如果有图像数据，可以用预训练的CNN（如ResNet）来提取特征。
        # 对于NGSIM数据，可以考虑将车辆轨迹和周围环境信息栅格化为一张"地图图像"，
        # 然后用一个小的CNN来处理这张图像。
        
        # 简单占位符实现：返回一个维度合适的随机张量
        # 假设视觉特征维度为32
        return torch.rand(self.history_length, 32).numpy()
    
    def extract_motion_features(self, history):
        """
        从车辆轨迹数据中提取运动学特征，并计算导数。
        特征包括：x, y, 速度, 加速度, 航向角, 航向角变化率。
        """
        # 提取基础运动学特征
        trajectory_data = history[['local_x', 'local_y', 'v_vel', 'v_acc']].values.astype(np.float32)

        # 确保数据帧足够长以计算导数
        if len(trajectory_data) < 2:
            # 如果轨迹太短，返回一个填充了0的张量
            return np.zeros((self.history_length, 6), dtype=np.float32)

        # 计算航向角（heading angle），使用atan2来处理所有象限
        dx = np.diff(trajectory_data[:, 0])
        dy = np.diff(trajectory_data[:, 1])
        # 使用np.arctan2计算航向角，可以处理所有象限
        heading = np.arctan2(dy, dx)
        heading = np.insert(heading, 0, heading[0] if len(heading) > 0 else 0)  # 在第一帧补上一个值

        # 计算航向角变化率（heading rate of change）
        # 考虑角度的周期性，使用np.unwrap来避免跳变
        unwrapped_heading = np.unwrap(heading)
        heading_rate = np.diff(unwrapped_heading)
        heading_rate = np.insert(heading_rate, 0, 0)  # 第一帧变化率为0

        # 将所有特征拼接起来
        motion_features = np.stack([
            trajectory_data[:, 0],  # local_x
            trajectory_data[:, 1],  # local_y
            trajectory_data[:, 2],  # v_vel
            trajectory_data[:, 3],  # v_acc
            heading,               # 航向角
            heading_rate           # 航向角变化率
        ], axis=1)

        return motion_features
    
    def extract_traffic_features(self, history):
        """
        提取交通环境特征，如与前后车的相对距离和相对速度。
        """
        # 获取当前车辆的ID
        current_vehicle_id = history['vehicle_id'].iloc[0]

        # 提取前后车ID
        preceding_id = history['preceding'].iloc[0]
        following_id = history['following'].iloc[0]

        # 默认特征，如果前后车不存在，则为0
        preceding_features = np.zeros(2)
        following_features = np.zeros(2)

        # 查找前车并计算相对特征
        if preceding_id > 0:
            preceding_vehicle_df = self.raw_data[self.raw_data['vehicle_id'] == preceding_id]
            if not preceding_vehicle_df.empty:
                # 找到同一帧的前车数据
                current_frame = history['frame_id'].iloc[0]
                preceding_frame_df = preceding_vehicle_df[preceding_vehicle_df['frame_id'] == current_frame]
                if not preceding_frame_df.empty:
                    preceding_pos = preceding_frame_df['local_y'].iloc[0]
                    preceding_vel = preceding_frame_df['v_vel'].iloc[0]
                    
                    # 相对位置和相对速度
                    relative_y = preceding_pos - history['local_y'].iloc[0]
                    relative_v = preceding_vel - history['v_vel'].iloc[0]
                    preceding_features = np.array([relative_y, relative_v], dtype=np.float32)

        # 查找后车并计算相对特征
        if following_id > 0:
            following_vehicle_df = self.raw_data[self.raw_data['vehicle_id'] == following_id]
            if not following_vehicle_df.empty:
                # 找到同一帧的后车数据
                current_frame = history['frame_id'].iloc[0]
                following_frame_df = following_vehicle_df[following_vehicle_df['frame_id'] == current_frame]
                if not following_frame_df.empty:
                    following_pos = following_frame_df['local_y'].iloc[0]
                    following_vel = following_frame_df['v_vel'].iloc[0]

                    # 相对位置和相对速度
                    relative_y = following_pos - history['local_y'].iloc[0]
                    relative_v = following_vel - history['v_vel'].iloc[0]
                    following_features = np.array([relative_y, relative_v], dtype=np.float32)

        # 将所有交通特征拼接起来
        traffic_features = np.concatenate([preceding_features, following_features])
        
        # 扩展为时间序列格式
        traffic_features = np.tile(traffic_features, (self.history_length, 1))
        
        return traffic_features
    
    def get_left_turn_intent(self, vehicle_data):
        """从数据中获取左转意图标签"""
        # 优先使用预处理数据的质量标记
        if 'is_high_quality' in vehicle_data.columns:
            return 1.0 if vehicle_data['is_high_quality'].iloc[0] else 0.0
        # 其次使用NGSIM原始movement标签
        elif 'movement' in vehicle_data.columns:
            # 检查是否有左转标签
            has_left_turn = np.any(vehicle_data['movement'] == 2.0)
            return 1.0 if has_left_turn else 0.0
        else:
            # 兼容性处理：默认为左转数据
            return 1.0
    
    def analyze_dataset(self):
        """分析数据集统计信息"""
        if self.use_prediction_mode and hasattr(self, 'samples'):
            print(f"📊 预测数据集分析:")
            print(f"   总样本数: {len(self.samples):,}")
            
            # 使用进度条进行数据分析
            with tqdm(total=4, desc="分析进度", unit="项") as pbar:
                # 标签分布
                labels = [sample['label'] for sample in self.samples]
                left_turn_count = sum(labels)
                non_left_turn_count = len(labels) - left_turn_count
                
                print(f"   左转样本: {left_turn_count:,} ({left_turn_count/len(labels)*100:.1f}%)")
                print(f"   非左转样本: {non_left_turn_count:,} ({non_left_turn_count/len(labels)*100:.1f}%)")
                pbar.update(1)
                
                # 车辆分布
                vehicle_ids = [sample['info']['vehicle_id'] for sample in self.samples]
                unique_vehicles = len(set(vehicle_ids))
                print(f"   涉及车辆数: {unique_vehicles:,}")
                print(f"   平均每车样本数: {len(self.samples)/unique_vehicles:.1f}")
                pbar.update(1)
                
                # 时间跨度分析
                if len(self.samples) > 0:
                    sample_info = self.samples[0]['info']
                    time_span = sample_info['end_frame'] - sample_info['start_frame']
                    print(f"   样本时间跨度: {time_span}帧 ({time_span*0.1:.1f}秒)")
                pbar.update(1)
                
                # 位置范围分析
                if len(self.samples) > 0:
                    all_positions = []
                    for sample in self.samples[:1000]:  # 只分析前1000个样本以加速
                        all_positions.extend(sample['info']['history_trajectory'])
                        all_positions.extend(sample['info']['future_trajectory'])
                    
                    if all_positions:
                        positions_array = np.array(all_positions)
                        min_x, min_y = np.min(positions_array, axis=0)
                        max_x, max_y = np.max(positions_array, axis=0)
                        print(f"   轨迹空间范围: x[{min_x:.1f}, {max_x:.1f}], y[{min_y:.1f}, {max_y:.1f}]")
                pbar.update(1)
        else:
            print(f"📊 传统数据集分析:")
            print(f"   数据记录数: {len(self.data):,}")
            
            with tqdm(total=2, desc="分析进度", unit="项") as pbar:
                # 车辆统计
                print(f"   车辆数: {len(self.data['vehicle_id'].unique()):,}")
                pbar.update(1)
                
                # 移动类型统计
                if 'movement' in self.data.columns:
                    movement_counts = self.data['movement'].value_counts()
                    print(f"   Movement分布:")
                    for movement, count in movement_counts.items():
                        movement_name = {1.0: '直行', 2.0: '左转', 3.0: '右转'}.get(movement, f'其他({movement})')
                        print(f"     {movement_name}: {count:,} ({count/len(self.data)*100:.1f}%)")
                pbar.update(1)

class VisualEncoder(nn.Module):
    """视觉特征编码器"""
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        # 视觉特征是序列形式 [batch_size, history_length, input_dim]
        # 使用LSTM处理序列特征
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        
        # 输出层
        self.out_layer = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [batch_size, history_length, input_dim]
        # 使用LSTM处理序列特征
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        
        # 通过输出层
        out = self.out_layer(last_out)
        out = self.relu(out)
        
        return out

class MotionEncoder(nn.Module):
    """运动特征编码器 - 处理序列形式的运动学特征"""
    
    def __init__(self, num_features: int = 6, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        # 直接使用运动学特征的数量作为输入维度
        # 假设输入是 [batch_size, history_length, num_features] 格式
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # x: [batch_size, history_length, num_features]
        # LSTM直接处理序列特征
        output, (hidden, cell) = self.lstm(x)
        # 使用最后一个时间步的输出
        return self.fc(output[:, -1, :])

class TrafficEncoder(nn.Module):
    """交通环境特征编码器 - 处理序列形式的交通环境特征"""
    
    def __init__(self, num_features: int = 4, hidden_dim: int = 128):
        super().__init__()
        # 交通环境特征是序列形式 [batch_size, history_length, num_features]
        # 使用LSTM处理序列特征
        self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        
        # 输出层
        self.out_layer = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [batch_size, history_length, num_features]
        # 使用LSTM处理序列特征
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        
        # 通过输出层
        out = self.out_layer(last_out)
        out = self.relu(out)
        
        return out

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
        fused_features = cross_attended.contiguous().view(cross_attended.size(0), -1)
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
        h0 = init_states[:, :self.hidden_dim*2].reshape(2, batch_size, self.hidden_dim)
        c0 = init_states[:, self.hidden_dim*2:].reshape(2, batch_size, self.hidden_dim)
        
        # 解码预测轨迹
        outputs = []
        hidden = (h0, c0)
        
        # 第一步输入 - 使用固定的输入特征
        decoder_input = input_features.unsqueeze(1)  # [batch, 1, input_dim]
        
        for t in range(self.seq_len):
            output, hidden = self.lstm(decoder_input, hidden)
            trajectory_point = self.output_layer(output)
            outputs.append(trajectory_point)
            
            # 保持输入维度一致，不添加trajectory_point
            # 使用相同的input_features作为下一步输入
            decoder_input = input_features.unsqueeze(1)
        
        # 拼接所有输出
        trajectory = torch.cat(outputs, dim=1)
        
        return trajectory

class LeftTurnPredictor(nn.Module):
    """
    左转预测模型，采用多模态融合架构
    整合了视觉、运动和交通环境特征进行左转意图预测和轨迹预测
    """

    def __init__(self, history_horizon: int = 50, num_motion_features: int = 6, 
                 num_traffic_features: int = 4, num_visual_features: int = 32, prediction_horizon: int = 12):
        super().__init__()
        
        # 运动学特征编码器（使用LSTM）
        self.motion_encoder = MotionEncoder(num_features=num_motion_features)
        
        # 交通环境特征编码器
        self.traffic_encoder = TrafficEncoder(num_features=num_traffic_features)
        
        # 视觉特征编码器
        self.visual_encoder = VisualEncoder(input_dim=num_visual_features)

        # 注意力融合模块
        self.attention_fusion = AttentionFusion()

        # 意图分类器
        self.intent_classifier = IntentClassifier()

        # 轨迹解码器 - 使用动态预测长度
        self.trajectory_decoder = TrajectoryDecoder(seq_len=prediction_horizon)

    def forward(self, visual_feat, motion_feat, traffic_feat):
        # 特征编码
        # 所有特征都是序列形式 [batch_size, history_length, feature_dim]
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
    
    def train_epoch(self, epoch_num=None, total_epochs=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        intent_correct = 0
        total_samples = 0
        traj_error = 0
        
        # 创建进度条
        desc = f"Epoch {epoch_num}/{total_epochs} [Train]" if epoch_num and total_epochs else "Training"
        pbar = tqdm(self.train_loader, desc=desc, leave=False)
        
        batch_losses = []
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
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
            batch_loss = total_batch_loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            intent_correct += ((intent_pred > 0.5) == (intent_target > 0.5)).sum().item()
            total_samples += intent_target.size(0)
            traj_error += torch.sqrt(torch.mean((traj_pred - traj_target) ** 2)).item()
            
            # 更新进度条信息
            if batch_idx % 5 == 0:  # 每5个batch更新一次
                current_avg_loss = np.mean(batch_losses[-10:]) if len(batch_losses) >= 10 else np.mean(batch_losses)
                current_intent_acc = intent_correct / total_samples if total_samples > 0 else 0
                
                # 计算处理速度
                elapsed_time = time.time() - batch_start_time
                samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
                
                pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.4f}',
                    'IntentAcc': f'{current_intent_acc:.3f}',
                    'Speed': f'{samples_per_sec:.1f}samples/s'
                })
        
        pbar.close()
        
        avg_loss = total_loss / len(self.train_loader)
        intent_acc = intent_correct / total_samples
        avg_traj_error = traj_error / len(self.train_loader)
        
        return avg_loss, intent_acc, avg_traj_error
    
    def validate(self, epoch_num=None, total_epochs=None):
        """验证"""
        self.model.eval()
        total_loss = 0
        intent_correct = 0
        total_samples = 0
        traj_error = 0
        
        # 创建验证进度条
        desc = f"Epoch {epoch_num}/{total_epochs} [Valid]" if epoch_num and total_epochs else "Validating"
        pbar = tqdm(self.val_loader, desc=desc, leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
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
                
                # 更新进度条信息
                if batch_idx % 3 == 0:  # 验证时更频繁更新
                    current_avg_loss = total_loss / (batch_idx + 1)
                    current_intent_acc = intent_correct / total_samples if total_samples > 0 else 0
                    
                    pbar.set_postfix({
                        'Loss': f'{current_avg_loss:.4f}',
                        'IntentAcc': f'{current_intent_acc:.3f}'
                    })
        
        pbar.close()
        
        avg_loss = total_loss / len(self.val_loader)
        intent_acc = intent_correct / total_samples
        avg_traj_error = traj_error / len(self.val_loader)
        
        return avg_loss, intent_acc, avg_traj_error
    
    def train(self, epochs: int = 50, early_stopping_patience: int = 15):
        """完整训练流程"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("🚀 开始训练...")
        print(f"📊 训练集大小: {len(self.train_loader.dataset):,}")
        print(f"📊 验证集大小: {len(self.val_loader.dataset):,}")
        print(f"🎯 目标轮数: {epochs}")
        print(f"⏰ 早停耐心: {early_stopping_patience}")
        print(f"💻 设备: {self.device}")
        print("=" * 80)
        
        # 总体进度条
        epoch_pbar = tqdm(range(epochs), desc="Overall Progress", position=0)
        
        training_start_time = time.time()
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_intent_acc, train_traj_error = self.train_epoch(epoch+1, epochs)
            
            # 验证
            val_loss, val_intent_acc, val_traj_error = self.validate(epoch+1, epochs)
            
            # 学习率调度
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['intent_acc'].append(train_intent_acc)
            self.train_history['traj_error'].append(train_traj_error)
            
            self.val_history['loss'].append(val_loss)
            self.val_history['intent_acc'].append(val_intent_acc)
            self.val_history['traj_error'].append(val_traj_error)
            
            # 计算时间统计
            epoch_time = time.time() - epoch_start_time
            total_elapsed = time.time() - training_start_time
            avg_epoch_time = total_elapsed / (epoch + 1)
            eta = avg_epoch_time * (epochs - epoch - 1)
            
            # 早停检查和模型保存
            improvement = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                improvement = "💾 [BEST]"
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    improvement = "⏹️ [EARLY STOP]"
            
            # 学习率变化提示
            lr_info = f"📉 LR: {old_lr:.2e}" if new_lr == old_lr else f"📉 LR: {old_lr:.2e}→{new_lr:.2e}"
            
            # 更新总体进度条
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_loss:.4f}',
                'Val_Loss': f'{val_loss:.4f}',
                'Val_Acc': f'{val_intent_acc:.3f}',
                'Patience': f'{patience_counter}/{early_stopping_patience}',
                'ETA': f'{eta/60:.1f}min'
            })
            
            # 详细信息输出
            epoch_info = f"📈 Epoch {epoch+1:3d}/{epochs} | " + \
                        f"⏱️ {epoch_time:.1f}s | " + \
                        f"🔄 Train: {train_loss:.4f} | " + \
                        f"✅ Valid: {val_loss:.4f} | " + \
                        f"🎯 Acc: {val_intent_acc:.3f} | " + \
                        f"📏 TrajErr: {val_traj_error:.3f}"
            print(epoch_info)
            
            detail_info = f"    {lr_info} | " + \
                         f"⏳ ETA: {eta/60:.1f}min | " + \
                         f"🕐 Total: {total_elapsed/60:.1f}min | " + \
                         f"{improvement}"
            print(detail_info)
            
            if patience_counter >= early_stopping_patience:
                print(f"⏹️ 早停触发！在第 {epoch+1} 轮停止训练")
                print(f"🏆 最佳验证损失: {best_val_loss:.4f}")
                break
        
        epoch_pbar.close()
        
        total_training_time = time.time() - training_start_time
        print("" + "=" * 80)
        print("🎉 训练完成！")
        print(f"⏱️ 总训练时间: {total_training_time/60:.1f} 分钟")
        print(f"🏆 最佳验证损失: {best_val_loss:.4f}")
        print(f"💾 最佳模型已保存为: best_model.pth")
        print("=" * 80)
        
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
    all_future_movements = []  # 存储future_movements用于与真实左转数据验证
    
    print("📈 开始评估模型性能...")
    
    # 使用进度条显示评估进度
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="评估进度", unit="批") as pbar:
            for batch_idx, batch in enumerate(test_loader):
                visual_feat = batch['visual_features'].to(device)
                motion_feat = batch['motion_features'].to(device)
                traffic_feat = batch['traffic_features'].to(device)
                intent_target = batch['left_turn_intent'].to(device)
                traj_target = batch['target_trajectory'].to(device)
                
                # 检查是否包含future_movements字段（用于与真实左转数据验证）
                if 'future_movements' in batch:
                    all_future_movements.extend(batch['future_movements'])
                
                intent_pred, traj_pred = model(visual_feat, motion_feat, traffic_feat)
                
                all_intent_preds.append(intent_pred.cpu().numpy())
                all_intent_targets.append(intent_target.cpu().numpy())
                all_traj_preds.append(traj_pred.cpu().numpy())
                all_traj_targets.append(traj_target.cpu().numpy())
                
                # 更新进度条
                pbar.update(1)
                
                # 定期显示当前进度的统计信息
                if batch_idx % 10 == 0 or batch_idx == len(test_loader) - 1:
                    # 计算当前的简单统计
                    current_samples = (batch_idx + 1) * len(batch['visual_features'])
                    pbar.set_postfix({
                        '已处理样本': current_samples
                    })
    
    print("✅ 模型评估完成，开始计算性能指标...")
    
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
    
    # 打印评估结果
    print("============================================================")
    print("                        模型评估结果")
    print("============================================================")
    print(f"意图识别准确率: {intent_accuracy:.4f}")
    print(f"意图识别精确率: {intent_precision:.4f}")
    print(f"意图识别召回率: {intent_recall:.4f}")
    print(f"意图识别F1分数: {intent_f1:.4f}")
    print("-" * 40)
    print(f"轨迹预测ADE: {ade:.4f} m")
    print(f"轨迹预测FDE: {fde:.4f} m")
    print("=" * 60)
    
    # 准备返回结果
    results = {
        'intent_accuracy': intent_accuracy,
        'intent_precision': intent_precision,
        'intent_recall': intent_recall,
        'intent_f1': intent_f1,
        'trajectory_ade': ade,
        'trajectory_fde': fde
    }
    
    # 如果存在future_movements数据，进行与真实左转数据的验证分析
    if all_future_movements:
        # 基于future_movements确定真正的左转车辆
        true_left_turns = [1 if any(m == 'L' for m in movements) else 0 for movements in all_future_movements]
        
        # 计算基于真实左转标签的准确率
        true_left_accuracy = accuracy_score(true_left_turns, intent_binary_preds[:len(true_left_turns)])
        true_left_precision = precision_score(true_left_turns, intent_binary_preds[:len(true_left_turns)], zero_division=0)
        true_left_recall = recall_score(true_left_turns, intent_binary_preds[:len(true_left_turns)], zero_division=0)
        true_left_f1 = f1_score(true_left_turns, intent_binary_preds[:len(true_left_turns)], zero_division=0)
        
        # 打印基于真实左转数据的验证结果
        print("📊 基于真实左转数据的验证结果 (与左转数据分析脚本对比):")
        print(f"   - 真正左转车辆数: {sum(true_left_turns)}/{len(true_left_turns)}")
        print(f"   - 预测准确率: {true_left_accuracy:.4f}")
        print(f"   - 精确率: {true_left_precision:.4f}")
        print(f"   - 召回率: {true_left_recall:.4f}")
        print(f"   - F1分数: {true_left_f1:.4f}")
        print("=" * 60)
        
        # 将验证结果添加到返回字典
        results.update({
            'true_left_turns_count': sum(true_left_turns),
            'true_left_turns_total': len(true_left_turns),
            'true_left_accuracy': true_left_accuracy,
            'true_left_precision': true_left_precision,
            'true_left_recall': true_left_recall,
            'true_left_f1': true_left_f1
        })
    
    return results

def main():
    """主函数"""
    print("🚀 车辆左转轨迹预测系统")
    print("基于历史轨迹的真正左转意图预测")
    print("=" * 60)
    data_dir = os.path.join(os.path.dirname(__file__), "../data")

    # 加载配置 (使用默认路径，自动定位到脚本所在目录的config.yaml)
    config = load_config()

    raw_csv_file =  config.get("raw_csv_file", "peachtree_filtered_data.csv")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")
    
    # 选择预测模式
    print("🎯 选择预测模式:")
    print("1. 真正的预测模式 (推荐) - 基于历史轨迹预测未来左转意图")
    print("2. 传统模式 - 多模态特征融合")
    
    mode_choice = input("请选择模式 (1/2, 默认: 1): ").strip()
    use_prediction_mode = mode_choice != '2'
    
    # 添加全局样本数限制参数
    max_samples_input = input("请输入要处理的最大样本数 (默认: 全部，输入正整数可缩减调试时间): ").strip()

    max_samples = config.get("max_samples", -1)
    max_samples = int(max_samples_input) if max_samples_input and max_samples_input.isdigit() and int(max_samples_input)>0 else max_samples
    
    if use_prediction_mode:
        print("✅ 使用真正的预测模式")
        print("   - 历史长度: 30帧 (3秒)")
        print("   - 预测范围: 50帧 (5秒)")
        print("   - 利用NGSIM movement标签进行真正的预测")
        if max_samples>0:
            print(f"   - 限制最大样本数: {max_samples}")
    else:
        print("✅ 使用传统多模态模式")
        if max_samples>0:
            print(f"   - 限制最大样本数: {max_samples}")
    
    # 创建数据集
    print("🔄 创建数据集...")
    try:
        # 获取数据路径
        raw_csv_file_fullpath = f"{data_dir}/{raw_csv_file}"
        data_path_input = input(f"请输入NGSIM数据路径 (默认: {raw_csv_file_fullpath}): ").strip()
        data_path_fullpath = data_path_input if data_path_input else raw_csv_file_fullpath
        
        # 检查数据是否存在
        if not os.path.exists(data_path_fullpath):
            print(f"❌ 数据文件不存在: {data_path_fullpath}")
            print("💡 请确保NGSIM数据文件存在")
            return
            
        # 创建DataPipeline实例
        data_pipeline = DataPipeline(data_path_fullpath)
        
        # 获取路口和方向信息（如果需要）
        int_id = config.get("int_id", 1)
        approach = config.get("approach", "northbound")
        
        if use_prediction_mode:
            # 预测模式下可以选择特定路口和方向
            filter_input = input("是否按路口和方向筛选数据？(y/n, 默认: y): ").strip().lower()
            filter_input = 'y' if not filter_input else 'n'
            if filter_input == 'y':
                try:
                    # 先使用LeftTurnAnalyzer发现并显示数据中的路口
                    print("🔍 正在分析数据中的路口信息...")
                    analyzer = LeftTurnAnalyzer(data_path_fullpath)
                    analyzer.load_data()
                    intersections = analyzer.discover_intersections()
                    
                    if intersections:
                        print("\n📋 数据中发现的路口信息：")
                        print("=" * 80)
                        print(f"{'路口ID':<8} {'总记录数':<12} {'车辆数':<10} {'方向':<20} {'机动类型':<15}")
                        print("-" * 80)
                        
                        for int_id_available, info in sorted(intersections.items()):
                            # 转换方向为名称
                            direction_names = []
                            for direction in info['directions'][:4]:  # 最多显示4个方向
                                if direction in analyzer.direction_names:
                                    direction_names.append(f"{direction}({analyzer.direction_names[direction].split(' ')[0]})")
                                else:
                                    direction_names.append(str(direction))
                            directions_str = ','.join(direction_names)
                            
                            # 转换机动类型为名称
                            movement_names = []
                            for movement in info['movements'][:4]:  # 最多显示4个机动类型
                                if movement in analyzer.movement_names:
                                    movement_names.append(f"{movement}({analyzer.movement_names[movement].split(' ')[0]})")
                                else:
                                    movement_names.append(str(movement))
                            movements_str = ','.join(movement_names)
                            
                            print(f"{int_id_available:<8} {info['total_records']:<12} {info['total_vehicles']:<10} {directions_str:<20} {movements_str:<15}")
                        print("=" * 80)
                    else:
                        print("⚠️ 未能在数据中发现路口信息")
                    
                    # 然后让用户选择路口ID
                    int_id_input = input("请输入路口ID (留空不筛选): ").strip()
                    int_id = int(int_id_input) if int_id_input else None
                    
                    if int_id is not None:
                        # 分析该路口的可用入口方向
                        try:
                            entrance_analyzer = LeftTurnAnalyzer(data_path_fullpath)
                            entrance_analyzer.load_data()
                            entrance_analyzer.intersection_id = int_id
                            entrance_stats = entrance_analyzer.analyze_intersection_entrances()
                            
                            if entrance_stats:
                                print(f"\n✅ 路口 {int_id} 的可用入口方向信息：")
                                print("=" * 70)
                                print(f"{'方向编号':<10} {'方向名称':<10} {'总车辆':<10} {'左转车辆':<10} {'左转比例':<10}")
                                print("-" * 70)
                                
                                for stats in entrance_stats.values():
                                    print(f"{stats['direction']:<10} {stats['direction_name']:<10} {stats['total_vehicles']:<10} {stats['left_turn_vehicles']:<10} {stats['left_turn_ratio']:.1f}%")
                                print("=" * 70)
                        except Exception as e:
                            print(f"⚠️ 分析入口方向时出错: {e}")
                        
                        approach_input = input("请输入入口方向 (1-东, 2-北, 3-西, 4-南, 留空不筛选): ").strip()
                        approach = int(approach_input) if approach_input and approach_input.isdigit() else None
                except ValueError:
                    print("⚠️ 无效的路口ID或方向，将使用所有数据")
        else:
            # 传统模式下需要路口和方向信息
            try:
                int_id_input = input("请输入路口ID (默认: 1): ").strip()
                int_id = int(int_id_input) if int_id_input else 1
                
                # 获取入口方向
                print("请选择入口方向:")
                print("1 - 东向")
                print("2 - 北向")
                print("3 - 西向")
                print("4 - 南向")
                entrance_direction_input = input("请选择入口方向 (默认: 1): ").strip()
                approach = int(entrance_direction_input) if entrance_direction_input else 1
            except Exception as e:
                print(f"❌ 获取路口和方向信息失败: {e}")
                print("💡 将使用默认值")
                int_id = 1
                approach = 1
        
        # 使用DataPipeline构建数据集
        history_length = config.get("history_length", 30)
        prediction_horizon = config.get("prediction_horizon", 12)  # 修改为12以匹配模型输出

        epochs = config.get("epochs", 50)
        epochs_input = input("请输入训练轮数 epochs (默认: epochs=50): ").strip()
        epochs = int(epochs_input) if epochs_input else epochs

        full_dataset = data_pipeline.build_dataset(
            int_id=int_id,
            approach=approach,
            history_length=history_length if use_prediction_mode else 8,
            prediction_horizon=prediction_horizon if use_prediction_mode else 12,
            min_trajectory_length=100 if use_prediction_mode else 20,
            use_prediction_mode=use_prediction_mode,
            max_samples=max_samples
        )
        
        # 分析数据集
        data_pipeline.get_dataset_statistics(full_dataset)
        
        # 使用DataPipeline进行数据集划分
        train_dataset, val_dataset, test_dataset = data_pipeline.split_dataset(full_dataset)
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        print("🔄 使用模拟数据进行演示...")
        # 使用与配置一致的history_length
        train_dataset = MockDataset(800, history_length=history_length)
        val_dataset = MockDataset(200, history_length=history_length)
        test_dataset = MockDataset(200, history_length=history_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型 - 传入预测长度参数
    print("创建模型...")
    model = LeftTurnPredictor(prediction_horizon=prediction_horizon)
    
    # 创建训练管理器
    trainer = TrainingManager(model, train_loader, val_loader, device)
    
    # 训练模型
    print("开始训练...")
    train_history, val_history = trainer.train(epochs=epochs)
    
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