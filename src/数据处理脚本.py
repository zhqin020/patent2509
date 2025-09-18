#!/usr/bin/env python3
"""
NGSIM数据处理脚本
专门用于提取和处理左转车辆轨迹数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

class NGSIMDataProcessor:
    """NGSIM数据处理器"""
    
    def __init__(self, data_path: str):
        """
        初始化数据处理器
        
        Args:
            data_path: NGSIM数据文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.left_turn_data = None
        
    def load_data(self):
        """加载NGSIM数据"""
        print("正在加载NGSIM数据...")
        
        try:
            # 根据文件扩展名选择读取方式
            if self.data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.txt'):
                # NGSIM原始数据通常是空格分隔的文本文件
                column_names = [
                    'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time',
                    'Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_Length',
                    'v_Width', 'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID',
                    'O_Zone', 'D_Zone', 'Int_ID', 'Section_ID', 'Direction',
                    'Movement', 'Preceding', 'Following', 'Space_Headway',
                    'Time_Headway'
                ]
                self.raw_data = pd.read_csv(self.data_path, sep='\s+', names=column_names)
            
            print(f"数据加载完成，共 {len(self.raw_data)} 条记录")
            print(f"数据列: {list(self.raw_data.columns)}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
        
        return True
    
    def preprocess_data(self):
        """数据预处理"""
        print("开始数据预处理...")
        
        if self.raw_data is None:
            print("请先加载数据")
            return False
        
        # 复制原始数据
        self.processed_data = self.raw_data.copy()
        
        # 1. 数据清洗
        self._clean_data()
        
        # 2. 特征工程
        self._feature_engineering()
        
        # 3. 轨迹平滑
        self._smooth_trajectories()
        
        # 4. 标准化坐标
        self._normalize_coordinates()
        
        print("数据预处理完成")
        return True
    
    def _clean_data(self):
        """数据清洗"""
        print("  - 数据清洗...")
        
        # 删除缺失值
        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data.dropna()
        print(f"    删除缺失值: {initial_count - len(self.processed_data)} 条")
        
        # 删除异常值（速度过大或过小）
        if 'v_Vel' in self.processed_data.columns:
            speed_threshold = 50  # m/s
            valid_speed = (self.processed_data['v_Vel'] >= 0) & (self.processed_data['v_Vel'] <= speed_threshold)
            removed_count = len(self.processed_data) - valid_speed.sum()
            self.processed_data = self.processed_data[valid_speed]
            print(f"    删除异常速度记录: {removed_count} 条")
        
        # 删除轨迹点过少的车辆
        min_trajectory_length = 10
        vehicle_counts = self.processed_data['Vehicle_ID'].value_counts()
        valid_vehicles = vehicle_counts[vehicle_counts >= min_trajectory_length].index
        self.processed_data = self.processed_data[self.processed_data['Vehicle_ID'].isin(valid_vehicles)]
        print(f"    保留轨迹点足够的车辆: {len(valid_vehicles)} 辆")
    
    def _feature_engineering(self):
        """特征工程"""
        print("  - 特征工程...")
        
        # 按车辆ID和时间排序
        self.processed_data = self.processed_data.sort_values(['Vehicle_ID', 'Frame_ID'])
        
        # 计算速度和加速度（如果没有的话）
        if 'v_Vel' not in self.processed_data.columns:
            self.processed_data['v_Vel'] = self._calculate_velocity()
        
        if 'v_Acc' not in self.processed_data.columns:
            self.processed_data['v_Acc'] = self._calculate_acceleration()
        
        # 计算航向角
        self.processed_data['Heading'] = self._calculate_heading()
        
        # 计算角速度
        self.processed_data['Angular_Velocity'] = self._calculate_angular_velocity()
        
        # 计算与车道中心线的距离
        self.processed_data['Lane_Offset'] = self._calculate_lane_offset()
        
        # 计算相对位置特征
        self.processed_data['Relative_X'] = self._calculate_relative_position('Local_X')
        self.processed_data['Relative_Y'] = self._calculate_relative_position('Local_Y')
    
    def _calculate_velocity(self):
        """计算速度"""
        velocities = []
        
        for vehicle_id in self.processed_data['Vehicle_ID'].unique():
            vehicle_data = self.processed_data[self.processed_data['Vehicle_ID'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('Frame_ID')
            
            # 计算位置差分
            dx = vehicle_data['Local_X'].diff()
            dy = vehicle_data['Local_Y'].diff()
            dt = vehicle_data['Frame_ID'].diff() * 0.1  # 假设每帧0.1秒
            
            # 计算速度大小
            velocity = np.sqrt(dx**2 + dy**2) / dt
            velocity.iloc[0] = velocity.iloc[1]  # 填充第一个值
            
            velocities.extend(velocity.values)
        
        return velocities
    
    def _calculate_acceleration(self):
        """计算加速度"""
        accelerations = []
        
        for vehicle_id in self.processed_data['Vehicle_ID'].unique():
            vehicle_data = self.processed_data[self.processed_data['Vehicle_ID'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('Frame_ID')
            
            # 计算速度差分
            dv = vehicle_data['v_Vel'].diff()
            dt = vehicle_data['Frame_ID'].diff() * 0.1
            
            acceleration = dv / dt
            acceleration.iloc[0] = 0  # 第一个值设为0
            
            accelerations.extend(acceleration.values)
        
        return accelerations
    
    def _calculate_heading(self):
        """计算航向角"""
        headings = []
        
        for vehicle_id in self.processed_data['Vehicle_ID'].unique():
            vehicle_data = self.processed_data[self.processed_data['Vehicle_ID'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('Frame_ID')
            
            # 计算航向角
            dx = vehicle_data['Local_X'].diff()
            dy = vehicle_data['Local_Y'].diff()
            
            heading = np.arctan2(dy, dx)
            heading.iloc[0] = heading.iloc[1]  # 填充第一个值
            
            headings.extend(heading.values)
        
        return headings
    
    def _calculate_angular_velocity(self):
        """计算角速度"""
        angular_velocities = []
        
        for vehicle_id in self.processed_data['Vehicle_ID'].unique():
            vehicle_data = self.processed_data[self.processed_data['Vehicle_ID'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('Frame_ID')
            
            # 计算角速度
            dheading = vehicle_data['Heading'].diff()
            dt = vehicle_data['Frame_ID'].diff() * 0.1
            
            # 处理角度跳跃
            dheading = np.where(dheading > np.pi, dheading - 2*np.pi, dheading)
            dheading = np.where(dheading < -np.pi, dheading + 2*np.pi, dheading)
            
            angular_velocity = dheading / dt
            angular_velocity.iloc[0] = 0
            
            angular_velocities.extend(angular_velocity.values)
        
        return angular_velocities
    
    def _calculate_lane_offset(self):
        """计算与车道中心线的距离"""
        # 简化实现，实际应根据道路几何结构计算
        return np.random.normal(0, 0.5, len(self.processed_data))
    
    def _calculate_relative_position(self, coord_column):
        """计算相对位置"""
        relative_positions = []
        
        for vehicle_id in self.processed_data['Vehicle_ID'].unique():
            vehicle_data = self.processed_data[self.processed_data['Vehicle_ID'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('Frame_ID')
            
            # 以第一个位置为原点
            relative_pos = vehicle_data[coord_column] - vehicle_data[coord_column].iloc[0]
            relative_positions.extend(relative_pos.values)
        
        return relative_positions
    
    def _smooth_trajectories(self):
        """轨迹平滑"""
        print("  - 轨迹平滑...")
        
        for vehicle_id in self.processed_data['Vehicle_ID'].unique():
            mask = self.processed_data['Vehicle_ID'] == vehicle_id
            vehicle_data = self.processed_data[mask].copy()
            
            if len(vehicle_data) >= 5:  # 至少需要5个点进行平滑
                # 使用Savitzky-Golay滤波器平滑轨迹
                window_length = min(5, len(vehicle_data) if len(vehicle_data) % 2 == 1 else len(vehicle_data) - 1)
                
                try:
                    smoothed_x = savgol_filter(vehicle_data['Local_X'], window_length, 3)
                    smoothed_y = savgol_filter(vehicle_data['Local_Y'], window_length, 3)
                    
                    self.processed_data.loc[mask, 'Local_X'] = smoothed_x
                    self.processed_data.loc[mask, 'Local_Y'] = smoothed_y
                except:
                    # 如果平滑失败，保持原始数据
                    pass
    
    def _normalize_coordinates(self):
        """标准化坐标"""
        print("  - 坐标标准化...")
        
        # 将坐标转换为相对于交叉口中心的坐标
        if 'Global_X' in self.processed_data.columns and 'Global_Y' in self.processed_data.columns:
            center_x = self.processed_data['Global_X'].mean()
            center_y = self.processed_data['Global_Y'].mean()
            
            self.processed_data['Normalized_X'] = self.processed_data['Global_X'] - center_x
            self.processed_data['Normalized_Y'] = self.processed_data['Global_Y'] - center_y
    
    def identify_left_turn_vehicles(self, heading_threshold=np.pi/3, min_heading_change=np.pi/6):
        """识别左转车辆"""
        print("识别左转车辆...")
        
        if self.processed_data is None:
            print("请先进行数据预处理")
            return False
        
        left_turn_vehicles = []
        
        for vehicle_id in self.processed_data['Vehicle_ID'].unique():
            vehicle_data = self.processed_data[self.processed_data['Vehicle_ID'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('Frame_ID')
            
            if len(vehicle_data) < 10:  # 轨迹太短，跳过
                continue
            
            # 计算总的航向角变化
            initial_heading = vehicle_data['Heading'].iloc[0]
            final_heading = vehicle_data['Heading'].iloc[-1]
            
            # 处理角度跳跃
            heading_change = final_heading - initial_heading
            if heading_change > np.pi:
                heading_change -= 2 * np.pi
            elif heading_change < -np.pi:
                heading_change += 2 * np.pi
            
            # 判断是否为左转（正的航向角变化）
            if heading_change > min_heading_change:
                # 进一步检查轨迹形状
                if self._is_left_turn_trajectory(vehicle_data):
                    left_turn_vehicles.append(vehicle_id)
        
        # 提取左转车辆数据
        self.left_turn_data = self.processed_data[
            self.processed_data['Vehicle_ID'].isin(left_turn_vehicles)
        ].copy()
        
        print(f"识别出 {len(left_turn_vehicles)} 辆左转车辆")
        print(f"左转轨迹数据点: {len(self.left_turn_data)} 条")
        
        return True
    
    def _is_left_turn_trajectory(self, vehicle_data):
        """判断是否为左转轨迹"""
        # 简化的左转判断逻辑
        # 实际应用中可以使用更复杂的几何分析
        
        # 检查轨迹的曲率
        x = vehicle_data['Local_X'].values
        y = vehicle_data['Local_Y'].values
        
        if len(x) < 5:
            return False
        
        # 计算轨迹的曲率
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        
        # 如果平均曲率大于阈值，认为是转弯轨迹
        avg_curvature = np.mean(curvature[~np.isnan(curvature)])
        
        return avg_curvature > 0.01  # 可调整的阈值
    
    def extract_trajectory_sequences(self, sequence_length=8, prediction_length=12, overlap=0.5):
        """提取轨迹序列用于训练"""
        print("提取轨迹序列...")
        
        if self.left_turn_data is None:
            print("请先识别左转车辆")
            return None
        
        sequences = []
        
        for vehicle_id in self.left_turn_data['Vehicle_ID'].unique():
            vehicle_data = self.left_turn_data[self.left_turn_data['Vehicle_ID'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('Frame_ID').reset_index(drop=True)
            
            total_length = sequence_length + prediction_length
            if len(vehicle_data) < total_length:
                continue
            
            # 计算步长
            step = int(sequence_length * (1 - overlap))
            step = max(1, step)
            
            # 提取序列
            for start_idx in range(0, len(vehicle_data) - total_length + 1, step):
                end_idx = start_idx + total_length
                
                sequence_data = vehicle_data.iloc[start_idx:end_idx].copy()
                
                # 分割历史和未来
                history = sequence_data.iloc[:sequence_length]
                future = sequence_data.iloc[sequence_length:]
                
                # 提取特征
                sequence_info = {
                    'vehicle_id': vehicle_id,
                    'start_frame': history.iloc[0]['Frame_ID'],
                    'end_frame': future.iloc[-1]['Frame_ID'],
                    
                    # 历史轨迹特征
                    'history_x': history['Local_X'].values,
                    'history_y': history['Local_Y'].values,
                    'history_velocity': history['v_Vel'].values,
                    'history_acceleration': history['v_Acc'].values,
                    'history_heading': history['Heading'].values,
                    'history_angular_velocity': history['Angular_Velocity'].values,
                    
                    # 未来轨迹（标签）
                    'future_x': future['Local_X'].values,
                    'future_y': future['Local_Y'].values,
                    
                    # 左转意图标签
                    'left_turn_intent': 1.0,  # 所有序列都是左转
                    
                    # 其他特征
                    'lane_id': history.iloc[0]['Lane_ID'] if 'Lane_ID' in history.columns else 0,
                    'vehicle_class': history.iloc[0]['v_Class'] if 'v_Class' in history.columns else 1,
                }
                
                sequences.append(sequence_info)
        
        print(f"提取了 {len(sequences)} 个轨迹序列")
        return sequences
    
    def save_processed_data(self, output_dir='processed_data'):
        """保存处理后的数据"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存完整的处理后数据
        if self.processed_data is not None:
            self.processed_data.to_csv(
                os.path.join(output_dir, 'processed_ngsim_data.csv'), 
                index=False
            )
            print(f"处理后数据已保存到: {output_dir}/processed_ngsim_data.csv")
        
        # 保存左转车辆数据
        if self.left_turn_data is not None:
            self.left_turn_data.to_csv(
                os.path.join(output_dir, 'left_turn_vehicles.csv'), 
                index=False
            )
            print(f"左转车辆数据已保存到: {output_dir}/left_turn_vehicles.csv")
        
        # 保存轨迹序列
        sequences = self.extract_trajectory_sequences()
        if sequences:
            import pickle
            with open(os.path.join(output_dir, 'trajectory_sequences.pkl'), 'wb') as f:
                pickle.dump(sequences, f)
            print(f"轨迹序列已保存到: {output_dir}/trajectory_sequences.pkl")
    
    def visualize_data(self, output_dir='visualizations'):
        """数据可视化"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. 轨迹分布图
        self._plot_trajectory_distribution(output_dir)
        
        # 2. 左转车辆轨迹
        self._plot_left_turn_trajectories(output_dir)
        
        # 3. 特征分布
        self._plot_feature_distributions(output_dir)
        
        # 4. 统计信息
        self._plot_statistics(output_dir)
    
    def _plot_trajectory_distribution(self, output_dir):
        """绘制轨迹分布图"""
        if self.processed_data is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # 随机选择一些车辆进行可视化
        sample_vehicles = np.random.choice(
            self.processed_data['Vehicle_ID'].unique(), 
            min(50, len(self.processed_data['Vehicle_ID'].unique())), 
            replace=False
        )
        
        for vehicle_id in sample_vehicles:
            vehicle_data = self.processed_data[self.processed_data['Vehicle_ID'] == vehicle_id]
            plt.plot(vehicle_data['Local_X'], vehicle_data['Local_Y'], 
                    alpha=0.6, linewidth=1)
        
        plt.xlabel('Local X (m)')
        plt.ylabel('Local Y (m)')
        plt.title('车辆轨迹分布')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trajectory_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_left_turn_trajectories(self, output_dir):
        """绘制左转车辆轨迹"""
        if self.left_turn_data is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # 绘制所有左转轨迹
        for vehicle_id in self.left_turn_data['Vehicle_ID'].unique():
            vehicle_data = self.left_turn_data[self.left_turn_data['Vehicle_ID'] == vehicle_id]
            plt.plot(vehicle_data['Local_X'], vehicle_data['Local_Y'], 
                    alpha=0.7, linewidth=2, label=f'Vehicle {vehicle_id}')
        
        plt.xlabel('Local X (m)')
        plt.ylabel('Local Y (m)')
        plt.title('左转车辆轨迹')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 如果车辆太多，不显示图例
        if len(self.left_turn_data['Vehicle_ID'].unique()) <= 10:
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'left_turn_trajectories.png'), dpi=300)
        plt.close()
    
    def _plot_feature_distributions(self, output_dir):
        """绘制特征分布"""
        if self.processed_data is None:
            return
        
        # 选择要可视化的特征
        features = ['v_Vel', 'v_Acc', 'Heading', 'Angular_Velocity']
        available_features = [f for f in features if f in self.processed_data.columns]
        
        if not available_features:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:4]):
            if i < len(axes):
                # 所有车辆的特征分布
                axes[i].hist(self.processed_data[feature].dropna(), 
                           bins=50, alpha=0.7, label='All Vehicles', density=True)
                
                # 左转车辆的特征分布
                if self.left_turn_data is not None:
                    axes[i].hist(self.left_turn_data[feature].dropna(), 
                               bins=50, alpha=0.7, label='Left Turn Vehicles', density=True)
                
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Density')
                axes[i].set_title(f'{feature} Distribution')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300)
        plt.close()
    
    def _plot_statistics(self, output_dir):
        """绘制统计信息"""
        if self.processed_data is None:
            return
        
        # 创建统计报告
        stats = {
            'Total Vehicles': len(self.processed_data['Vehicle_ID'].unique()),
            'Total Trajectory Points': len(self.processed_data),
            'Left Turn Vehicles': len(self.left_turn_data['Vehicle_ID'].unique()) if self.left_turn_data is not None else 0,
            'Left Turn Trajectory Points': len(self.left_turn_data) if self.left_turn_data is not None else 0,
        }
        
        # 保存统计信息到文件
        with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
            f.write("NGSIM数据处理统计报告\n")
            f.write("=" * 40 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print("统计信息已保存到: statistics.txt")

def main():
    """主函数"""
    print("NGSIM数据处理脚本")
    print("专门用于提取左转车辆轨迹数据")
    print("=" * 50)
    
    # 数据文件路径（需要根据实际情况修改）
    data_path = input("请输入NGSIM数据文件路径 (默认: transportation_data_complete.csv): ").strip()
    if not data_path:
        data_path = "transportation_data_complete.csv"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 文件 {data_path} 不存在")
        print("请确保数据文件在当前目录中")
        return
    
    # 创建数据处理器
    processor = NGSIMDataProcessor(data_path)
    
    # 加载数据
    if not processor.load_data():
        return
    
    # 数据预处理
    if not processor.preprocess_data():
        return
    
    # 识别左转车辆
    if not processor.identify_left_turn_vehicles():
        return
    
    # 保存处理后的数据
    processor.save_processed_data()
    
    # 数据可视化
    print("生成数据可视化...")
    processor.visualize_data()
    
    print("\n" + "=" * 50)
    print("数据处理完成！")
    print("输出文件:")
    print("  - processed_data/processed_ngsim_data.csv")
    print("  - processed_data/left_turn_vehicles.csv")
    print("  - processed_data/trajectory_sequences.pkl")
    print("  - visualizations/")
    print("=" * 50)

if __name__ == "__main__":
    main()