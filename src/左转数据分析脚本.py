#!/usr/bin/env python3
"""
左转车辆数据筛选和轨迹分析脚本
用于详细分析左转车辆的特征和轨迹，输出可视化结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class LeftTurnAnalyzer:
    """左转车辆分析器"""
    
    def __init__(self, data_path: str):
        """
        初始化分析器
        
        Args:
            data_path: NGSIM数据文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.left_turn_data = None
        self.sample_vehicles = []
        
    def load_data(self):
        """加载数据"""
        try:
            print(f"正在加载数据: {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
            print(f"数据加载成功，共 {len(self.raw_data)} 条记录")
            print(f"包含 {len(self.raw_data['vehicle_id'].unique())} 辆车辆")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def identify_left_turn_vehicles(self, heading_threshold=75, min_trajectory_length=100):
        """
        精确的左转车辆识别算法 - 区分左转、掉头、复杂机动
        
        Args:
            heading_threshold: 航向角变化阈值（度，默认75度）
            min_trajectory_length: 最小轨迹长度（默认100个点）
        """
        if self.raw_data is None:
            print("请先加载数据")
            return False
        
        print("正在进行精确的车辆机动分类...")
        
        # 统计各种机动类型
        maneuver_stats = {
            "left_turn": [],
            "right_turn": [],
            "u_turn_or_complex_maneuver": [],
            "straight_or_slight_curve": [],
            "noisy_data": [],
            "stationary_or_minimal_movement": [],
            "complex_trajectory": [],
            "other_maneuver": [],
            "insufficient_data": []
        }
        
        total_vehicles = len(self.raw_data['vehicle_id'].unique())
        processed = 0
        
        for vehicle_id in self.raw_data['vehicle_id'].unique():
            processed += 1
            if processed % 200 == 0:
                print(f"  已处理 {processed}/{total_vehicles} 辆车辆")
            
            vehicle_data = self.raw_data[self.raw_data['vehicle_id'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('frame_id')
            
            if len(vehicle_data) < min_trajectory_length:
                maneuver_stats["insufficient_data"].append(vehicle_id)
                continue
            
            # 精确分类机动类型
            maneuver_type = self.classify_vehicle_maneuver(vehicle_data)
            maneuver_stats[maneuver_type].append(vehicle_id)
        
        # 输出分类统计
        print("=== 车辆机动分类统计 ===")
        for maneuver_type, vehicles in maneuver_stats.items():
            count = len(vehicles)
            percentage = count / total_vehicles * 100
            type_name = {
                "left_turn": "真正左转",
                "right_turn": "右转",
                "u_turn_or_complex_maneuver": "掉头/复杂机动",
                "straight_or_slight_curve": "直行/轻微弯曲",
                "noisy_data": "数据噪声",
                "stationary_or_minimal_movement": "静止/微小移动",
                "complex_trajectory": "复杂轨迹",
                "other_maneuver": "其他机动",
                "insufficient_data": "数据不足"
            }.get(maneuver_type, maneuver_type)
            print(f"{type_name}: {count} 辆 ({percentage:.2f}%)")
        
        # 只提取真正的左转车辆
        left_turn_vehicles = maneuver_stats["left_turn"]
        self.left_turn_data = self.raw_data[self.raw_data['vehicle_id'].isin(left_turn_vehicles)]
        
        print(f"✅ 精确识别出 {len(left_turn_vehicles)} 辆真正的左转车辆")
        print(f"左转车辆占比: {len(left_turn_vehicles)/total_vehicles*100:.2f}%")
        
        # 保存分类统计信息
        self.maneuver_stats = maneuver_stats
        
        return True
    
    def clean_trajectory_data(self, vehicle_data):
        """清洗轨迹数据，移除异常跳跃点"""
        if len(vehicle_data) < 3:
            return vehicle_data
        
        # 计算相邻点之间的距离
        dx = vehicle_data['local_x'].diff().fillna(0)
        dy = vehicle_data['local_y'].diff().fillna(0)
        distances = np.sqrt(dx**2 + dy**2)
        
        # 移除距离异常大的点（可能是数据错误）
        distance_threshold = distances.quantile(0.95) * 3  # 使用95分位数的3倍作为阈值
        valid_indices = distances <= distance_threshold
        valid_indices.iloc[0] = True  # 保留第一个点
        
        return vehicle_data[valid_indices].copy()
    
    def ultra_clean_trajectory(self, vehicle_data):
        """超强轨迹清洗 - 5轮迭代清洗"""
        if len(vehicle_data) < 10:
            return vehicle_data
        
        clean_data = vehicle_data.copy()
        
        # 5轮超强清洗
        for iteration in range(5):
            if len(clean_data) < 10:
                break
            
            # 计算相邻点距离
            dx = clean_data['local_x'].diff().fillna(0)
            dy = clean_data['local_y'].diff().fillna(0)
            distances = np.sqrt(dx**2 + dy**2)
            
            # 使用四分位距方法检测异常
            q25 = distances.quantile(0.25)
            q75 = distances.quantile(0.75)
            iqr = q75 - q25
            
            # 更严格的上界
            upper_bound = q75 + 3 * iqr
            lower_bound = q25 - 1.5 * iqr
            
            # 移除异常点
            valid_mask = (distances >= lower_bound) & (distances <= upper_bound)
            valid_mask.iloc[0] = True  # 保留第一个点
            
            new_clean_data = clean_data[valid_mask].copy()
            
            # 如果没有移除任何点，停止清洗
            if len(new_clean_data) == len(clean_data):
                break
            
            clean_data = new_clean_data
        
        return clean_data
    
    def classify_vehicle_maneuver(self, vehicle_data):
        """精确分类车辆机动类型"""
        if len(vehicle_data) < 20:
            return "insufficient_data"
        
        # 超强清洗
        clean_data = self.ultra_clean_trajectory(vehicle_data)
        
        if len(clean_data) < 20:
            return "insufficient_data"
        
        x_coords = clean_data['local_x'].values
        y_coords = clean_data['local_y'].values
        
        # 基本几何特征
        start_x, start_y = x_coords[0], y_coords[0]
        end_x, end_y = x_coords[-1], y_coords[-1]
        
        straight_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # 计算路径长度
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        
        # 曲率比
        curvature_ratio = path_length / straight_distance if straight_distance > 0 else float('inf')
        
        # 空间跨度
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        max_span = max(x_range, y_range)
        
        # 计算总航向角变化
        total_heading_change = self.calculate_total_heading_change(clean_data)
        
        # 分类逻辑
        if curvature_ratio > 20:  # 路径长度是直线距离的20倍以上
            return "noisy_data"
        
        if max_span < 15:  # 空间跨度小于15米
            return "stationary_or_minimal_movement"
        
        if abs(total_heading_change) > 150:  # 总航向角变化超过150度
            if straight_distance < 50:  # 净位移小于50米
                return "u_turn_or_complex_maneuver"
            else:
                return "complex_trajectory"
        
        if 60 < abs(total_heading_change) < 120:  # 60-120度的转向
            if straight_distance > 30 and curvature_ratio < 5:  # 有明显位移且路径相对平滑
                if total_heading_change > 0:
                    return "left_turn"
                else:
                    return "right_turn"
        
        if abs(total_heading_change) < 30:  # 航向角变化小于30度
            return "straight_or_slight_curve"
        
        return "other_maneuver"
    
    def calculate_total_heading_change(self, vehicle_data):
        """计算总航向角变化"""
        if len(vehicle_data) < 3:
            return 0
        
        x_coords = vehicle_data['local_x'].values
        y_coords = vehicle_data['local_y'].values
        
        # 计算平滑的航向角
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        
        # 使用较大的平滑窗口
        window_size = min(15, len(dx) // 3)
        if window_size >= 3:
            dx_smooth = pd.Series(dx).rolling(window=window_size, center=True).mean().fillna(pd.Series(dx))
            dy_smooth = pd.Series(dy).rolling(window=window_size, center=True).mean().fillna(pd.Series(dy))
        else:
            dx_smooth = pd.Series(dx)
            dy_smooth = pd.Series(dy)
        
        headings = np.degrees(np.arctan2(dy_smooth, dx_smooth))
        
        # 计算累积航向角变化
        heading_diffs = np.diff(headings)
        
        # 处理角度跨越
        heading_diffs = np.where(heading_diffs > 180, heading_diffs - 360, heading_diffs)
        heading_diffs = np.where(heading_diffs < -180, heading_diffs + 360, heading_diffs)
        
        # 返回累积变化
        return np.sum(heading_diffs)
    
    def detect_left_turn_pattern(self, vehicle_data, heading_threshold):
        """
        改进的左转模式检测
        
        Args:
            vehicle_data: 清洗后的车辆轨迹数据
            heading_threshold: 航向角变化阈值
            
        Returns:
            bool: 是否为左转
        """
        if len(vehicle_data) < 10:
            return False
        
        # 进一步清洗数据 - 使用更严格的标准
        clean_data = self.aggressive_clean_trajectory(vehicle_data)
        
        if len(clean_data) < 10:
            return False
        
        # 计算轨迹的几何特征
        geometry_check = self.check_trajectory_geometry(clean_data)
        if not geometry_check:
            return False
        
        # 基于分段分析的左转检测
        segment_analysis = self.analyze_trajectory_segments(clean_data)
        
        # 计算累积转向
        cumulative_turn = self.calculate_smooth_cumulative_turn(clean_data)
        
        # 多重验证条件
        conditions = {
            'max_left_turn': np.max(cumulative_turn) > heading_threshold,
            'final_turn': cumulative_turn.iloc[-1] > heading_threshold * 0.7,
            'consistent_turn': segment_analysis['consistent_left_turn'],
            'geometry_valid': geometry_check,
            'turn_smoothness': np.std(cumulative_turn) > 15
        }
        
        # 至少满足4个条件才认为是左转
        valid_conditions = sum(conditions.values())
        is_left_turn = valid_conditions >= 4
        
        return is_left_turn
    
    def aggressive_clean_trajectory(self, vehicle_data):
        """更激进的轨迹清洗"""
        if len(vehicle_data) < 5:
            return vehicle_data
        
        clean_data = vehicle_data.copy()
        
        # 多轮清洗
        for iteration in range(3):  # 最多3轮清洗
            if len(clean_data) < 5:
                break
                
            # 计算相邻点距离
            dx = clean_data['local_x'].diff().fillna(0)
            dy = clean_data['local_y'].diff().fillna(0)
            distances = np.sqrt(dx**2 + dy**2)
            
            # 使用更严格的阈值
            median_distance = distances.median()
            mad = np.median(np.abs(distances - median_distance))  # 中位数绝对偏差
            threshold = median_distance + 5 * mad  # 更严格的阈值
            
            # 移除异常点
            valid_indices = distances <= threshold
            valid_indices.iloc[0] = True  # 保留第一个点
            
            new_clean_data = clean_data[valid_indices].copy()
            
            # 如果没有移除任何点，停止清洗
            if len(new_clean_data) == len(clean_data):
                break
                
            clean_data = new_clean_data
        
        return clean_data
    
    def check_trajectory_geometry(self, vehicle_data):
        """检查轨迹几何特征的合理性"""
        if len(vehicle_data) < 5:
            return False
        
        x_coords = vehicle_data['local_x'].values
        y_coords = vehicle_data['local_y'].values
        
        # 计算直线距离和路径长度
        straight_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2)
        
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        
        # 曲率比检查
        if straight_distance > 0:
            curvature_ratio = path_length / straight_distance
            # 对于真实的左转，曲率比应该在合理范围内
            if curvature_ratio > 50:  # 如果路径长度是直线距离的50倍以上，可能是噪声
                return False
        
        # 检查轨迹是否有明显的空间变化
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        
        # 轨迹应该有一定的空间跨度
        if max(x_range, y_range) < 10:  # 小于10米的变化可能不是真正的转向
            return False
        
        return True
    
    def analyze_trajectory_segments(self, vehicle_data):
        """分段分析轨迹"""
        if len(vehicle_data) < 20:
            return {'consistent_left_turn': False}
        
        x_coords = vehicle_data['local_x'].values
        y_coords = vehicle_data['local_y'].values
        
        # 将轨迹分成5段
        n_segments = 5
        segment_size = len(x_coords) // n_segments
        
        segment_directions = []
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, len(x_coords))
            
            if end_idx - start_idx < 3:
                continue
            
            # 计算段的主方向
            dx_seg = x_coords[end_idx-1] - x_coords[start_idx]
            dy_seg = y_coords[end_idx-1] - y_coords[start_idx]
            
            if dx_seg == 0 and dy_seg == 0:
                continue
            
            direction = np.degrees(np.arctan2(dy_seg, dx_seg))
            segment_directions.append(direction)
        
        if len(segment_directions) < 3:
            return {'consistent_left_turn': False}
        
        # 计算方向变化
        direction_changes = []
        for i in range(1, len(segment_directions)):
            change = segment_directions[i] - segment_directions[i-1]
            # 标准化角度
            while change > 180:
                change -= 360
            while change < -180:
                change += 360
            direction_changes.append(change)
        
        # 检查是否有一致的左转趋势
        left_turns = [change for change in direction_changes if change > 10]  # 大于10度的左转
        total_left_turn = sum(left_turns)
        
        consistent_left_turn = (len(left_turns) >= len(direction_changes) * 0.6 and 
                               total_left_turn > 30)
        
        return {'consistent_left_turn': consistent_left_turn}
    
    def calculate_smooth_cumulative_turn(self, vehicle_data):
        """计算平滑的累积转向"""
        if len(vehicle_data) < 3:
            return pd.Series([0])
        
        x_coords = vehicle_data['local_x'].values
        y_coords = vehicle_data['local_y'].values
        
        # 计算移动方向
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        
        # 使用更大的平滑窗口
        window_size = min(10, len(dx) // 3)
        if window_size >= 3:
            dx_smooth = pd.Series(dx).rolling(window=window_size, center=True).mean().fillna(pd.Series(dx))
            dy_smooth = pd.Series(dy).rolling(window=window_size, center=True).mean().fillna(pd.Series(dy))
        else:
            dx_smooth = pd.Series(dx)
            dy_smooth = pd.Series(dy)
        
        # 计算航向角
        headings = np.degrees(np.arctan2(dy_smooth, dx_smooth))
        
        # 计算累积转向
        return self.calculate_cumulative_turn(headings)
    
    def calculate_cumulative_turn(self, headings):
        """计算累积转向角度"""
        if len(headings) < 2:
            return pd.Series([0])
        
        # 计算相邻航向角的差异
        heading_diffs = headings.diff().fillna(0)
        
        # 处理角度跨越问题
        heading_diffs = heading_diffs.apply(lambda x: self.normalize_angle(x))
        
        # 计算累积转向（正值表示左转，负值表示右转）
        cumulative_turn = heading_diffs.cumsum()
        
        return cumulative_turn
    
    def normalize_angle(self, angle):
        """标准化角度到[-180, 180]范围"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def calculate_heading_change(self, vehicle_data):
        """
        计算车辆的航向角变化
        
        Args:
            vehicle_data: 单个车辆的轨迹数据
            
        Returns:
            float: 航向角变化（度）
        """
        if len(vehicle_data) < 2:
            return 0
        
        # 基于位置变化计算航向角
        dx = vehicle_data['local_x'].diff().fillna(0)
        dy = vehicle_data['local_y'].diff().fillna(0)
        headings = np.degrees(np.arctan2(dy, dx))
        
        if len(headings) > 1:
            heading_start = headings.iloc[1]  # 跳过第一个NaN值
            heading_end = headings.iloc[-1]
            heading_change = heading_end - heading_start
        else:
            heading_change = 0
        
        # 处理角度跨越问题
        if heading_change > 180:
            heading_change -= 360
        elif heading_change < -180:
            heading_change += 360
            
        return abs(heading_change)
    
    def select_sample_vehicles(self, num_samples=5):
        """选择代表性的左转车辆样例"""
        if self.left_turn_data is None:
            print("请先识别左转车辆")
            return False
        
        left_turn_vehicles = self.left_turn_data['vehicle_id'].unique()
        
        # 计算每辆车的轨迹长度和特征
        vehicle_stats = []
        for vehicle_id in left_turn_vehicles:
            vehicle_data = self.left_turn_data[self.left_turn_data['vehicle_id'] == vehicle_id]
            
            # 计算航向角变化
            if 'local_x' in vehicle_data.columns and 'local_y' in vehicle_data.columns:
                dx = vehicle_data['local_x'].diff().fillna(0)
                dy = vehicle_data['local_y'].diff().fillna(0)
                headings = np.degrees(np.arctan2(dy, dx))
                heading_change = abs(headings.iloc[-1] - headings.iloc[1]) if len(headings) > 1 else 0
            else:
                heading_change = 0
            
            stats = {
                'vehicle_id': vehicle_id,
                'trajectory_length': len(vehicle_data),
                'speed_variance': vehicle_data['v_vel'].var() if 'v_vel' in vehicle_data.columns else 0,
                'heading_change': heading_change
            }
            vehicle_stats.append(stats)
        
        # 按轨迹长度和特征多样性选择样例
        vehicle_stats_df = pd.DataFrame(vehicle_stats)
        vehicle_stats_df = vehicle_stats_df.sort_values(['trajectory_length', 'speed_variance'], ascending=[False, False])
        
        # 选择前num_samples个车辆
        selected_vehicles = vehicle_stats_df.head(num_samples)['vehicle_id'].tolist()
        self.sample_vehicles = selected_vehicles
        
        print(f"选择了 {len(selected_vehicles)} 辆代表性左转车辆进行详细分析")
        return True
    
    def extract_features(self, vehicle_ids):
        """提取车辆特征"""
        if not vehicle_ids:
            return {}
        
        features = {}
        for vehicle_id in vehicle_ids:
            vehicle_data = self.left_turn_data[self.left_turn_data['vehicle_id'] == vehicle_id]
            if len(vehicle_data) == 0:
                continue
                
            # 提取基本特征
            features[vehicle_id] = {
                'trajectory_length': len(vehicle_data),
                'avg_speed': vehicle_data['v_Vel'].mean(),
                'max_speed': vehicle_data['v_Vel'].max(),
                'start_position': (vehicle_data.iloc[0]['local_x'], vehicle_data.iloc[0]['local_y']),
                'end_position': (vehicle_data.iloc[-1]['local_x'], vehicle_data.iloc[-1]['local_y']),
                'heading_change': self.calculate_heading_change(vehicle_data)
            }
        
        return features
    
    def analyze_sample_features(self, output_dir='left_turn_analysis'):
        """分析样例车辆特征"""
        if not self.sample_vehicles:
            print("请先选择样例车辆")
            return None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n=== 左转车辆样例特征分析 ===")
        
        sample_features = []
        
        for i, vehicle_id in enumerate(self.sample_vehicles):
            vehicle_data = self.left_turn_data[self.left_turn_data['vehicle_id'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('frame_id')
            
            # 计算航向角
            if 'local_x' in vehicle_data.columns and 'local_y' in vehicle_data.columns:
                dx = vehicle_data['local_x'].diff().fillna(0)
                dy = vehicle_data['local_y'].diff().fillna(0)
                headings = np.degrees(np.arctan2(dy, dx))
                heading_start = headings.iloc[1] if len(headings) > 1 else 0
                heading_end = headings.iloc[-1] if len(headings) > 1 else 0
                heading_change = heading_end - heading_start
            else:
                heading_start = heading_end = heading_change = 0
            
            # 计算特征
            features = {
                'vehicle_id': vehicle_id,
                'trajectory_length': len(vehicle_data),
                'duration': (vehicle_data['frame_id'].max() - vehicle_data['frame_id'].min()) * 0.1,  # 假设0.1s/frame
                'avg_speed': vehicle_data['v_vel'].mean() if 'v_vel' in vehicle_data.columns else 0,
                'max_speed': vehicle_data['v_vel'].max() if 'v_vel' in vehicle_data.columns else 0,
                'min_speed': vehicle_data['v_vel'].min() if 'v_vel' in vehicle_data.columns else 0,
                'speed_std': vehicle_data['v_vel'].std() if 'v_vel' in vehicle_data.columns else 0,
                'avg_acceleration': vehicle_data['v_acc'].mean() if 'v_acc' in vehicle_data.columns else 0,
                'max_acceleration': vehicle_data['v_acc'].max() if 'v_acc' in vehicle_data.columns else 0,
                'min_acceleration': vehicle_data['v_acc'].min() if 'v_acc' in vehicle_data.columns else 0,
                'acc_std': vehicle_data['v_acc'].std() if 'v_acc' in vehicle_data.columns else 0,
                'heading_start': heading_start,
                'heading_end': heading_end,
                'heading_change': heading_change,
                'start_x': vehicle_data['local_x'].iloc[0] if 'local_x' in vehicle_data.columns else 0,
                'start_y': vehicle_data['local_y'].iloc[0] if 'local_y' in vehicle_data.columns else 0,
                'end_x': vehicle_data['local_x'].iloc[-1] if 'local_x' in vehicle_data.columns else 0,
                'end_y': vehicle_data['local_y'].iloc[-1] if 'local_y' in vehicle_data.columns else 0,
                'total_distance': np.sqrt((vehicle_data['local_x'].iloc[-1] - vehicle_data['local_x'].iloc[0])**2 + 
                                        (vehicle_data['local_y'].iloc[-1] - vehicle_data['local_y'].iloc[0])**2) if 'local_x' in vehicle_data.columns and 'local_y' in vehicle_data.columns else 0,
                'path_length': np.sum(np.sqrt(np.diff(vehicle_data['local_x'])**2 + np.diff(vehicle_data['local_y'])**2)) if 'local_x' in vehicle_data.columns and 'local_y' in vehicle_data.columns else 0
            }
            
            # 处理航向角跨越问题
            if features['heading_change'] > 180:
                features['heading_change'] -= 360
            elif features['heading_change'] < -180:
                features['heading_change'] += 360
            
            sample_features.append(features)
            
            # 输出详细特征信息
            print(f"\n--- 车辆 {vehicle_id} 特征分析 ---")
            print(f"轨迹长度: {features['trajectory_length']} 个时间步")
            print(f"持续时间: {features['duration']:.1f} 秒")
            print(f"速度统计: 平均 {features['avg_speed']:.2f} m/s, 范围 [{features['min_speed']:.2f}, {features['max_speed']:.2f}] m/s")
            print(f"速度标准差: {features['speed_std']:.2f} m/s")
            print(f"加速度统计: 平均 {features['avg_acceleration']:.2f} m/s², 范围 [{features['min_acceleration']:.2f}, {features['max_acceleration']:.2f}] m/s²")
            print(f"加速度标准差: {features['acc_std']:.2f} m/s²")
            print(f"航向角变化: {features['heading_start']:.1f}° → {features['heading_end']:.1f}° (变化 {features['heading_change']:.1f}°)")
            print(f"起点坐标: ({features['start_x']:.1f}, {features['start_y']:.1f}) m")
            print(f"终点坐标: ({features['end_x']:.1f}, {features['end_y']:.1f}) m")
            print(f"直线距离: {features['total_distance']:.2f} m")
            print(f"路径长度: {features['path_length']:.2f} m")
            print(f"路径曲率: {features['path_length']/features['total_distance']:.2f}")
        
        # 保存特征数据
        features_df = pd.DataFrame(sample_features)
        features_df.to_csv(os.path.join(output_dir, 'left_turn_sample_features.csv'), index=False)
        print(f"\n特征数据已保存到: {output_dir}/left_turn_sample_features.csv")
        
        return sample_features
    
    def visualize_trajectories(self, output_dir='left_turn_analysis'):
        """可视化左转车辆轨迹"""
        if not self.sample_vehicles:
            print("请先选择样例车辆")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建综合分析图
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, vehicle_id in enumerate(self.sample_vehicles):
            vehicle_data = self.left_turn_data[self.left_turn_data['vehicle_id'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('frame_id')
            
            color = colors[i % len(colors)]
            time_steps = range(len(vehicle_data))
            
            # 1. 轨迹图 (左上)
            axes[0, 0].plot(vehicle_data['local_x'], vehicle_data['local_y'], 
                           color=color, linewidth=3, alpha=0.8, 
                           label=f'车辆 {vehicle_id}')
            # 标记起点和终点
            axes[0, 0].scatter(vehicle_data['local_x'].iloc[0], vehicle_data['local_y'].iloc[0], 
                              color=color, s=150, marker='o', edgecolor='black', linewidth=2, zorder=5)
            axes[0, 0].scatter(vehicle_data['local_x'].iloc[-1], vehicle_data['local_y'].iloc[-1], 
                              color=color, s=150, marker='s', edgecolor='black', linewidth=2, zorder=5)
            
            # 2. 速度变化 (右上)
            axes[0, 1].plot(time_steps, vehicle_data['v_vel'], 
                           color=color, linewidth=2, alpha=0.8, 
                           label=f'车辆 {vehicle_id}')
            
            # 3. 航向角变化 (中上) - 基于位置计算
            if 'local_x' in vehicle_data.columns and 'local_y' in vehicle_data.columns:
                dx = vehicle_data['local_x'].diff().fillna(0)
                dy = vehicle_data['local_y'].diff().fillna(0)
                headings = np.degrees(np.arctan2(dy, dx))
                axes[0, 2].plot(time_steps, headings, 
                               color=color, linewidth=2, alpha=0.8, 
                               label=f'车辆 {vehicle_id}')
            
            # 4. 加速度变化 (左下)
            axes[1, 0].plot(time_steps, vehicle_data['v_acc'], 
                           color=color, linewidth=2, alpha=0.8, 
                           label=f'车辆 {vehicle_id}')
            
            # 5. X-Y坐标随时间变化 (右下)
            axes[1, 1].plot(time_steps, vehicle_data['local_x'], 
                           color=color, linewidth=2, alpha=0.8, linestyle='-',
                           label=f'车辆 {vehicle_id} X')
            axes[1, 1].plot(time_steps, vehicle_data['local_y'], 
                           color=color, linewidth=2, alpha=0.6, linestyle='--')
        
        # 设置图表标题和标签
        axes[0, 0].set_title('左转车辆轨迹 (○起点 ■终点)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('X坐标 (m)')
        axes[0, 0].set_ylabel('Y坐标 (m)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].axis('equal')
        
        axes[0, 1].set_title('速度变化曲线', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('速度 (m/s)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[0, 2].set_title('航向角变化曲线', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('时间步')
        axes[0, 2].set_ylabel('航向角 (度)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        axes[1, 0].set_title('加速度变化曲线', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('加速度 (m/s²)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].set_title('坐标随时间变化 (实线X, 虚线Y)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('坐标 (m)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # 6. 特征统计对比 (中下)
        if hasattr(self, 'sample_features') and self.sample_features:
            feature_names = ['平均速度\n(m/s)', '最大速度\n(m/s)', '平均加速度\n(m/s²)', '航向角变化\n(度)', '轨迹长度\n(步数)']
            avg_values = [
                np.mean([f['avg_speed'] for f in self.sample_features]),
                np.mean([f['max_speed'] for f in self.sample_features]),
                np.mean([f['avg_acceleration'] for f in self.sample_features]),
                np.mean([abs(f['heading_change']) for f in self.sample_features]),
                np.mean([f['trajectory_length'] for f in self.sample_features])
            ]
            
            bars = axes[1, 2].bar(feature_names, avg_values, 
                                 color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
            axes[1, 2].set_title('左转车辆特征统计', fontsize=14, fontweight='bold')
            axes[1, 2].set_ylabel('数值')
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, avg_values):
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_values)*0.01, 
                               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'left_turn_trajectories_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"轨迹可视化图表已保存到: {output_dir}/left_turn_trajectories_analysis.png")
    
    def generate_detailed_report(self, output_dir='left_turn_analysis'):
        """生成详细分析报告"""
        if not self.sample_features:
            print("请先进行特征分析")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report_path = os.path.join(output_dir, 'left_turn_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("左转车辆轨迹预测 - 数据筛选与特征分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. 数据概览\n")
            f.write("-" * 30 + "\n")
            f.write(f"数据文件: {self.data_path}\n")
            f.write(f"总车辆数: {len(self.raw_data['vehicle_id'].unique())}\n")
            f.write(f"左转车辆数: {len(self.left_turn_data['vehicle_id'].unique())}\n")
            f.write(f"左转车辆占比: {len(self.left_turn_data['vehicle_id'].unique())/len(self.raw_data['vehicle_id'].unique())*100:.2f}%\n")
            f.write(f"分析样例数: {len(self.sample_vehicles)}\n\n")
            
            f.write("2. 样例车辆详细特征\n")
            f.write("-" * 30 + "\n")
            
            for i, features in enumerate(self.sample_features):
                f.write(f"\n车辆 {features['vehicle_id']}:\n")
                f.write(f"  轨迹长度: {features['trajectory_length']} 个时间步\n")
                f.write(f"  持续时间: {features['duration']:.1f} 秒\n")
                f.write(f"  速度统计: 平均 {features['avg_speed']:.2f} m/s (标准差 {features['speed_std']:.2f})\n")
                f.write(f"  速度范围: [{features['min_speed']:.2f}, {features['max_speed']:.2f}] m/s\n")
                f.write(f"  加速度统计: 平均 {features['avg_acceleration']:.2f} m/s² (标准差 {features['acc_std']:.2f})\n")
                f.write(f"  加速度范围: [{features['min_acceleration']:.2f}, {features['max_acceleration']:.2f}] m/s²\n")
                f.write(f"  航向角变化: {features['heading_start']:.1f}° → {features['heading_end']:.1f}° (变化 {features['heading_change']:.1f}°)\n")
                f.write(f"  起点坐标: ({features['start_x']:.1f}, {features['start_y']:.1f}) m\n")
                f.write(f"  终点坐标: ({features['end_x']:.1f}, {features['end_y']:.1f}) m\n")
                f.write(f"  直线距离: {features['total_distance']:.2f} m\n")
                f.write(f"  路径长度: {features['path_length']:.2f} m\n")
                f.write(f"  路径曲率: {features['path_length']/features['total_distance']:.2f}\n")
            
            f.write("\n3. 统计汇总\n")
            f.write("-" * 30 + "\n")
            
            # 计算统计指标
            avg_speed = np.mean([f['avg_speed'] for f in self.sample_features])
            avg_acc = np.mean([f['avg_acceleration'] for f in self.sample_features])
            avg_heading_change = np.mean([abs(f['heading_change']) for f in self.sample_features])
            avg_trajectory_length = np.mean([f['trajectory_length'] for f in self.sample_features])
            avg_duration = np.mean([f['duration'] for f in self.sample_features])
            
            f.write(f"平均速度: {avg_speed:.2f} m/s\n")
            f.write(f"平均加速度: {avg_acc:.2f} m/s²\n")
            f.write(f"平均航向角变化: {avg_heading_change:.2f}°\n")
            f.write(f"平均轨迹长度: {avg_trajectory_length:.1f} 个时间步\n")
            f.write(f"平均持续时间: {avg_duration:.1f} 秒\n")
            
            f.write("\n4. 输出文件\n")
            f.write("-" * 30 + "\n")
            f.write(f"特征数据: {output_dir}/left_turn_sample_features.csv\n")
            f.write(f"轨迹可视化: {output_dir}/left_turn_trajectories_analysis.png\n")
            f.write(f"分析报告: {output_dir}/left_turn_analysis_report.txt\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 60 + "\n")
        
        print(f"详细分析报告已保存到: {report_path}")
    
    def run_complete_analysis(self, num_samples=5, output_dir='left_turn_analysis'):
        """运行完整的左转数据分析流程"""
        print("开始左转车辆数据筛选和轨迹分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 识别左转车辆
        if not self.identify_left_turn_vehicles():
            return False
        
        # 3. 选择样例车辆
        if not self.select_sample_vehicles(num_samples):
            return False
        
        # 4. 分析特征
        self.sample_features = self.analyze_sample_features(output_dir)
        if not self.sample_features:
            return False
        
        # 5. 可视化轨迹
        self.visualize_trajectories(output_dir)
        
        # 6. 生成报告
        self.generate_detailed_report(output_dir)
        
        print(f"\n{'='*50}")
        print("左转车辆分析完成！")
        print("输出文件:")
        print(f"  - {output_dir}/left_turn_sample_features.csv")
        print(f"  - {output_dir}/left_turn_trajectories_analysis.png")
        print(f"  - {output_dir}/left_turn_analysis_report.txt")
        print("="*50)
        
        return True


def main():
    """主函数"""
    # 数据文件路径
    data_path = input("请输入NGSIM数据文件路径 (默认: ../data/peachtree_filtered_data.csv): ").strip()
    if not data_path:
        data_path = "../data/peachtree_filtered_data.csv"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 文件 {data_path} 不存在")
        print("请确保数据文件路径正确")
        return
    
    # 样例数量
    try:
        num_samples = int(input("请输入要分析的样例车辆数量 (默认: 5): ").strip() or "5")
    except ValueError:
        num_samples = 5
    
    # 创建分析器并运行分析
    analyzer = LeftTurnAnalyzer(data_path)
    analyzer.run_complete_analysis(num_samples=num_samples)


if __name__ == "__main__":
    main()