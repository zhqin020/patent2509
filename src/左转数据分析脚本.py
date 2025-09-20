#!/usr/bin/env python3
"""
左转车辆数据筛选和轨迹分析脚本
用于详细分析左转车辆的特征和轨迹，输出可视化结果
集成精确机动分类与空间约束，支持多路口参数配置
解决问题3：如何区分NGSIM数据中，在一个路口的一个入口的行驶记录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import seaborn as sns  # 注释掉可选依赖
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class LeftTurnAnalyzer:
    """左转车辆分析器 - 集成精确机动分类与多路口支持"""
    
    def __init__(self, data_path: str, intersection_id: Optional[int] = None, entrance_direction: Optional[int] = None, entrance_lane: Optional[int] = None):
        """
        初始化分析器
        
        Args:
            data_path: NGSIM数据文件路径
            intersection_id: 路口ID (1, 2, 3, 4等)，None表示稍后选择
            entrance_direction: 入口方向 (1=东向, 2=北向, 3=西向, 4=南向)
            entrance_lane: 入口车道ID
        """
        self.data_path = data_path
        self.intersection_id = intersection_id
        self.entrance_direction = entrance_direction
        self.entrance_lane = entrance_lane
        self.raw_data = None
        self.left_turn_data = None
        self.sample_vehicles = []
        self.maneuver_classification = {}
        self.entrance_analysis = {}
        self.intersection_data = {}
        self.selected_entrance = None
        self.intersections = {}
        self.selected_intersection = None
        self.selected_entrance_key = None
        self.sample_features = None
        
        # 方向映射
        self.direction_names = {
            1: "东向 (Eastbound)",
            2: "北向 (Northbound)", 
            3: "西向 (Westbound)",
            4: "南向 (Southbound)"
        }
        
        # 机动类型映射
        self.movement_names = {
            1: "直行 (Through)",
            2: "左转 (Left Turn)",
            3: "右转 (Right Turn)",
            4: "掉头 (U-Turn)"
        }
        
        # 根据路口ID设置参数
        self.setup_intersection_parameters()
    
    def setup_intersection_parameters(self):
        """根据路口ID设置分析参数"""
        if self.intersection_id is None:
            # 如果还没有选择路口，使用默认参数
            self.params = {
                'min_trajectory_length': 18,
                'angle_threshold': 28,
                'curvature_threshold': 0.009,
                'speed_threshold': 1.8,
                'position_change_threshold': 9.0,
                'direction_consistency_threshold': 0.65,
                'spatial_constraint_enabled': True,
                'left_turn_region': {
                    'x_min': -55, 'x_max': 55,
                    'y_min': -55, 'y_max': 55
                }
            }
        elif self.intersection_id == 1:
            # 路口1: 优化参数，基于测试结果调整
            self.params = {
                'min_trajectory_length': 15,      # 最小轨迹长度
                'angle_threshold': 25,            # 角度变化阈值 (度)
                'curvature_threshold': 0.008,     # 曲率阈值
                'speed_threshold': 2.0,           # 最小速度阈值 (m/s)
                'position_change_threshold': 8.0,  # 位置变化阈值 (m)
                'direction_consistency_threshold': 0.7,  # 方向一致性阈值
                'spatial_constraint_enabled': True,      # 启用空间约束
                'left_turn_region': {
                    'x_min': -50, 'x_max': 50,
                    'y_min': -50, 'y_max': 50
                }
            }
        elif self.intersection_id == 2:
            # 路口2: 标准参数
            self.params = {
                'min_trajectory_length': 20,
                'angle_threshold': 30,
                'curvature_threshold': 0.01,
                'speed_threshold': 1.5,
                'position_change_threshold': 10.0,
                'direction_consistency_threshold': 0.6,
                'spatial_constraint_enabled': True,
                'left_turn_region': {
                    'x_min': -60, 'x_max': 60,
                    'y_min': -60, 'y_max': 60
                }
            }
        else:
            # 默认参数
            self.params = {
                'min_trajectory_length': 18,
                'angle_threshold': 28,
                'curvature_threshold': 0.009,
                'speed_threshold': 1.8,
                'position_change_threshold': 9.0,
                'direction_consistency_threshold': 0.65,
                'spatial_constraint_enabled': True,
                'left_turn_region': {
                    'x_min': -55, 'x_max': 55,
                    'y_min': -55, 'y_max': 55
                }
            }
    
    def print_parameters(self):
        """打印当前使用的参数配置"""
        print(f"\n📋 路口 {self.intersection_id} 分析参数配置:")
        print("="*50)
        for key, value in self.params.items():
            if key != 'left_turn_region':
                print(f"  {key}: {value}")
        print(f"  left_turn_region: {self.params['left_turn_region']}")
        print("="*50)
    
    def load_data(self) -> bool:
        """加载并预处理数据"""
        try:
            if self.intersection_id is not None:
                print(f"正在加载路口 {self.intersection_id} 的数据: {self.data_path}")
            else:
                print(f"正在加载数据: {self.data_path}")
            
            # 读取数据
            self.raw_data = pd.read_csv(self.data_path)
            
            # 过滤指定路口的数据（如果已选择路口）
            if self.intersection_id is not None and 'int_id' in self.raw_data.columns:
                intersection_data = self.raw_data[self.raw_data['int_id'] == self.intersection_id]
                if len(intersection_data) == 0:
                    print(f"⚠️ 警告: 路口 {self.intersection_id} 没有数据")
                    return False
                self.raw_data = intersection_data
                print(f"✅ 已过滤路口 {self.intersection_id} 相关车辆数据: {len(self.raw_data)}/{len(pd.read_csv(self.data_path))} 条记录")
            else:
                print(f"✅ 已加载完整数据: {len(self.raw_data)} 条记录")
            
            # 进一步过滤入口方向和车道（如果指定）
            if self.entrance_direction is not None:
                self.raw_data = self.raw_data[self.raw_data['direction'] == self.entrance_direction]
                print(f"✅ 已过滤入口方向 {self.entrance_direction}: {self.direction_names.get(self.entrance_direction, '未知')}")
            
            if self.entrance_lane is not None:
                self.raw_data = self.raw_data[self.raw_data['lane_id'] == self.entrance_lane]
                print(f"✅ 已过滤入口车道 {self.entrance_lane}")
            
            print(f"✅ 已过滤路口 {self.intersection_id} 相关车辆数据: {len(self.raw_data)}/{len(pd.read_csv(self.data_path))} 条记录")
            print(f"包含 {self.raw_data['vehicle_id'].nunique()} 辆车辆")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def discover_intersections(self):
        """发现数据中的所有路口"""
        try:
            if self.raw_data is None:
                # 临时加载数据来发现路口
                temp_data = pd.read_csv(self.data_path)
            else:
                temp_data = pd.read_csv(self.data_path)  # 重新读取完整数据
            
            if 'int_id' not in temp_data.columns:
                print("⚠️ 数据中没有 'int_id' 列，无法识别路口")
                return {}
            
            # 统计每个路口的基本信息
            intersections = {}
            for int_id in temp_data['int_id'].unique():
                int_data = temp_data[temp_data['int_id'] == int_id]
                intersections[int_id] = {
                    'total_records': len(int_data),
                    'total_vehicles': int_data['vehicle_id'].nunique(),
                    'directions': sorted(int_data['direction'].unique()) if 'direction' in int_data.columns else [],
                    'movements': sorted(int_data['movement'].unique()) if 'movement' in int_data.columns else []
                }
            
            self.intersections = intersections
            return intersections
            
        except Exception as e:
            print(f"❌ 路口发现失败: {e}")
            return {}
    
    def analyze_all_intersections(self):
        """分析所有路口并显示统计信息"""
        intersections = self.discover_intersections()
        
        if not intersections:
            print("❌ 没有发现任何路口数据")
            return False
        
        print(f"\n🔍 发现 {len(intersections)} 个路口:")
        print("="*80)
        print(f"{'路口ID':<8} {'总记录数':<12} {'车辆数':<10} {'方向':<20} {'机动类型':<15}")
        print("-"*80)
        
        for int_id, info in sorted(intersections.items()):
            # 转换方向为名称
            direction_names = []
            for direction in info['directions'][:4]:  # 最多显示4个方向
                if direction in self.direction_names:
                    direction_names.append(f"{direction}({self.direction_names[direction].split(' ')[0]})")
                else:
                    direction_names.append(str(direction))
            directions_str = ','.join(direction_names)
            
            # 转换机动类型为名称
            movement_names = []
            for movement in info['movements'][:4]:  # 最多显示4个机动类型
                if movement in self.movement_names:
                    movement_names.append(f"{movement}({self.movement_names[movement].split(' ')[0]})")
                else:
                    movement_names.append(str(movement))
            movements_str = ','.join(movement_names)
            
            print(f"{int_id:<8} {info['total_records']:<12} {info['total_vehicles']:<10} {directions_str:<20} {movements_str:<15}")
        
        print("="*80)
        return True
    
    def select_intersection_interactive(self):
        """交互式选择路口"""
        if not self.intersections:
            if not self.analyze_all_intersections():
                return False
        
        print(f"\n🎯 请选择要分析的路口:")
        print("="*40)
        
        for int_id, info in sorted(self.intersections.items()):
            print(f"{int_id}. 路口{int_id} - 车辆: {info['total_vehicles']} 辆")
        
        try:
            selected_id = int(input(f"请输入路口ID (默认: {self.intersection_id}): ").strip() or str(self.intersection_id))
            
            if selected_id not in self.intersections:
                print(f"⚠️ 路口 {selected_id} 不存在，使用默认路口 {self.intersection_id}")
                selected_id = self.intersection_id
            
            self.intersection_id = selected_id
            self.selected_intersection = self.intersections[selected_id]
            
            # 重新设置参数
            self.setup_intersection_parameters()
            
            print(f"✅ 选择路口: {selected_id}")
            return True
            
        except ValueError:
            print(f"⚠️ 输入无效，使用默认路口 {self.intersection_id}")
            self.selected_intersection = self.intersections.get(self.intersection_id, {})
            return True
    
    def analyze_intersection_entrances(self):
        """分析路口的入口方向统计"""
        if self.raw_data is None:
            print("❌ 请先加载数据")
            return None
        
        print(f"\n🔍 分析路口 {self.intersection_id} 的入口情况...")
        
        # 按方向分组统计
        entrance_stats = {}
        
        if 'direction' not in self.raw_data.columns:
            print("⚠️ 数据中没有 'direction' 列，无法分析入口")
            return None
        
        direction_groups = self.raw_data.groupby('direction')
        
        for direction, group_data in direction_groups:
            # 统计该方向的车辆
            total_vehicles = group_data['vehicle_id'].nunique()
            
            # 统计左转车辆 (movement = 2)
            left_turn_vehicles = 0
            if 'movement' in group_data.columns:
                left_turn_data = group_data[group_data['movement'] == 2]
                left_turn_vehicles = left_turn_data['vehicle_id'].nunique()
            
            # 计算左转比例
            left_turn_ratio = (left_turn_vehicles / total_vehicles * 100) if total_vehicles > 0 else 0
            
            entrance_key = f"方向{direction}"
            entrance_stats[entrance_key] = {
                'direction': direction,
                'direction_name': self.direction_names.get(direction, f"方向{direction}"),
                'total_vehicles': total_vehicles,
                'left_turn_vehicles': left_turn_vehicles,
                'left_turn_ratio': left_turn_ratio,
                'total_records': len(group_data)
            }
        
        # 显示统计结果
        print("="*70)
        print(f"路口 {self.intersection_id} 入口分析结果（按方向分组）")
        print("="*70)
        print(f"{'入口编号':<10} {'入口方向':<25} {'总车辆':<10} {'左转车辆':<10} {'左转比例':<10}")
        print("-"*70)
        
        for i, (key, stats) in enumerate(sorted(entrance_stats.items()), 1):
            print(f"{i:<10} {stats['direction_name']:<25} {stats['total_vehicles']:<10} {stats['left_turn_vehicles']:<10} {stats['left_turn_ratio']:.1f}%")
        
        print("="*70)
        print(f"总计: {len(entrance_stats)} 个入口方向")
        
        self.entrance_analysis = entrance_stats
        return entrance_stats
    
    def select_entrance_for_analysis(self):
        """用户选择特定入口进行分析"""
        if not self.entrance_analysis:
            print("❌ 请先进行入口分析")
            return False
        
        print(f"\n🎯 请选择要详细分析的入口方向:")
        print("="*50)
        
        entrance_list = list(self.entrance_analysis.items())
        
        for i, (key, stats) in enumerate(entrance_list, 1):
            print(f"{i}. {stats['direction_name']} - 左转车辆: {stats['left_turn_vehicles']} 辆 ({stats['left_turn_ratio']:.1f}%)")
        
        print("0. 分析所有入口方向的左转车辆")
        
        try:
            choice = int(input(f"请输入入口编号 (0-{len(entrance_list)}): ").strip())
            
            if choice == 0:
                print("✅ 选择分析所有入口方向")
                self.selected_entrance = None
                self.selected_entrance_key = None
                return True
            elif 1 <= choice <= len(entrance_list):
                selected_key, selected_stats = entrance_list[choice - 1]
                self.selected_entrance = selected_stats
                self.selected_entrance_key = selected_key
                print(f"✅ 选择入口方向: {selected_stats['direction_name']}")
                print(f"   该入口方向有 {selected_stats['left_turn_vehicles']} 辆左转车辆")
                return True
            else:
                print("⚠️ 选择无效，将分析所有入口方向")
                self.selected_entrance = None
                self.selected_entrance_key = None
                return True
                
        except ValueError:
            print("⚠️ 输入无效，将分析所有入口方向")
            self.selected_entrance = None
            self.selected_entrance_key = None
            return True
    
    def filter_entrance_data(self):
        """根据路口和方向筛选数据"""
        if self.raw_data is None:
            print("❌ 请先加载数据")
            return False
        
        original_count = len(self.raw_data)
        original_vehicles = self.raw_data['vehicle_id'].nunique()
        
        if self.selected_entrance is not None:
            # 筛选特定入口方向的数据
            direction = self.selected_entrance['direction']
            print(f"\n🔍 筛选入口数据: {self.selected_entrance['direction_name']}")
            
            # 按方向筛选
            self.raw_data = self.raw_data[self.raw_data['direction'] == direction]
            
            # 进一步筛选左转车辆 (movement = 2)
            if 'movement' in self.raw_data.columns:
                left_turn_data = self.raw_data[self.raw_data['movement'] == 2]
                if len(left_turn_data) > 0:
                    self.raw_data = left_turn_data
                    print(f"✅ 已筛选入口左转数据:")
                    print(f"   左转车辆数: {self.raw_data['vehicle_id'].nunique()}")
                    print(f"   轨迹记录数: {len(self.raw_data):,}/{original_count:,}")
                else:
                    print(f"⚠️ 该入口方向没有左转车辆数据")
                    return False
            else:
                print(f"⚠️ 数据中没有 'movement' 列，无法筛选左转车辆")
        else:
            # 分析所有入口方向的左转车辆
            print(f"\n🔍 筛选所有入口方向的左转车辆")
            if 'movement' in self.raw_data.columns:
                left_turn_data = self.raw_data[self.raw_data['movement'] == 2]
                if len(left_turn_data) > 0:
                    self.raw_data = left_turn_data
                    print(f"✅ 已筛选所有入口左转数据:")
                    print(f"   左转车辆数: {self.raw_data['vehicle_id'].nunique()}")
                    print(f"   轨迹记录数: {len(self.raw_data):,}/{original_count:,}")
                else:
                    print(f"⚠️ 没有找到左转车辆数据")
                    return False
            else:
                print(f"⚠️ 数据中没有 'movement' 列，无法筛选左转车辆")
        
        return True
    
    def calculate_trajectory_angle_change(self, trajectory_data: pd.DataFrame) -> float:
        """计算轨迹的角度变化"""
        if len(trajectory_data) < 3:
            return 0.0
        
        # 按时间排序
        traj = trajectory_data.sort_values('frame_id')
        
        angles = []
        for i in range(1, len(traj) - 1):
            # 计算前后两段的方向向量
            x1, y1 = traj.iloc[i-1][['local_x', 'local_y']]
            x2, y2 = traj.iloc[i][['local_x', 'local_y']]
            x3, y3 = traj.iloc[i+1][['local_x', 'local_y']]
            
            # 向量1: (x1,y1) -> (x2,y2)
            v1 = np.array([x2-x1, y2-y1])
            # 向量2: (x2,y2) -> (x3,y3)
            v2 = np.array([x3-x2, y3-y2])
            
            # 计算角度
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)  # 防止数值误差
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        
        return np.sum(angles) if angles else 0.0
    
    def calculate_trajectory_curvature(self, trajectory_data: pd.DataFrame) -> float:
        """计算轨迹的平均曲率"""
        if len(trajectory_data) < 3:
            return 0.0
        
        # 按时间排序
        traj = trajectory_data.sort_values('frame_id')
        
        curvatures = []
        for i in range(1, len(traj) - 1):
            # 获取三个连续点
            x1, y1 = traj.iloc[i-1][['local_x', 'local_y']]
            x2, y2 = traj.iloc[i][['local_x', 'local_y']]
            x3, y3 = traj.iloc[i+1][['local_x', 'local_y']]
            
            # 计算曲率 k = |det(v1, v2)| / |v1|^3
            # 其中 v1 = (x2-x1, y2-y1), v2 = (x3-x2, y3-y2)
            v1_x, v1_y = x2-x1, y2-y1
            v2_x, v2_y = x3-x2, y3-y2
            
            # 行列式
            det = v1_x * v2_y - v1_y * v2_x
            
            # v1的模长
            v1_norm = np.sqrt(v1_x**2 + v1_y**2)
            
            if v1_norm > 0:
                curvature = abs(det) / (v1_norm**3)
                curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def classify_vehicle_maneuver(self, vehicle_data: pd.DataFrame) -> str:
        """
        精确分类车辆机动类型
        
        Args:
            vehicle_data: 单个车辆的轨迹数据
            
        Returns:
            机动类型: 'left_turn', 'right_turn', 'through', 'u_turn', 'insufficient_data'
        """
        if len(vehicle_data) < self.params['min_trajectory_length']:
            return 'insufficient_data'
        
        # 按时间排序
        traj = vehicle_data.sort_values('frame_id').reset_index(drop=True)
        
        # 1. 基于movement字段的直接判断（如果可用且可靠）
        if 'movement' in traj.columns:
            movements = traj['movement'].unique()
            if len(movements) == 1:
                movement = movements[0]
                if movement == 2:
                    return 'left_turn'
                elif movement == 3:
                    return 'right_turn'
                elif movement == 1:
                    return 'through'
                elif movement == 4:
                    return 'u_turn'
        
        # 2. 基于轨迹几何特征的分析
        
        # 计算总角度变化
        total_angle_change = self.calculate_trajectory_angle_change(traj)
        
        # 计算平均曲率
        avg_curvature = self.calculate_trajectory_curvature(traj)
        
        # 计算位置变化
        start_pos = np.array([traj.iloc[0]['local_x'], traj.iloc[0]['local_y']])
        end_pos = np.array([traj.iloc[-1]['local_x'], traj.iloc[-1]['local_y']])
        position_change = np.linalg.norm(end_pos - start_pos)
        
        # 计算方向变化
        if len(traj) >= 2:
            initial_direction = np.arctan2(
                traj.iloc[1]['local_y'] - traj.iloc[0]['local_y'],
                traj.iloc[1]['local_x'] - traj.iloc[0]['local_x']
            )
            final_direction = np.arctan2(
                traj.iloc[-1]['local_y'] - traj.iloc[-2]['local_y'],
                traj.iloc[-1]['local_x'] - traj.iloc[-2]['local_x']
            )
            direction_change = abs(final_direction - initial_direction) * 180 / np.pi
            if direction_change > 180:
                direction_change = 360 - direction_change
        else:
            direction_change = 0
        
        # 3. 分类逻辑
        
        # 左转判断
        if (total_angle_change > self.params['angle_threshold'] and 
            avg_curvature > self.params['curvature_threshold'] and
            position_change > self.params['position_change_threshold'] and
            direction_change > 45):
            
            # 进一步判断是左转还是右转
            # 通过轨迹的弯曲方向判断
            mid_point = len(traj) // 2
            if mid_point > 0 and mid_point < len(traj) - 1:
                # 计算轨迹中点的弯曲方向
                x1, y1 = traj.iloc[0][['local_x', 'local_y']]
                x2, y2 = traj.iloc[mid_point][['local_x', 'local_y']]
                x3, y3 = traj.iloc[-1][['local_x', 'local_y']]
                
                # 使用叉积判断弯曲方向
                cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                
                if cross_product > 0:
                    return 'left_turn'
                else:
                    return 'right_turn'
            
            return 'left_turn'  # 默认左转
        
        # 掉头判断
        elif direction_change > 135 and avg_curvature > self.params['curvature_threshold'] * 2:
            return 'u_turn'
        
        # 右转判断
        elif (total_angle_change > self.params['angle_threshold'] * 0.6 and
              avg_curvature > self.params['curvature_threshold'] * 0.5 and
              30 < direction_change < 120):
            return 'right_turn'
        
        # 直行判断
        elif (total_angle_change < self.params['angle_threshold'] * 0.5 and
              avg_curvature < self.params['curvature_threshold'] * 0.5 and
              direction_change < 30):
            return 'through'
        
        # 默认情况
        else:
            return 'insufficient_data'
    
    def identify_left_turn_vehicles(self) -> bool:
        """识别左转车辆"""
        if self.raw_data is None:
            print("❌ 请先加载数据")
            return False
        
        print(f"\n🔍 开始识别左转车辆...")
        
        left_turn_vehicles = []
        vehicle_classifications = {}
        
        # 获取所有车辆ID
        vehicle_ids = self.raw_data['vehicle_id'].unique()
        
        print(f"正在分析 {len(vehicle_ids)} 辆车辆的机动类型...")
        
        for i, vehicle_id in enumerate(vehicle_ids):
            if i % 50 == 0:  # 每50辆车显示一次进度
                print(f"  进度: {i+1}/{len(vehicle_ids)} ({(i+1)/len(vehicle_ids)*100:.1f}%)")
            
            # 获取车辆轨迹数据
            vehicle_data = self.raw_data[self.raw_data['vehicle_id'] == vehicle_id]
            
            # 分类机动类型
            maneuver_type = self.classify_vehicle_maneuver(vehicle_data)
            vehicle_classifications[vehicle_id] = maneuver_type
            
            # 如果是左转车辆，添加到列表
            if maneuver_type == 'left_turn':
                left_turn_vehicles.append(vehicle_id)
        
        # 保存结果
        self.maneuver_classification = vehicle_classifications
        
        # 筛选左转车辆数据
        if left_turn_vehicles:
            self.left_turn_data = self.raw_data[self.raw_data['vehicle_id'].isin(left_turn_vehicles)]
            
            print(f"\n✅ 左转车辆识别完成!")
            print(f"   总车辆数: {len(vehicle_ids)}")
            print(f"   左转车辆数: {len(left_turn_vehicles)}")
            print(f"   左转比例: {len(left_turn_vehicles)/len(vehicle_ids)*100:.1f}%")
            
            # 显示各类机动的统计
            maneuver_counts = {}
            for maneuver in vehicle_classifications.values():
                maneuver_counts[maneuver] = maneuver_counts.get(maneuver, 0) + 1
            
            print(f"\n📊 机动类型统计:")
            for maneuver, count in sorted(maneuver_counts.items()):
                percentage = count / len(vehicle_ids) * 100
                print(f"   {maneuver}: {count} 辆 ({percentage:.1f}%)")
            
            return True
        else:
            print(f"\n⚠️ 未识别到左转车辆")
            print("可能原因:")
            print("1. 数据中确实没有左转车辆")
            print("2. 轨迹长度不足或质量较差")
            print("3. 参数设置需要调整")
            
            # 显示参数建议
            print(f"\n当前参数设置:")
            print(f"  最小轨迹长度: {self.params['min_trajectory_length']}")
            print(f"  角度阈值: {self.params['angle_threshold']}°")
            print(f"  曲率阈值: {self.params['curvature_threshold']}")
            
            return False
    
    def select_sample_vehicles(self, num_samples: int = 5) -> bool:
        """选择样例车辆进行详细分析"""
        if self.left_turn_data is None or len(self.left_turn_data) == 0:
            print("❌ 没有左转车辆数据可供分析")
            return False
        
        # 获取所有左转车辆ID
        left_turn_vehicle_ids = self.left_turn_data['vehicle_id'].unique()
        
        if len(left_turn_vehicle_ids) == 0:
            print("❌ 没有找到左转车辆")
            return False
        
        # 选择样例车辆
        num_samples = min(num_samples, len(left_turn_vehicle_ids))
        
        # 优先选择轨迹较长的车辆
        vehicle_trajectory_lengths = {}
        for vehicle_id in left_turn_vehicle_ids:
            vehicle_data = self.left_turn_data[self.left_turn_data['vehicle_id'] == vehicle_id]
            vehicle_trajectory_lengths[vehicle_id] = len(vehicle_data)
        
        # 按轨迹长度排序，选择前num_samples个
        sorted_vehicles = sorted(vehicle_trajectory_lengths.items(), key=lambda x: x[1], reverse=True)
        self.sample_vehicles = [vehicle_id for vehicle_id, _ in sorted_vehicles[:num_samples]]
        
        print(f"\n✅ 已选择 {len(self.sample_vehicles)} 个样例车辆进行详细分析:")
        for i, vehicle_id in enumerate(self.sample_vehicles, 1):
            trajectory_length = vehicle_trajectory_lengths[vehicle_id]
            print(f"   {i}. 车辆 {vehicle_id}: {trajectory_length} 个轨迹点")
        
        return True
    
    def analyze_sample_features(self, output_dir: str) -> Optional[pd.DataFrame]:
        """分析样例车辆的详细特征"""
        if not self.sample_vehicles:
            print("❌ 没有样例车辆可供分析")
            return None
        
        print(f"\n🔍 分析 {len(self.sample_vehicles)} 个样例车辆的特征...")
        
        features_list = []
        
        for vehicle_id in self.sample_vehicles:
            # 获取车辆轨迹数据
            vehicle_data = self.left_turn_data[self.left_turn_data['vehicle_id'] == vehicle_id].sort_values('frame_id')
            
            if len(vehicle_data) < 2:
                continue
            
            # 基本特征
            trajectory_length = len(vehicle_data)
            duration = vehicle_data['frame_id'].max() - vehicle_data['frame_id'].min()
            
            # 位置特征
            start_x, start_y = vehicle_data.iloc[0][['local_x', 'local_y']]
            end_x, end_y = vehicle_data.iloc[-1][['local_x', 'local_y']]
            total_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # 速度特征
            if 'v_vel' in vehicle_data.columns:
                avg_speed = vehicle_data['v_vel'].mean()
                max_speed = vehicle_data['v_vel'].max()
                min_speed = vehicle_data['v_vel'].min()
                speed_std = vehicle_data['v_vel'].std()
            else:
                avg_speed = max_speed = min_speed = speed_std = 0
            
            # 加速度特征
            if 'v_acc' in vehicle_data.columns:
                avg_acceleration = vehicle_data['v_acc'].mean()
                max_acceleration = vehicle_data['v_acc'].max()
                min_acceleration = vehicle_data['v_acc'].min()
                acc_std = vehicle_data['v_acc'].std()
            else:
                avg_acceleration = max_acceleration = min_acceleration = acc_std = 0
            
            # 几何特征
            total_angle_change = self.calculate_trajectory_angle_change(vehicle_data)
            avg_curvature = self.calculate_trajectory_curvature(vehicle_data)
            
            # 机动分类
            maneuver_type = self.maneuver_classification.get(vehicle_id, 'unknown')
            
            # 入口信息
            if 'direction' in vehicle_data.columns:
                entrance_direction = vehicle_data['direction'].iloc[0]
                entrance_name = self.direction_names.get(entrance_direction, f"方向{entrance_direction}")
            else:
                entrance_direction = 0
                entrance_name = "未知"
            
            features = {
                'vehicle_id': vehicle_id,
                'trajectory_length': trajectory_length,
                'duration_frames': duration,
                'start_x': start_x,
                'start_y': start_y,
                'end_x': end_x,
                'end_y': end_y,
                'total_distance': total_distance,
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'min_speed': min_speed,
                'speed_std': speed_std,
                'avg_acceleration': avg_acceleration,
                'max_acceleration': max_acceleration,
                'min_acceleration': min_acceleration,
                'acceleration_std': acc_std,
                'total_angle_change': total_angle_change,
                'avg_curvature': avg_curvature,
                'maneuver_type': maneuver_type,
                'entrance_direction': entrance_direction,
                'entrance_name': entrance_name
            }
            
            features_list.append(features)
        
        # 创建DataFrame
        features_df = pd.DataFrame(features_list)
        
        # 保存特征数据
        os.makedirs(output_dir, exist_ok=True)
        features_file = os.path.join(output_dir, f'intersection_{self.intersection_id}_left_turn_sample_features.csv')
        features_df.to_csv(features_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ 样例车辆特征分析完成，已保存到: {features_file}")
        
        # 显示统计摘要
        print(f"\n📊 样例车辆特征摘要:")
        print(f"   平均轨迹长度: {features_df['trajectory_length'].mean():.1f} 点")
        print(f"   平均持续时间: {features_df['duration_frames'].mean():.1f} 帧")
        print(f"   平均速度: {features_df['avg_speed'].mean():.2f} m/s")
        print(f"   平均角度变化: {features_df['total_angle_change'].mean():.1f}°")
        print(f"   平均曲率: {features_df['avg_curvature'].mean():.6f}")
        
        return features_df
    
    def visualize_trajectories(self, output_dir: str):
        """可视化样例车辆轨迹"""
        if not self.sample_vehicles:
            print("❌ 没有样例车辆可供可视化")
            return
        
        print(f"\n🎨 生成 {len(self.sample_vehicles)} 个样例车辆的轨迹可视化...")
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 为每个样例车辆绘制轨迹
        for i, vehicle_id in enumerate(self.sample_vehicles):
            if i >= 6:  # 最多显示6个
                break
                
            ax = axes[i]
            
            # 获取车辆轨迹数据
            vehicle_data = self.left_turn_data[self.left_turn_data['vehicle_id'] == vehicle_id].sort_values('frame_id')
            
            if len(vehicle_data) < 2:
                continue
            
            # 绘制轨迹
            x_coords = vehicle_data['local_x'].values
            y_coords = vehicle_data['local_y'].values
            
            # 轨迹线
            ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='轨迹')
            
            # 起点和终点
            ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='起点')
            ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='终点')
            
            # 方向箭头
            if len(x_coords) > 1:
                for j in range(0, len(x_coords)-1, max(1, len(x_coords)//10)):
                    dx = x_coords[j+1] - x_coords[j]
                    dy = y_coords[j+1] - y_coords[j]
                    if dx != 0 or dy != 0:
                        ax.arrow(x_coords[j], y_coords[j], dx*0.3, dy*0.3, 
                                head_width=1, head_length=1, fc='red', ec='red', alpha=0.6)
            
            # 设置标题和标签
            maneuver_type = self.maneuver_classification.get(vehicle_id, 'unknown')
            ax.set_title(f'车辆 {vehicle_id}\n机动类型: {maneuver_type}', fontsize=10)
            ax.set_xlabel('X坐标 (m)')
            ax.set_ylabel('Y坐标 (m)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # 设置坐标轴范围
            margin = 10
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        
        # 隐藏多余的子图
        for i in range(len(self.sample_vehicles), 6):
            axes[i].set_visible(False)
        
        # 设置总标题
        entrance_info = ""
        if self.selected_entrance:
            entrance_info = f" - {self.selected_entrance['direction_name']}"
        
        plt.suptitle(f'路口 {self.intersection_id} 左转车辆轨迹分析{entrance_info}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图形
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, f'intersection_{self.intersection_id}_left_turn_trajectories_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 轨迹可视化完成，已保存到: {plot_file}")
    
    def generate_detailed_report(self, output_dir: str):
        """生成详细的分析报告"""
        print(f"\n📝 生成详细分析报告...")
        
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, f'intersection_{self.intersection_id}_left_turn_analysis_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"路口 {self.intersection_id} 左转车辆分析报告\n")
            f.write("="*80 + "\n\n")
            
            # 基本信息
            f.write("1. 基本信息\n")
            f.write("-"*40 + "\n")
            f.write(f"数据文件: {self.data_path}\n")
            f.write(f"路口ID: {self.intersection_id}\n")
            
            if self.selected_entrance:
                f.write(f"分析入口: {self.selected_entrance['direction_name']}\n")
                f.write(f"入口方向: {self.selected_entrance['direction']}\n")
            else:
                f.write("分析范围: 所有入口方向\n")
            
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 数据统计
            f.write("2. 数据统计\n")
            f.write("-"*40 + "\n")
            if self.raw_data is not None:
                f.write(f"总记录数: {len(self.raw_data):,}\n")
                f.write(f"总车辆数: {self.raw_data['vehicle_id'].nunique()}\n")
            
            if self.left_turn_data is not None:
                f.write(f"左转记录数: {len(self.left_turn_data):,}\n")
                f.write(f"左转车辆数: {self.left_turn_data['vehicle_id'].nunique()}\n")
            
            f.write(f"样例车辆数: {len(self.sample_vehicles)}\n\n")
            
            # 入口分析结果
            if self.entrance_analysis:
                f.write("3. 入口分析结果\n")
                f.write("-"*40 + "\n")
                for key, stats in self.entrance_analysis.items():
                    f.write(f"{stats['direction_name']}:\n")
                    f.write(f"  总车辆: {stats['total_vehicles']} 辆\n")
                    f.write(f"  左转车辆: {stats['left_turn_vehicles']} 辆\n")
                    f.write(f"  左转比例: {stats['left_turn_ratio']:.1f}%\n")
                    f.write(f"  总记录: {stats['total_records']} 条\n\n")
            
            # 机动分类统计
            if self.maneuver_classification:
                f.write("4. 机动分类统计\n")
                f.write("-"*40 + "\n")
                maneuver_counts = {}
                for maneuver in self.maneuver_classification.values():
                    maneuver_counts[maneuver] = maneuver_counts.get(maneuver, 0) + 1
                
                total_vehicles = len(self.maneuver_classification)
                for maneuver, count in sorted(maneuver_counts.items()):
                    percentage = count / total_vehicles * 100
                    f.write(f"{maneuver}: {count} 辆 ({percentage:.1f}%)\n")
                f.write("\n")
            
            # 参数配置
            f.write("5. 参数配置\n")
            f.write("-"*40 + "\n")
            for key, value in self.params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # 样例车辆详情
            if self.sample_vehicles and self.sample_features is not None:
                f.write("6. 样例车辆详情\n")
                f.write("-"*40 + "\n")
                for _, row in self.sample_features.iterrows():
                    f.write(f"车辆 {row['vehicle_id']}:\n")
                    f.write(f"  轨迹长度: {row['trajectory_length']} 点\n")
                    f.write(f"  持续时间: {row['duration_frames']} 帧\n")
                    f.write(f"  平均速度: {row['avg_speed']:.2f} m/s\n")
                    f.write(f"  角度变化: {row['total_angle_change']:.1f}°\n")
                    f.write(f"  平均曲率: {row['avg_curvature']:.6f}\n")
                    f.write(f"  机动类型: {row['maneuver_type']}\n")
                    f.write(f"  入口方向: {row['entrance_name']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("报告生成完成\n")
            f.write("="*80 + "\n")
        
        print(f"✅ 分析报告生成完成，已保存到: {report_file}")
    
    def export_processed_data(self, output_dir: str):
        """导出处理后的数据供深度学习使用"""
        if self.left_turn_data is None:
            print("❌ 没有左转数据可供导出")
            return
        
        print(f"\n💾 导出处理后的左转数据...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加机动分类信息
        export_data = self.left_turn_data.copy()
        export_data['predicted_maneuver'] = export_data['vehicle_id'].map(self.maneuver_classification)
        
        # 添加入口信息
        if self.selected_entrance:
            export_data['selected_entrance'] = self.selected_entrance['direction_name']
            export_data['selected_entrance_direction'] = self.selected_entrance['direction']
        
        # 添加样例标记
        export_data['is_sample'] = export_data['vehicle_id'].isin(self.sample_vehicles)
        
        # 保存数据
        data_file = os.path.join(output_dir, f'intersection_{self.intersection_id}_processed_left_turn_data.csv')
        export_data.to_csv(data_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ 处理后数据导出完成，已保存到: {data_file}")
        
        # 生成数据质量报告
        quality_report_file = os.path.join(output_dir, f'intersection_{self.intersection_id}_data_quality_report.txt')
        
        with open(quality_report_file, 'w', encoding='utf-8') as f:
            f.write("数据质量报告\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"总记录数: {len(export_data):,}\n")
            f.write(f"车辆数: {export_data['vehicle_id'].nunique()}\n")
            f.write(f"样例车辆数: {export_data['is_sample'].sum()}\n\n")
            
            # 轨迹长度分布
            trajectory_lengths = export_data.groupby('vehicle_id').size()
            f.write("轨迹长度统计:\n")
            f.write(f"  平均长度: {trajectory_lengths.mean():.1f} 点\n")
            f.write(f"  最短轨迹: {trajectory_lengths.min()} 点\n")
            f.write(f"  最长轨迹: {trajectory_lengths.max()} 点\n")
            f.write(f"  标准差: {trajectory_lengths.std():.1f} 点\n\n")
            
            # 速度统计
            if 'v_vel' in export_data.columns:
                f.write("速度统计:\n")
                f.write(f"  平均速度: {export_data['v_vel'].mean():.2f} m/s\n")
                f.write(f"  最大速度: {export_data['v_vel'].max():.2f} m/s\n")
                f.write(f"  最小速度: {export_data['v_vel'].min():.2f} m/s\n")
                f.write(f"  速度标准差: {export_data['v_vel'].std():.2f} m/s\n\n")
            
            # 机动分类分布
            if 'predicted_maneuver' in export_data.columns:
                maneuver_dist = export_data.groupby('vehicle_id')['predicted_maneuver'].first().value_counts()
                f.write("机动分类分布:\n")
                for maneuver, count in maneuver_dist.items():
                    percentage = count / len(maneuver_dist) * 100
                    f.write(f"  {maneuver}: {count} 辆 ({percentage:.1f}%)\n")
        
        print(f"✅ 数据质量报告生成完成，已保存到: {quality_report_file}")
    
    def run_complete_analysis(self, num_samples: int = 5, output_dir: str = None) -> bool:
        """运行完整的左转车辆分析流程"""
        print("🚀 开始完整的左转车辆分析...")
        print("="*60)
        
        if output_dir is None:
            output_dir = f'intersection_{self.intersection_id}_left_turn_analysis'
        
        # 显示参数配置
        self.print_parameters()
        
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
        if self.sample_features is None or self.sample_features.empty:
            return False
        
        # 5. 可视化轨迹
        self.visualize_trajectories(output_dir)
        
        # 6. 生成报告
        self.generate_detailed_report(output_dir)
        
        # 7. 导出处理后的数据
        self.export_processed_data(output_dir)
        
        print(f"\n{'='*60}")
        print(f"🎉 路口 {self.intersection_id} 左转车辆分析完成！")
        print("输出文件:")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_left_turn_sample_features.csv")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_left_turn_trajectories_analysis.png")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_left_turn_analysis_report.txt")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_processed_left_turn_data.csv")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_data_quality_report.txt")
        print("="*60)
        
        return True
    
    def run_entrance_analysis(self, num_samples=5, output_dir=None):
        """运行入口分析流程 - 解决问题3，并继续完整分析"""
        print("🔍 开始入口分析流程...")
        print("="*60)
        
        # 1. 先加载完整数据来发现路口
        if not self.load_data():
            return False
        
        # 2. 发现所有路口
        if not self.analyze_all_intersections():
            return False
        
        # 3. 选择路口
        if not self.select_intersection_interactive():
            return False
        
        # 4. 重新加载选定路口的数据
        if not self.load_data():
            return False
        
        # 5. 分析入口
        entrance_stats = self.analyze_intersection_entrances()
        if not entrance_stats:
            return False
        
        # 6. 选择入口
        if not self.select_entrance_for_analysis():
            return False
        
        # 7. 过滤数据
        if not self.filter_entrance_data():
            return False
        
        print("✅ 入口分析完成！")
        print("已筛选出指定入口的左转车辆数据，现在开始详细分析...")
        
        # 8. 继续进行完整的左转分析流程
        print("\n" + "="*60)
        print("🚀 开始详细的左转车辆轨迹分析...")
        print("="*60)
        
        if output_dir is None:
            if self.selected_entrance:
                entrance_name = self.selected_entrance['direction_name'].replace(' ', '_').replace('(', '').replace(')', '')
                output_dir = f'intersection_{self.intersection_id}_{entrance_name}_left_turn_analysis'
            else:
                output_dir = f'intersection_{self.intersection_id}_all_entrances_left_turn_analysis'
        
        # 显示参数配置
        self.print_parameters()
        
        # 8. 识别左转车辆
        if not self.identify_left_turn_vehicles():
            print("⚠️ 左转车辆识别失败，但继续使用已筛选的数据")
            # 如果识别失败，使用已经筛选的左转车辆数据
            self.left_turn_data = self.raw_data
        
        # 9. 选择样例车辆
        if not self.select_sample_vehicles(num_samples):
            print("⚠️ 样例车辆选择失败")
            return False
        
        # 10. 分析特征
        self.sample_features = self.analyze_sample_features(output_dir)
        if self.sample_features is None or self.sample_features.empty:
            print("⚠️ 特征分析失败")
            return False
        
        # 11. 可视化轨迹
        self.visualize_trajectories(output_dir)
        
        # 12. 生成报告
        self.generate_detailed_report(output_dir)
        
        # 13. 导出处理后的数据
        self.export_processed_data(output_dir)
        
        print(f"\n{'='*60}")
        print(f"🎉 路口 {self.intersection_id} 入口分析和左转车辆分析完成！")
        if self.selected_entrance:
            print(f"分析入口: {self.selected_entrance['direction_name']}")
        else:
            print("分析范围: 所有入口方向")
        print("输出文件:")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_left_turn_sample_features.csv")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_left_turn_trajectories_analysis.png")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_left_turn_analysis_report.txt")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_processed_left_turn_data.csv")
        print(f"  - {output_dir}/intersection_{self.intersection_id}_data_quality_report.txt")
        print("="*60)
        
        return True


def main():
    """主函数"""
    print("🎯 左转车辆数据筛选和轨迹分析脚本 - 支持多路口分析")
    print("="*60)
    
    # 数据文件路径
    data_path = input("请输入NGSIM数据文件路径 (默认: ../data/peachtree_filtered_data.csv): ").strip()
    if not data_path:
        data_path = "../data/peachtree_filtered_data.csv"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件 {data_path} 不存在")
        print("请确保数据文件路径正确")
        return
    
    # 选择分析模式
    print("\n请选择分析模式:")
    print("1 - 入口分析模式 (解决问题3: 区分路口入口)")
    print("2 - 标准左转分析模式")
    
    try:
        mode = int(input("请选择模式 (默认: 1): ").strip() or "1")
    except ValueError:
        mode = 1
    
    if mode == 1:
        # 入口分析模式
        # 样例数量
        try:
            num_samples = int(input("请输入要分析的样例车辆数量 (默认: 5): ").strip() or "5")
        except ValueError:
            num_samples = 5
        
        analyzer = LeftTurnAnalyzer(data_path)
        analyzer.run_entrance_analysis(num_samples=num_samples)
    else:
        # 标准分析模式
        print("\n可选择的路口:")
        print("1 - 路口1 (优化参数: 召回率优先，基于测试结果调整)")
        print("2 - 路口2 (标准参数)")
        print("其他 - 使用默认参数")
        
        try:
            intersection_id = int(input("请选择路口ID (默认: 1): ").strip() or "1")
        except ValueError:
            intersection_id = 1
        
        # 样例数量
        try:
            num_samples = int(input("请输入要分析的样例车辆数量 (默认: 5): ").strip() or "5")
        except ValueError:
            num_samples = 5
        
        # 创建分析器并运行分析
        analyzer = LeftTurnAnalyzer(data_path, intersection_id=intersection_id)
        analyzer.run_complete_analysis(num_samples=num_samples)


if __name__ == "__main__":
    main()