#!/usr/bin/env python3
"""
数据预处理管道 - 为深度学习框架准备高质量左转数据
"""

import os
import sys
import pandas as pd
import numpy as np
from 左转数据分析脚本 import LeftTurnAnalyzer

class DataPreprocessingPipeline:
    """数据预处理管道"""
    
    def __init__(self, raw_data_path, output_dir='processed_data'):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.analyzer = None
        
    def run_preprocessing(self):
        """运行完整的数据预处理流程"""
        print("=" * 60)
        print("数据预处理管道启动")
        print("=" * 60)
        
        # 1. 创建左转分析器
        print("1. 初始化左转分析器...")
        self.analyzer = LeftTurnAnalyzer(self.raw_data_path)
        
        # 2. 加载和分析数据
        print("2. 加载原始数据...")
        if not self.analyzer.load_data():
            print("❌ 数据加载失败")
            return False
        
        # 3. 识别左转车辆
        print("3. 识别左转车辆（应用空间约束和精确分类）...")
        if not self.analyzer.identify_left_turn_vehicles():
            print("❌ 左转识别失败")
            return False
        
        # 4. 导出预处理数据
        print("4. 导出预处理数据...")
        if not self.export_for_deep_learning():
            print("❌ 数据导出失败")
            return False
        
        print("\n" + "=" * 60)
        print("✅ 数据预处理完成！")
        print("=" * 60)
        return True
    
    def export_for_deep_learning(self):
        """导出适合深度学习的数据格式"""
        if self.analyzer.left_turn_data is None or len(self.analyzer.left_turn_data) == 0:
            print("没有左转数据可导出")
            return False
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 导出完整的左转数据
        export_path = os.path.join(self.output_dir, 'processed_left_turn_data.csv')
        
        # 添加质量标记和特征
        export_data = self.analyzer.left_turn_data.copy()
        
        # 为每个车辆添加质量评估和特征
        vehicle_features = {}
        for vehicle_id in export_data['vehicle_id'].unique():
            vehicle_data = export_data[export_data['vehicle_id'] == vehicle_id]
            
            # 计算数据质量指标
            trajectory_length = len(vehicle_data)
            x_coords = vehicle_data['local_x'].values
            y_coords = vehicle_data['local_y'].values
            
            if len(x_coords) < 2:
                continue
            
            # 计算轨迹平滑度
            dx = np.diff(x_coords)
            dy = np.diff(y_coords)
            distances = np.sqrt(dx**2 + dy**2)
            smoothness = np.std(distances) if len(distances) > 0 else 0
            
            # 计算总航向角变化
            total_heading_change = self.analyzer.calculate_total_heading_change(vehicle_data)
            
            # 计算空间跨度
            x_range = x_coords.max() - x_coords.min()
            y_range = y_coords.max() - y_coords.min()
            spatial_span = max(x_range, y_range)
            
            # 计算速度特征
            if len(distances) > 0:
                avg_speed = np.mean(distances) * 10  # 假设10Hz采样率
                max_speed = np.max(distances) * 10
            else:
                avg_speed = max_speed = 0
            
            # 计算加速度特征
            if len(distances) > 1:
                accelerations = np.diff(distances) * 100  # 假设10Hz采样率
                avg_acceleration = np.mean(accelerations)
                max_acceleration = np.max(np.abs(accelerations))
            else:
                avg_acceleration = max_acceleration = 0
            
            # 判断是否为高质量数据
            is_high_quality = (
                trajectory_length >= 50 and 
                smoothness < 10 and 
                60 <= abs(total_heading_change) <= 120 and
                spatial_span < 200 and
                avg_speed > 0.5  # 最小速度阈值
            )
            
            vehicle_features[vehicle_id] = {
                'trajectory_length': trajectory_length,
                'smoothness': smoothness,
                'total_heading_change': abs(total_heading_change),
                'spatial_span': spatial_span,
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'avg_acceleration': avg_acceleration,
                'max_acceleration': max_acceleration,
                'is_high_quality': is_high_quality
            }
        
        # 添加特征到导出数据
        for feature_name in ['trajectory_length', 'smoothness', 'total_heading_change', 
                           'spatial_span', 'avg_speed', 'max_speed', 'avg_acceleration', 
                           'max_acceleration', 'is_high_quality']:
            export_data[feature_name] = export_data['vehicle_id'].map(
                lambda x: vehicle_features.get(x, {}).get(feature_name, 0)
            )
        
        # 添加序列标记（用于深度学习的序列划分）
        export_data = export_data.sort_values(['vehicle_id', 'frame_id'])
        export_data['sequence_id'] = export_data['vehicle_id']
        export_data['time_step'] = export_data.groupby('vehicle_id').cumcount()
        
        # 保存数据
        export_data.to_csv(export_path, index=False)
        
        # 统计信息
        total_vehicles = len(export_data['vehicle_id'].unique())
        high_quality_vehicles = len([v for v in vehicle_features.values() if v['is_high_quality']])
        total_points = len(export_data)
        
        print(f"\n📊 数据导出统计:")
        print(f"   导出文件: {export_path}")
        print(f"   总车辆数: {total_vehicles}")
        print(f"   高质量车辆数: {high_quality_vehicles} ({high_quality_vehicles/total_vehicles*100:.1f}%)")
        print(f"   总数据点: {total_points}")
        
        # 保存质量统计报告
        self.save_quality_report(vehicle_features, export_path)
        
        # 保存训练配置文件
        self.save_training_config(export_path)
        
        return True
    
    def save_quality_report(self, vehicle_features, export_path):
        """保存数据质量报告"""
        quality_report_path = os.path.join(self.output_dir, 'data_quality_report.txt')
        
        total_vehicles = len(vehicle_features)
        high_quality_vehicles = len([v for v in vehicle_features.values() if v['is_high_quality']])
        
        with open(quality_report_path, 'w', encoding='utf-8') as f:
            f.write("左转数据预处理质量报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📈 数据统计:\n")
            f.write(f"   总车辆数: {total_vehicles}\n")
            f.write(f"   高质量车辆数: {high_quality_vehicles}\n")
            f.write(f"   高质量比例: {high_quality_vehicles/total_vehicles*100:.1f}%\n\n")
            
            f.write("✅ 质量标准:\n")
            f.write("   - 轨迹长度 >= 50 个点\n")
            f.write("   - 轨迹平滑度 < 10\n")
            f.write("   - 航向角变化 60°-120°\n")
            f.write("   - 空间跨度 < 200米\n")
            f.write("   - 平均速度 > 0.5 m/s\n\n")
            
            f.write("📋 导出数据列说明:\n")
            f.write("   - vehicle_id: 车辆ID\n")
            f.write("   - frame_id: 帧ID\n")
            f.write("   - local_x, local_y: 车辆坐标\n")
            f.write("   - sequence_id: 序列ID（等于vehicle_id）\n")
            f.write("   - time_step: 时间步（从0开始）\n")
            f.write("   - is_high_quality: 是否为高质量数据\n")
            f.write("   - trajectory_length: 轨迹长度\n")
            f.write("   - smoothness: 轨迹平滑度\n")
            f.write("   - total_heading_change: 总航向角变化\n")
            f.write("   - spatial_span: 空间跨度\n")
            f.write("   - avg_speed, max_speed: 速度特征\n")
            f.write("   - avg_acceleration, max_acceleration: 加速度特征\n\n")
            
            f.write("🎯 使用建议:\n")
            f.write("   1. 优先使用 is_high_quality=True 的数据进行训练\n")
            f.write("   2. 可以根据 trajectory_length 进行序列长度筛选\n")
            f.write("   3. sequence_id 和 time_step 可用于构建深度学习序列\n")
            f.write("   4. 各种特征可用于数据增强和质量控制\n")
        
        print(f"   质量报告: {quality_report_path}")
    
    def save_training_config(self, export_path):
        """保存训练配置文件"""
        config_path = os.path.join(self.output_dir, 'training_config.py')
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\n')
            f.write('深度学习训练配置文件\n')
            f.write('由数据预处理管道自动生成\n')
            f.write('"""\n\n')
            
            f.write('# 数据路径配置\n')
            f.write(f'PROCESSED_DATA_PATH = "{export_path}"\n')
            f.write(f'OUTPUT_DIR = "{self.output_dir}"\n\n')
            
            f.write('# 模型训练参数\n')
            f.write('SEQUENCE_LENGTH = 8  # 历史轨迹长度\n')
            f.write('PREDICTION_LENGTH = 12  # 预测轨迹长度\n')
            f.write('BATCH_SIZE = 32\n')
            f.write('LEARNING_RATE = 0.001\n')
            f.write('NUM_EPOCHS = 100\n')
            f.write('EARLY_STOPPING_PATIENCE = 10\n\n')
            
            f.write('# 数据筛选参数\n')
            f.write('USE_HIGH_QUALITY_ONLY = True  # 是否只使用高质量数据\n')
            f.write('MIN_TRAJECTORY_LENGTH = 50  # 最小轨迹长度\n')
            f.write('MAX_SPATIAL_SPAN = 200  # 最大空间跨度\n\n')
            
            f.write('# 特征配置\n')
            f.write('VISUAL_FEATURE_DIM = 64\n')
            f.write('MOTION_FEATURE_DIM = 40\n')
            f.write('TRAFFIC_FEATURE_DIM = 32\n')
        
        print(f"   训练配置: {config_path}")

def main():
    """主函数"""
    print("数据预处理管道")
    print("将原始NGSIM数据转换为深度学习训练数据")
    
    # 获取输入参数
    raw_data_path = input("请输入原始NGSIM数据路径 (默认: ../data/peachtree_filtered_data.csv): ").strip()
    if not raw_data_path:
        raw_data_path = "../data/peachtree_filtered_data.csv"
    
    output_dir = input("请输入输出目录 (默认: processed_data): ").strip()
    if not output_dir:
        output_dir = "processed_data"
    
    # 检查输入文件
    if not os.path.exists(raw_data_path):
        print(f"❌ 错误: 输入文件不存在 {raw_data_path}")
        return
    
    # 运行预处理管道
    pipeline = DataPreprocessingPipeline(raw_data_path, output_dir)
    success = pipeline.run_preprocessing()
    
    if success:
        print(f"\n🎉 预处理完成！输出目录: {output_dir}")
        print("📁 生成的文件:")
        print(f"   - processed_left_turn_data.csv  (训练数据)")
        print(f"   - data_quality_report.txt       (质量报告)")
        print(f"   - training_config.py            (训练配置)")
        print("\n💡 下一步: 使用 代码实现框架.py 进行模型训练")
    else:
        print("❌ 预处理失败")

if __name__ == "__main__":
    main()