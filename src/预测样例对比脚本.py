#!/usr/bin/env python3
"""
车辆左转轨迹预测样例对比脚本
用于展示预测数据和实际数据的对比分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class PredictionComparator:
    """预测结果对比分析器"""
    
    def __init__(self, data_path: str):
        """
        初始化对比分析器
        
        Args:
            data_path: NGSIM数据文件路径
        """
        self.data_path = data_path
        self.test_data = None
        self.sample_vehicles = []
        self.prediction_results = []
        
    def load_data(self):
        """加载数据（别名方法）"""
        return self.load_test_data()
    
    def load_test_data(self):
        """加载测试数据"""
        try:
            print(f"正在加载测试数据: {self.data_path}")
            self.test_data = pd.read_csv(self.data_path)
            print(f"测试数据加载成功，共 {len(self.test_data)} 条记录")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def select_prediction_samples(self, num_samples=5):
        """选择用于预测对比的样例车辆"""
        if self.test_data is None:
            print("请先加载测试数据")
            return False
        
        # 筛选左转车辆
        left_turn_vehicles = []
        for vehicle_id in self.test_data['vehicle_id'].unique():
            vehicle_data = self.test_data[self.test_data['vehicle_id'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('frame_id')
            
            if len(vehicle_data) < 20:  # 需要足够长的轨迹进行预测
                continue
            
            # 简单的左转判断（基于位置变化计算航向角）
            if 'local_x' in vehicle_data.columns and 'local_y' in vehicle_data.columns:
                dx = vehicle_data['local_x'].diff().fillna(0)
                dy = vehicle_data['local_y'].diff().fillna(0)
                headings = np.degrees(np.arctan2(dy, dx))
                
                if len(headings) > 1:
                    heading_start = headings.iloc[1]  # 跳过第一个NaN值
                    heading_end = headings.iloc[-1]
                    heading_change = heading_end - heading_start
                    
                    # 处理角度跨越问题
                    if heading_change > 180:
                        heading_change -= 360
                    elif heading_change < -180:
                        heading_change += 360
                    
                    if heading_change > 30:  # 左转阈值
                        left_turn_vehicles.append(vehicle_id)
        
        # 选择样例
        selected_vehicles = np.random.choice(left_turn_vehicles, 
                                           min(num_samples, len(left_turn_vehicles)), 
                                           replace=False)
        self.sample_vehicles = selected_vehicles.tolist()
        
        print(f"选择了 {len(self.sample_vehicles)} 辆左转车辆进行预测对比分析")
        return True
    
    def generate_mock_predictions(self):
        """生成模拟预测结果（用于演示）"""
        print("生成模拟预测结果...")
        
        self.prediction_results = []
        
        for vehicle_id in self.sample_vehicles:
            vehicle_data = self.test_data[self.test_data['vehicle_id'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('frame_id')
            
            # 分割历史和未来轨迹
            split_point = len(vehicle_data) // 2
            historical_data = vehicle_data.iloc[:split_point]
            future_actual = vehicle_data.iloc[split_point:]
            
            # 生成模拟预测（基于历史数据的趋势）
            last_x = historical_data['local_x'].iloc[-1] if 'local_x' in historical_data.columns else 0
            last_y = historical_data['local_y'].iloc[-1] if 'local_y' in historical_data.columns else 0
            
            # 计算最后的速度分量
            if 'v_vel' in historical_data.columns and 'local_x' in historical_data.columns and 'local_y' in historical_data.columns:
                # 基于位置变化计算航向角
                dx = historical_data['local_x'].diff().fillna(0)
                dy = historical_data['local_y'].diff().fillna(0)
                last_heading = np.arctan2(dy.iloc[-1], dx.iloc[-1])
                
                last_vx = historical_data['v_vel'].iloc[-1] * np.cos(last_heading)
                last_vy = historical_data['v_vel'].iloc[-1] * np.sin(last_heading)
            else:
                last_vx = last_vy = 1.0  # 默认速度
            
            # 预测未来轨迹点
            predicted_trajectory = []
            current_x, current_y = last_x, last_y
            current_vx, current_vy = last_vx, last_vy
            
            for i in range(len(future_actual)):
                # 添加一些随机性和左转趋势
                turn_factor = 0.1 * i  # 逐渐增加的左转趋势
                noise_x = np.random.normal(0, 0.5)
                noise_y = np.random.normal(0, 0.5)
                
                # 模拟左转运动
                current_x += current_vx * 0.1 + noise_x
                current_y += current_vy * 0.1 + turn_factor + noise_y
                
                # 更新速度（模拟左转时的速度变化）
                current_vx *= 0.98  # 轻微减速
                current_vy += turn_factor * 0.1  # 增加Y方向速度分量
                
                predicted_trajectory.append({
                    'frame': future_actual['frame_id'].iloc[i] if i < len(future_actual) else future_actual['frame_id'].iloc[-1] + i + 1,
                    'pred_x': current_x,
                    'pred_y': current_y,
                    'actual_x': future_actual['local_x'].iloc[i] if i < len(future_actual) else np.nan,
                    'actual_y': future_actual['local_y'].iloc[i] if i < len(future_actual) else np.nan,
                    'pred_speed': np.sqrt(current_vx**2 + current_vy**2),
                    'actual_speed': future_actual['v_vel'].iloc[i] if i < len(future_actual) else np.nan
                })
            
            result = {
                'vehicle_id': vehicle_id,
                'historical_data': historical_data,
                'future_actual': future_actual,
                'predicted_trajectory': predicted_trajectory,
                'left_turn_intent_pred': np.random.uniform(0.7, 0.95),  # 模拟左转意图预测概率
                'left_turn_intent_actual': 1.0  # 实际为左转
            }
            
            self.prediction_results.append(result)
        
        print(f"已生成 {len(self.prediction_results)} 个预测样例")
        return True
    
    def calculate_prediction_metrics(self):
        """计算预测性能指标"""
        print("计算预测性能指标...")
        
        metrics_summary = {
            'intent_accuracy': [],
            'ade_errors': [],  # Average Displacement Error
            'fde_errors': [],  # Final Displacement Error
            'speed_mae': []    # Speed Mean Absolute Error
        }
        
        for result in self.prediction_results:
            # 意图分类准确性
            intent_pred = result['left_turn_intent_pred'] > 0.5
            intent_actual = result['left_turn_intent_actual'] > 0.5
            metrics_summary['intent_accuracy'].append(int(intent_pred == intent_actual))
            
            # 轨迹预测误差
            trajectory = result['predicted_trajectory']
            valid_points = [p for p in trajectory if not np.isnan(p['actual_x'])]
            
            if valid_points:
                # ADE: 平均位移误差
                displacement_errors = []
                speed_errors = []
                
                for point in valid_points:
                    disp_error = np.sqrt((point['pred_x'] - point['actual_x'])**2 + 
                                       (point['pred_y'] - point['actual_y'])**2)
                    displacement_errors.append(disp_error)
                    
                    if not np.isnan(point['actual_speed']):
                        speed_error = abs(point['pred_speed'] - point['actual_speed'])
                        speed_errors.append(speed_error)
                
                if displacement_errors:
                    metrics_summary['ade_errors'].append(np.mean(displacement_errors))
                    metrics_summary['fde_errors'].append(displacement_errors[-1])  # 最后一个点的误差
                
                if speed_errors:
                    metrics_summary['speed_mae'].append(np.mean(speed_errors))
        
        # 计算总体指标
        overall_metrics = {
            'intent_accuracy': np.mean(metrics_summary['intent_accuracy']) * 100,
            'average_ade': np.mean(metrics_summary['ade_errors']),
            'average_fde': np.mean(metrics_summary['fde_errors']),
            'average_speed_mae': np.mean(metrics_summary['speed_mae'])
        }
        
        print(f"预测性能指标:")
        print(f"  左转意图识别准确率: {overall_metrics['intent_accuracy']:.1f}%")
        print(f"  平均位移误差 (ADE): {overall_metrics['average_ade']:.2f} m")
        print(f"  最终位移误差 (FDE): {overall_metrics['average_fde']:.2f} m")
        print(f"  速度预测平均绝对误差: {overall_metrics['average_speed_mae']:.2f} m/s")
        
        return overall_metrics
    
    def visualize_prediction_comparison(self, output_dir='prediction_comparison'):
        """可视化预测对比结果"""
        if not self.prediction_results:
            print("请先生成预测结果")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 为每个样例创建详细对比图
        for i, result in enumerate(self.prediction_results):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            vehicle_id = result['vehicle_id']
            historical = result['historical_data']
            future_actual = result['future_actual']
            predicted = result['predicted_trajectory']
            
            # 1. 轨迹对比图 (左上)
            axes[0, 0].plot(historical['local_x'], historical['local_y'], 
                           'b-', linewidth=3, alpha=0.8, label='历史轨迹')
            axes[0, 0].plot(future_actual['local_x'], future_actual['local_y'], 
                           'g-', linewidth=3, alpha=0.8, label='实际未来轨迹')
            
            pred_x = [p['pred_x'] for p in predicted if not np.isnan(p['pred_x'])]
            pred_y = [p['pred_y'] for p in predicted if not np.isnan(p['pred_y'])]
            axes[0, 0].plot(pred_x, pred_y, 
                           'r--', linewidth=3, alpha=0.8, label='预测未来轨迹')
            
            # 标记关键点
            axes[0, 0].scatter(historical['local_x'].iloc[0], historical['local_y'].iloc[0], 
                              s=150, c='blue', marker='o', edgecolor='black', linewidth=2, 
                              label='起点', zorder=5)
            axes[0, 0].scatter(historical['local_x'].iloc[-1], historical['local_y'].iloc[-1], 
                              s=150, c='orange', marker='s', edgecolor='black', linewidth=2, 
                              label='预测起始点', zorder=5)
            axes[0, 0].scatter(future_actual['local_x'].iloc[-1], future_actual['local_y'].iloc[-1], 
                              s=150, c='green', marker='^', edgecolor='black', linewidth=2, 
                              label='实际终点', zorder=5)
            if pred_x and pred_y:
                axes[0, 0].scatter(pred_x[-1], pred_y[-1], 
                                  s=150, c='red', marker='v', edgecolor='black', linewidth=2, 
                                  label='预测终点', zorder=5)
            
            axes[0, 0].set_title(f'车辆 {vehicle_id} - 轨迹预测对比', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('X坐标 (m)')
            axes[0, 0].set_ylabel('Y坐标 (m)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            axes[0, 0].axis('equal')
            
            # 2. 速度对比 (右上)
            future_frames = range(len(future_actual))
            pred_speeds = [p['pred_speed'] for p in predicted if not np.isnan(p['pred_speed'])]
            
            axes[0, 1].plot(future_frames, future_actual['v_vel'], 
                           'g-', linewidth=3, alpha=0.8, label='实际速度')
            if pred_speeds:
                axes[0, 1].plot(range(len(pred_speeds)), pred_speeds, 
                               'r--', linewidth=3, alpha=0.8, label='预测速度')
            
            axes[0, 1].set_title(f'车辆 {vehicle_id} - 速度预测对比', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('时间步')
            axes[0, 1].set_ylabel('速度 (m/s)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # 3. 位移误差分析 (左下)
            valid_points = [p for p in predicted if not np.isnan(p['actual_x'])]
            if valid_points:
                displacement_errors = []
                for point in valid_points:
                    error = np.sqrt((point['pred_x'] - point['actual_x'])**2 + 
                                  (point['pred_y'] - point['actual_y'])**2)
                    displacement_errors.append(error)
                
                axes[1, 0].plot(range(len(displacement_errors)), displacement_errors, 
                               'purple', linewidth=3, alpha=0.8, marker='o')
                axes[1, 0].axhline(y=np.mean(displacement_errors), color='red', linestyle='--', 
                                  alpha=0.7, label=f'平均误差: {np.mean(displacement_errors):.2f}m')
                
                axes[1, 0].set_title(f'车辆 {vehicle_id} - 位移误差变化', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('预测时间步')
                axes[1, 0].set_ylabel('位移误差 (m)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # 4. 预测置信度和意图分类 (右下)
            intent_info = [
                ['实际左转意图', result['left_turn_intent_actual'], 'green'],
                ['预测左转概率', result['left_turn_intent_pred'], 'red']
            ]
            
            labels = [info[0] for info in intent_info]
            values = [info[1] for info in intent_info]
            colors = [info[2] for info in intent_info]
            
            bars = axes[1, 1].bar(labels, values, color=colors, alpha=0.7)
            axes[1, 1].set_title(f'车辆 {vehicle_id} - 左转意图预测', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('概率/置信度')
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 添加预测结果文本
            prediction_correct = (result['left_turn_intent_pred'] > 0.5) == (result['left_turn_intent_actual'] > 0.5)
            result_text = "预测正确" if prediction_correct else "预测错误"
            result_color = "green" if prediction_correct else "red"
            axes[1, 1].text(0.5, 0.9, result_text, transform=axes[1, 1].transAxes, 
                           ha='center', va='center', fontsize=12, fontweight='bold', 
                           bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.3))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'prediction_comparison_vehicle_{vehicle_id}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"预测对比可视化图表已保存到: {output_dir}/")
    
    def generate_summary_comparison(self, output_dir='prediction_comparison'):
        """生成预测对比汇总图表"""
        if not self.prediction_results:
            print("请先生成预测结果")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建汇总对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 所有车辆轨迹对比 (左上)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, result in enumerate(self.prediction_results):
            color = colors[i % len(colors)]
            vehicle_id = result['vehicle_id']
            historical = result['historical_data']
            future_actual = result['future_actual']
            predicted = result['predicted_trajectory']
            
            # 历史轨迹
            axes[0, 0].plot(historical['local_x'], historical['local_y'], 
                           color=color, linewidth=2, alpha=0.6, linestyle='-')
            # 实际未来轨迹
            axes[0, 0].plot(future_actual['local_x'], future_actual['local_y'], 
                           color=color, linewidth=2, alpha=0.8, linestyle='-', 
                           label=f'车辆 {vehicle_id} 实际')
            # 预测轨迹
            pred_x = [p['pred_x'] for p in predicted if not np.isnan(p['pred_x'])]
            pred_y = [p['pred_y'] for p in predicted if not np.isnan(p['pred_y'])]
            if pred_x and pred_y:
                axes[0, 0].plot(pred_x, pred_y, 
                               color=color, linewidth=2, alpha=0.8, linestyle='--', 
                               label=f'车辆 {vehicle_id} 预测')
        
        axes[0, 0].set_title('所有样例车辆轨迹对比 (实线:实际, 虚线:预测)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('X坐标 (m)')
        axes[0, 0].set_ylabel('Y坐标 (m)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. 预测误差分布 (右上)
        all_ade_errors = []
        all_fde_errors = []
        
        for result in self.prediction_results:
            predicted = result['predicted_trajectory']
            valid_points = [p for p in predicted if not np.isnan(p['actual_x'])]
            
            if valid_points:
                displacement_errors = []
                for point in valid_points:
                    error = np.sqrt((point['pred_x'] - point['actual_x'])**2 + 
                                  (point['pred_y'] - point['actual_y'])**2)
                    displacement_errors.append(error)
                
                if displacement_errors:
                    all_ade_errors.append(np.mean(displacement_errors))
                    all_fde_errors.append(displacement_errors[-1])
        
        if all_ade_errors and all_fde_errors:
            axes[0, 1].hist(all_ade_errors, bins=10, alpha=0.7, label='ADE (平均位移误差)', color='skyblue')
            axes[0, 1].hist(all_fde_errors, bins=10, alpha=0.7, label='FDE (最终位移误差)', color='lightcoral')
            axes[0, 1].axvline(np.mean(all_ade_errors), color='blue', linestyle='--', 
                              label=f'平均ADE: {np.mean(all_ade_errors):.2f}m')
            axes[0, 1].axvline(np.mean(all_fde_errors), color='red', linestyle='--', 
                              label=f'平均FDE: {np.mean(all_fde_errors):.2f}m')
        
        axes[0, 1].set_title('预测误差分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('误差 (m)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. 意图分类结果 (左下)
        intent_results = []
        for result in self.prediction_results:
            intent_pred = result['left_turn_intent_pred'] > 0.5
            intent_actual = result['left_turn_intent_actual'] > 0.5
            intent_results.append('正确' if intent_pred == intent_actual else '错误')
        
        intent_counts = pd.Series(intent_results).value_counts()
        colors_pie = ['lightgreen', 'lightcoral']
        
        axes[1, 0].pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%', 
                      colors=colors_pie, startangle=90)
        axes[1, 0].set_title('左转意图分类准确性', fontsize=14, fontweight='bold')
        
        # 4. 性能指标汇总 (右下)
        metrics = self.calculate_prediction_metrics()
        
        metric_names = ['意图识别\n准确率(%)', 'ADE\n(m)', 'FDE\n(m)', '速度MAE\n(m/s)']
        metric_values = [
            metrics['intent_accuracy'],
            metrics['average_ade'],
            metrics['average_fde'],
            metrics['average_speed_mae']
        ]
        
        bars = axes[1, 1].bar(metric_names, metric_values, 
                             color=['lightgreen', 'skyblue', 'lightcoral', 'gold'])
        axes[1, 1].set_title('预测性能指标汇总', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('数值')
        
        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01, 
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_summary_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"预测对比汇总图表已保存到: {output_dir}/prediction_summary_comparison.png")
    
    def generate_comparison_report(self, output_dir='prediction_comparison'):
        """生成预测对比分析报告"""
        if not self.prediction_results:
            print("请先生成预测结果")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        metrics = self.calculate_prediction_metrics()
        report_path = os.path.join(output_dir, 'prediction_comparison_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("车辆左转轨迹预测 - 预测结果对比分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. 测试概览\n")
            f.write("-" * 30 + "\n")
            f.write(f"测试数据文件: {self.data_path}\n")
            f.write(f"测试样例数量: {len(self.prediction_results)}\n")
            f.write(f"样例车辆ID: {[r['vehicle_id'] for r in self.prediction_results]}\n\n")
            
            f.write("2. 整体性能指标\n")
            f.write("-" * 30 + "\n")
            f.write(f"左转意图识别准确率: {metrics['intent_accuracy']:.1f}%\n")
            f.write(f"平均位移误差 (ADE): {metrics['average_ade']:.2f} m\n")
            f.write(f"最终位移误差 (FDE): {metrics['average_fde']:.2f} m\n")
            f.write(f"速度预测平均绝对误差: {metrics['average_speed_mae']:.2f} m/s\n\n")
            
            f.write("3. 各样例详细结果\n")
            f.write("-" * 30 + "\n")
            
            for i, result in enumerate(self.prediction_results):
                vehicle_id = result['vehicle_id']
                f.write(f"\n样例 {i+1} - 车辆 {vehicle_id}:\n")
                
                # 意图分类结果
                intent_pred = result['left_turn_intent_pred']
                intent_actual = result['left_turn_intent_actual']
                intent_correct = (intent_pred > 0.5) == (intent_actual > 0.5)
                
                f.write(f"  左转意图预测: {intent_pred:.3f} (实际: {intent_actual:.3f})\n")
                f.write(f"  意图分类结果: {'正确' if intent_correct else '错误'}\n")
                
                # 轨迹预测结果
                predicted = result['predicted_trajectory']
                valid_points = [p for p in predicted if not np.isnan(p['actual_x'])]
                
                if valid_points:
                    displacement_errors = []
                    for point in valid_points:
                        error = np.sqrt((point['pred_x'] - point['actual_x'])**2 + 
                                      (point['pred_y'] - point['actual_y'])**2)
                        displacement_errors.append(error)
                    
                    if displacement_errors:
                        f.write(f"  轨迹预测点数: {len(displacement_errors)}\n")
                        f.write(f"  平均位移误差: {np.mean(displacement_errors):.2f} m\n")
                        f.write(f"  最终位移误差: {displacement_errors[-1]:.2f} m\n")
                        f.write(f"  最大位移误差: {max(displacement_errors):.2f} m\n")
                        f.write(f"  最小位移误差: {min(displacement_errors):.2f} m\n")
            
            f.write("\n4. 性能评估\n")
            f.write("-" * 30 + "\n")
            
            # 性能等级评估
            if metrics['intent_accuracy'] >= 90:
                intent_grade = "优秀"
            elif metrics['intent_accuracy'] >= 80:
                intent_grade = "良好"
            elif metrics['intent_accuracy'] >= 70:
                intent_grade = "一般"
            else:
                intent_grade = "需要改进"
            
            if metrics['average_ade'] <= 1.0:
                trajectory_grade = "优秀"
            elif metrics['average_ade'] <= 2.0:
                trajectory_grade = "良好"
            elif metrics['average_ade'] <= 3.0:
                trajectory_grade = "一般"
            else:
                trajectory_grade = "需要改进"
            
            f.write(f"意图识别性能: {intent_grade}\n")
            f.write(f"轨迹预测性能: {trajectory_grade}\n")
            
            f.write("\n5. 输出文件\n")
            f.write("-" * 30 + "\n")
            f.write(f"各车辆详细对比图: {output_dir}/prediction_comparison_vehicle_*.png\n")
            f.write(f"汇总对比图: {output_dir}/prediction_summary_comparison.png\n")
            f.write(f"分析报告: {output_dir}/prediction_comparison_report.txt\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("预测对比分析报告生成完成\n")
            f.write("=" * 60 + "\n")
        
        print(f"预测对比分析报告已保存到: {report_path}")
    
    def run_complete_comparison(self, num_samples=5, output_dir='prediction_comparison'):
        """运行完整的预测对比分析流程"""
        print("开始预测结果对比分析...")
        
        # 1. 加载测试数据
        if not self.load_test_data():
            return False
        
        # 2. 选择预测样例
        if not self.select_prediction_samples(num_samples):
            return False
        
        # 3. 生成模拟预测结果
        if not self.generate_mock_predictions():
            return False
        
        # 4. 计算性能指标
        metrics = self.calculate_prediction_metrics()
        
        # 5. 可视化详细对比
        self.visualize_prediction_comparison(output_dir)
        
        # 6. 生成汇总对比
        self.generate_summary_comparison(output_dir)
        
        # 7. 生成分析报告
        self.generate_comparison_report(output_dir)
        
        print(f"\n{'='*50}")
        print("预测对比分析完成！")
        print("输出文件:")
        print(f"  - {output_dir}/prediction_comparison_vehicle_*.png")
        print(f"  - {output_dir}/prediction_summary_comparison.png")
        print(f"  - {output_dir}/prediction_comparison_report.txt")
        print("="*50)
        
        return True


def main():
    """主函数"""
    # 数据文件路径
    data_path = input("请输入NGSIM数据文件路径 (默认: data/peachtree_filtered_data.csv): ").strip()
    if not data_path:
        data_path = "data/peachtree_filtered_data.csv"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 文件 {data_path} 不存在")
        print("请确保数据文件路径正确")
        return
    
    # 样例数量
    try:
        num_samples = int(input("请输入要对比的预测样例数量 (默认: 5): ").strip() or "5")
    except ValueError:
        num_samples = 5
    
    # 创建对比分析器并运行分析
    comparator = PredictionComparator(data_path)
    comparator.run_complete_comparison(num_samples=num_samples)


if __name__ == "__main__":
    main()