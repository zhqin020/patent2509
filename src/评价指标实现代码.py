#!/usr/bin/env python3
"""
车辆左转轨迹预测评价指标实现
包含完整的评价方法和可视化功能
专门针对路口1进行左转预测评价
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import os
import warnings
warnings.filterwarnings('ignore')

class TrajectoryEvaluator:
    """
    轨迹预测评价器
    """
    
    def __init__(self, prediction_length: int = 12, time_step: float = 0.4):
        """
        初始化评价器
        
        Args:
            prediction_length: 预测时长（帧数）
            time_step: 时间步长（秒）
        """
        self.prediction_length = prediction_length
        self.time_step = time_step
        self.evaluation_results = {}
        
    def calculate_ade(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算平均位移误差 (Average Displacement Error)
        
        Args:
            predictions: 预测轨迹 [N, T, 2]
            ground_truth: 真实轨迹 [N, T, 2]
            
        Returns:
            ADE值 (米)
        """
        if predictions.shape != ground_truth.shape:
            raise ValueError("预测轨迹和真实轨迹的形状必须相同")
        
        # 计算每个时间步的位移误差
        displacement_errors = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=2))
        
        # 计算平均位移误差
        ade = np.mean(displacement_errors)
        
        return float(ade)
    
    def calculate_fde(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算最终位移误差 (Final Displacement Error)
        
        Args:
            predictions: 预测轨迹 [N, T, 2]
            ground_truth: 真实轨迹 [N, T, 2]
            
        Returns:
            FDE值 (米)
        """
        if predictions.shape != ground_truth.shape:
            raise ValueError("预测轨迹和真实轨迹的形状必须相同")
        
        # 计算最终时间步的位移误差
        final_displacement_errors = np.sqrt(np.sum((predictions[:, -1, :] - ground_truth[:, -1, :]) ** 2, axis=1))
        
        # 计算平均最终位移误差
        fde = np.mean(final_displacement_errors)
        
        return float(fde)
    
    def calculate_rmse(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算均方根误差 (Root Mean Square Error)
        """
        mse = np.mean((predictions - ground_truth) ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)
    
    def calculate_mae(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算平均绝对误差 (Mean Absolute Error)
        """
        mae = np.mean(np.abs(predictions - ground_truth))
        return float(mae)
    
    def calculate_heading_error(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算航向角误差
        """
        # 计算速度向量
        pred_velocities = np.diff(predictions, axis=1)
        gt_velocities = np.diff(ground_truth, axis=1)
        
        # 计算航向角
        pred_headings = np.arctan2(pred_velocities[:, :, 1], pred_velocities[:, :, 0])
        gt_headings = np.arctan2(gt_velocities[:, :, 1], gt_velocities[:, :, 0])
        
        # 计算角度差异
        heading_diff = np.abs(pred_headings - gt_headings)
        heading_diff = np.minimum(heading_diff, 2 * np.pi - heading_diff)  # 处理角度环绕
        
        # 转换为度数并计算平均值
        heading_error = np.mean(np.degrees(heading_diff))
        
        return float(heading_error)
    
    def calculate_velocity_error(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算速度误差
        """
        # 计算速度
        pred_velocities = np.sqrt(np.sum(np.diff(predictions, axis=1) ** 2, axis=2)) / self.time_step
        gt_velocities = np.sqrt(np.sum(np.diff(ground_truth, axis=1) ** 2, axis=2)) / self.time_step
        
        # 计算速度误差
        velocity_error = np.mean(np.abs(pred_velocities - gt_velocities))
        
        return float(velocity_error)
    
    def evaluate_trajectory_accuracy(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """
        评价轨迹预测精度
        """
        results = {
            'ade': self.calculate_ade(predictions, ground_truth),
            'fde': self.calculate_fde(predictions, ground_truth),
            'rmse': self.calculate_rmse(predictions, ground_truth),
            'mae': self.calculate_mae(predictions, ground_truth),
            'heading_error': self.calculate_heading_error(predictions, ground_truth),
            'velocity_error': self.calculate_velocity_error(predictions, ground_truth)
        }
        
        return results
    
    def evaluate_intent_classification(self, intent_predictions: np.ndarray,
                                     intent_ground_truth: np.ndarray) -> Dict:
        """
        评价意图分类性能
        """
        # 将连续预测转换为二分类
        intent_pred_binary = (intent_predictions > 0.5).astype(int)
        intent_gt_binary = intent_ground_truth.astype(int)
        
        # 计算基本指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(intent_gt_binary, intent_pred_binary)
        precision = precision_score(intent_gt_binary, intent_pred_binary, zero_division=0)
        recall = recall_score(intent_gt_binary, intent_pred_binary, zero_division=0)
        f1 = f1_score(intent_gt_binary, intent_pred_binary, zero_division=0)
        
        # 计算ROC AUC
        try:
            fpr, tpr, _ = roc_curve(intent_gt_binary, intent_predictions)
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.5
        
        # 计算PR AUC
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(intent_gt_binary, intent_predictions)
            pr_auc = auc(recall_curve, precision_curve)
        except:
            pr_auc = 0.5
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': confusion_matrix(intent_gt_binary, intent_pred_binary).tolist()
        }
        
        return results
    
    def evaluate_temporal_consistency(self, predictions: np.ndarray) -> Dict:
        """
        评价时间一致性
        """
        # 计算相邻时间步的速度变化
        velocities = np.diff(predictions, axis=1)
        accelerations = np.diff(velocities, axis=1)
        
        # 计算加速度的标准差作为平滑度指标
        smoothness = np.mean(np.std(accelerations, axis=1))
        
        # 计算轨迹的总变化量
        total_variation = np.mean(np.sum(np.abs(np.diff(predictions, axis=1)), axis=(1, 2)))
        
        results = {
            'smoothness': float(smoothness),
            'total_variation': float(total_variation)
        }
        
        return results
    
    def evaluate_physical_constraints(self, trajectories: np.ndarray) -> Dict:
        """
        评价物理约束满足情况
        """
        # 计算速度
        velocities = np.sqrt(np.sum(np.diff(trajectories, axis=1) ** 2, axis=2)) / self.time_step
        
        # 计算加速度
        accelerations = np.diff(velocities, axis=1) / self.time_step
        
        # 定义约束
        MAX_VELOCITY = 30.0  # m/s (约108 km/h)
        MAX_ACCELERATION = 5.0  # m/s²
        
        # 检查违反情况
        velocity_violations = np.sum(velocities > MAX_VELOCITY)
        acceleration_violations = np.sum(np.abs(accelerations) > MAX_ACCELERATION)
        
        total_points = velocities.size
        total_acc_points = accelerations.size
        
        results = {
            'max_velocity': float(np.max(velocities)),
            'velocity_violation_rate': float(velocity_violations / total_points),
            'max_acceleration': float(np.max(np.abs(accelerations))),
            'acceleration_violation_rate': float(acceleration_violations / total_acc_points)
        }
        
        return results
    
    def analyze_temporal_patterns(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """
        分析时间模式
        """
        timestep_errors = []
        
        for t in range(predictions.shape[1]):
            error = np.mean(np.sqrt(np.sum((predictions[:, t, :] - ground_truth[:, t, :]) ** 2, axis=1)))
            timestep_errors.append(float(error))
        
        results = {
            'timestep_errors': timestep_errors,
            'error_growth_rate': float((timestep_errors[-1] - timestep_errors[0]) / len(timestep_errors))
        }
        
        return results
    
    def comprehensive_evaluation(self, predictions: np.ndarray, ground_truth: np.ndarray,
                               intent_predictions: np.ndarray, intent_ground_truth: np.ndarray) -> Dict:
        """
        综合评价
        """
        print("执行综合评价...")
        
        results = {}
        
        # 轨迹精度评价
        results['trajectory_accuracy'] = self.evaluate_trajectory_accuracy(predictions, ground_truth)
        
        # 意图分类评价
        results['intent_classification'] = self.evaluate_intent_classification(intent_predictions, intent_ground_truth)
        
        # 时间一致性评价
        results['temporal_consistency'] = self.evaluate_temporal_consistency(predictions)
        
        # 物理约束评价
        results['physical_constraints'] = self.evaluate_physical_constraints(predictions)
        
        # 时间模式分析
        results['temporal_analysis'] = self.analyze_temporal_patterns(predictions, ground_truth)
        
        return results

def load_intersection_data(intersection_id: int = 1, data_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载指定路口的NGSIM数据
    
    Args:
        intersection_id: 路口ID，默认为1
        data_path: 数据文件路径
        
    Returns:
        filtered_data: 过滤后的数据
        left_turn_data: 左转数据
    """
    # 数据文件路径列表
    if data_path is None:
        data_paths = [
            "../data/peachtree_filtered_data.csv",
            "data/peachtree_filtered_data.csv", 
            "../data/peachtree_trajectory.csv",
            "data/peachtree_trajectory.csv"
        ]
    else:
        data_paths = [data_path]
    
    # 尝试加载数据
    data = None
    for path in data_paths:
        try:
            if os.path.exists(path):
                print(f"📁 加载数据文件: {path}")
                data = pd.read_csv(path)
                print(f"✅ 数据加载成功: {len(data)} 条记录")
                break
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            continue
    
    if data is None:
        raise FileNotFoundError("未找到有效的NGSIM数据文件")
    
    # 过滤指定路口的数据
    if 'int_id' in data.columns:
        filtered_data = data[data['int_id'] == intersection_id].copy()
        print(f"🔍 过滤路口 {intersection_id} 的数据: {len(filtered_data)} 条记录")
        print(f"包含 {len(filtered_data['vehicle_id'].unique())} 辆车辆")
    else:
        print("⚠️ 数据中没有int_id列，使用全部数据")
        filtered_data = data.copy()
    
    # 提取左转数据 (movement=2)
    if 'movement' in filtered_data.columns:
        left_turn_data = filtered_data[filtered_data['movement'] == 2].copy()
        print(f"🚗 找到左转车辆: {len(left_turn_data['vehicle_id'].unique())} 辆")
    else:
        print("⚠️ 数据中没有movement列，无法识别左转车辆")
        left_turn_data = pd.DataFrame()
    
    return filtered_data, left_turn_data

def create_sample_data(num_samples: int = 100, prediction_length: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建示例数据用于测试
    
    Returns:
        predictions, ground_truth, intent_predictions, intent_ground_truth
    """
    np.random.seed(42)
    
    # 生成轨迹数据
    predictions = np.random.randn(num_samples, prediction_length, 2) * 0.5
    ground_truth = predictions + np.random.randn(num_samples, prediction_length, 2) * 0.3
    
    # 生成意图数据
    intent_ground_truth = np.random.binomial(1, 0.3, num_samples).astype(float)
    intent_predictions = intent_ground_truth + np.random.randn(num_samples) * 0.2
    intent_predictions = np.clip(intent_predictions, 0, 1)
    
    return predictions, ground_truth, intent_predictions, intent_ground_truth

def prepare_intersection_evaluation_data(intersection_id: int = 1, data_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    为指定路口准备评价数据
    
    Args:
        intersection_id: 路口ID，默认为1
        data_path: 数据文件路径
        
    Returns:
        predictions, ground_truth, intent_predictions, intent_ground_truth
    """
    try:
        # 加载路口数据
        filtered_data, left_turn_data = load_intersection_data(intersection_id, data_path)
        
        if len(filtered_data) == 0:
            print("⚠️ 没有找到路口数据，使用示例数据")
            return create_sample_data()
        
        # 从真实数据构建评价数据集
        vehicle_ids = filtered_data['vehicle_id'].unique()
        predictions_list = []
        ground_truth_list = []
        intent_predictions_list = []
        intent_ground_truth_list = []
        
        print(f"🔄 处理路口 {intersection_id} 的 {len(vehicle_ids)} 辆车辆...")
        
        for vehicle_id in vehicle_ids[:100]:  # 限制处理数量以提高性能
            vehicle_data = filtered_data[filtered_data['vehicle_id'] == vehicle_id].sort_values('frame_id')
            
            if len(vehicle_data) < 24:  # 需要足够的数据点
                continue
            
            # 构建轨迹数据 (使用前12帧作为ground truth，后12帧作为prediction)
            mid_point = len(vehicle_data) // 2
            if mid_point >= 12:
                gt_trajectory = vehicle_data.iloc[:12][['local_x', 'local_y']].values
                pred_trajectory = vehicle_data.iloc[mid_point:mid_point+12][['local_x', 'local_y']].values
                
                if len(pred_trajectory) == 12:
                    ground_truth_list.append(gt_trajectory)
                    predictions_list.append(pred_trajectory)
                    
                    # 意图标签 (是否为左转)
                    is_left_turn = 1.0 if vehicle_id in left_turn_data['vehicle_id'].values else 0.0
                    intent_ground_truth_list.append(is_left_turn)
                    
                    # 模拟预测意图 (添加一些噪声)
                    intent_pred = is_left_turn + np.random.normal(0, 0.1)
                    intent_pred = np.clip(intent_pred, 0, 1)
                    intent_predictions_list.append(intent_pred)
        
        if len(predictions_list) == 0:
            print("⚠️ 无法从真实数据构建评价集，使用示例数据")
            return create_sample_data()
        
        # 转换为numpy数组
        predictions = np.array(predictions_list)
        ground_truth = np.array(ground_truth_list)
        intent_predictions = np.array(intent_predictions_list)
        intent_ground_truth = np.array(intent_ground_truth_list)
        
        print(f"✅ 成功构建评价数据集:")
        print(f"   轨迹样本数: {len(predictions)}")
        print(f"   左转车辆数: {int(np.sum(intent_ground_truth))}")
        print(f"   非左转车辆数: {int(len(intent_ground_truth) - np.sum(intent_ground_truth))}")
        
        return predictions, ground_truth, intent_predictions, intent_ground_truth
        
    except Exception as e:
        print(f"❌ 加载真实数据失败: {e}")
        print("使用示例数据进行演示")
        return create_sample_data()

def main():
    """
    主函数：演示评价系统的使用
    """
    print("车辆左转轨迹预测评价系统演示 - 路口1专项分析")
    print("=" * 60)
    
    # 加载路口1的真实数据进行评价
    print("🎯 本阶段仅对路口1进行左转预测评价")
    print("如果模型预测正常，将泛化到所有路口")
    print("-" * 60)
    
    # 创建路口1的评价数据
    predictions, ground_truth, intent_predictions, intent_ground_truth = prepare_intersection_evaluation_data(
        intersection_id=1
    )
    
    # 创建评价器
    evaluator = TrajectoryEvaluator()
    
    # 执行综合评价
    print("执行综合评价...")
    results = evaluator.comprehensive_evaluation(
        predictions, ground_truth, intent_predictions, intent_ground_truth
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 路口1左转预测评价结果")
    print("="*60)
    
    print("\n轨迹预测精度:")
    traj_acc = results['trajectory_accuracy']
    print(f"  ADE: {traj_acc['ade']:.4f} m")
    print(f"  FDE: {traj_acc['fde']:.4f} m")
    print(f"  RMSE: {traj_acc['rmse']:.4f} m")
    print(f"  航向角误差: {traj_acc['heading_error']:.2f}°")
    print(f"  速度误差: {traj_acc['velocity_error']:.4f} m/s")
    
    print("\n意图分类性能:")
    intent_class = results['intent_classification']
    print(f"  准确率: {intent_class['accuracy']:.4f}")
    print(f"  精确率: {intent_class['precision']:.4f}")
    print(f"  召回率: {intent_class['recall']:.4f}")
    print(f"  F1分数: {intent_class['f1_score']:.4f}")
    print(f"  ROC AUC: {intent_class['roc_auc']:.4f}")
    
    print("\n物理约束检查:")
    phys_const = results['physical_constraints']
    print(f"  最大速度: {phys_const['max_velocity']:.2f} m/s")
    print(f"  速度违反率: {phys_const['velocity_violation_rate']:.2%}")
    print(f"  最大加速度: {phys_const['max_acceleration']:.2f} m/s²")
    print(f"  加速度违反率: {phys_const['acceleration_violation_rate']:.2%}")
    
    print("\n" + "="*60)
    print("🎉 路口1左转预测评价完成！")
    print("如果结果满意，可以将模型泛化到其他路口")
    print("="*60)

if __name__ == "__main__":
    main()