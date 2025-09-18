#!/usr/bin/env python3
"""
车辆左转轨迹预测评价指标实现
包含完整的评价方法和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
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
        
        return ade
    
    def calculate_fde(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算最终位移误差 (Final Displacement Error)
        
        Args:
            predictions: 预测轨迹 [N, T, 2]
            ground_truth: 真实轨迹 [N, T, 2]
            
        Returns:
            FDE值 (米)
        """
        # 提取最终时刻的位置
        final_pred = predictions[:, -1, :]
        final_gt = ground_truth[:, -1, :]
        
        # 计算最终位移误差
        final_displacement_errors = np.sqrt(np.sum((final_pred - final_gt) ** 2, axis=1))
        fde = np.mean(final_displacement_errors)
        
        return fde
    
    def calculate_rmse(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算均方根误差 (Root Mean Square Error)
        """
        mse = np.mean((predictions - ground_truth) ** 2)
        rmse = np.sqrt(mse)
        return rmse
    
    def calculate_mae(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算平均绝对误差 (Mean Absolute Error)
        """
        mae = np.mean(np.abs(predictions - ground_truth))
        return mae
    
    def calculate_heading_error(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算航向角误差
        
        Args:
            predictions: 预测轨迹 [N, T, 2]
            ground_truth: 真实轨迹 [N, T, 2]
            
        Returns:
            平均航向角误差 (度)
        """
        def get_headings(trajectories):
            """计算轨迹的航向角"""
            dx = np.diff(trajectories[:, :, 0], axis=1)
            dy = np.diff(trajectories[:, :, 1], axis=1)
            headings = np.arctan2(dy, dx)
            return headings
        
        pred_headings = get_headings(predictions)
        gt_headings = get_headings(ground_truth)
        
        # 计算角度差异
        heading_diff = pred_headings - gt_headings
        
        # 将角度差异标准化到[-π, π]
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        
        # 计算平均绝对航向角误差
        mean_heading_error = np.mean(np.abs(heading_diff))
        
        # 转换为度数
        return np.degrees(mean_heading_error)
    
    def calculate_velocity_error(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算速度预测误差
        
        Args:
            predictions: 预测轨迹 [N, T, 2]
            ground_truth: 真实轨迹 [N, T, 2]
            
        Returns:
            平均速度误差 (m/s)
        """
        def get_velocities(trajectories):
            """计算轨迹的速度"""
            dx = np.diff(trajectories[:, :, 0], axis=1) / self.time_step
            dy = np.diff(trajectories[:, :, 1], axis=1) / self.time_step
            velocities = np.sqrt(dx**2 + dy**2)
            return velocities
        
        pred_velocities = get_velocities(predictions)
        gt_velocities = get_velocities(ground_truth)
        
        velocity_error = np.mean(np.abs(pred_velocities - gt_velocities))
        return velocity_error
    
    def calculate_acceleration_error(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算加速度预测误差
        """
        def get_accelerations(trajectories):
            """计算轨迹的加速度"""
            velocities = self._get_velocities_for_acceleration(trajectories)
            accelerations = np.diff(velocities, axis=1) / self.time_step
            return accelerations
        
        pred_accelerations = get_accelerations(predictions)
        gt_accelerations = get_accelerations(ground_truth)
        
        acceleration_error = np.mean(np.abs(pred_accelerations - gt_accelerations))
        return acceleration_error
    
    def _get_velocities_for_acceleration(self, trajectories):
        """辅助函数：计算用于加速度计算的速度"""
        dx = np.diff(trajectories[:, :, 0], axis=1) / self.time_step
        dy = np.diff(trajectories[:, :, 1], axis=1) / self.time_step
        velocities = np.sqrt(dx**2 + dy**2)
        return velocities
    
    def calculate_curvature_error(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        计算曲率预测误差
        """
        def get_curvatures(trajectories):
            """计算轨迹的曲率"""
            # 计算一阶和二阶导数
            dx = np.gradient(trajectories[:, :, 0], axis=1)
            dy = np.gradient(trajectories[:, :, 1], axis=1)
            ddx = np.gradient(dx, axis=1)
            ddy = np.gradient(dy, axis=1)
            
            # 计算曲率
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
            
            # 处理数值不稳定性
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
            
            return curvature
        
        pred_curvatures = get_curvatures(predictions)
        gt_curvatures = get_curvatures(ground_truth)
        
        curvature_error = np.mean(np.abs(pred_curvatures - gt_curvatures))
        return curvature_error
    
    def calculate_timestep_errors(self, predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        计算每个时间步的误差
        
        Returns:
            每个时间步的平均位移误差 [T]
        """
        displacement_errors = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=2))
        timestep_errors = np.mean(displacement_errors, axis=0)
        return timestep_errors
    
    def evaluate_intent_classification(self, intent_predictions: np.ndarray, 
                                     intent_ground_truth: np.ndarray) -> Dict:
        """
        评价左转意图分类性能
        
        Args:
            intent_predictions: 意图预测概率 [N]
            intent_ground_truth: 真实意图标签 [N]
            
        Returns:
            分类性能指标字典
        """
        # 二值化预测结果
        binary_predictions = (intent_predictions > 0.5).astype(int)
        binary_ground_truth = (intent_ground_truth > 0.5).astype(int)
        
        # 计算分类指标
        report = classification_report(binary_ground_truth, binary_predictions,
                                     target_names=['Non-Left-Turn', 'Left-Turn'],
                                     output_dict=True, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(binary_ground_truth, binary_predictions)
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(binary_ground_truth, intent_predictions)
        roc_auc = auc(fpr, tpr)
        
        # PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(binary_ground_truth, intent_predictions)
        pr_auc = auc(recall_curve, precision_curve)
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['Left-Turn']['precision'],
            'recall': report['Left-Turn']['recall'],
            'f1_score': report['Left-Turn']['f1-score'],
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        }
    
    def evaluate_physical_constraints(self, trajectories: np.ndarray) -> Dict:
        """
        评价轨迹是否符合物理约束
        
        Args:
            trajectories: 轨迹数据 [N, T, 2]
            
        Returns:
            物理约束违反情况
        """
        results = {}
        
        # 1. 速度约束检查
        velocities = self._get_velocities_for_acceleration(trajectories)
        max_velocity = np.max(velocities)
        results['max_velocity'] = max_velocity
        results['velocity_violation'] = max_velocity > 25.0  # 25 m/s 限制
        results['velocity_violation_rate'] = np.mean(velocities > 25.0)
        
        # 2. 加速度约束检查
        accelerations = np.abs(np.diff(velocities, axis=1) / self.time_step)
        max_acceleration = np.max(accelerations)
        results['max_acceleration'] = max_acceleration
        results['acceleration_violation'] = max_acceleration > 5.0  # 5 m/s² 限制
        results['acceleration_violation_rate'] = np.mean(accelerations > 5.0)
        
        # 3. 转弯半径约束检查
        curvatures = self._calculate_curvatures_for_constraints(trajectories)
        max_curvature = np.max(curvatures)
        min_radius = 1.0 / (max_curvature + 1e-8)  # 避免除零
        results['min_turning_radius'] = min_radius
        results['radius_violation'] = min_radius < 3.0  # 3m 最小转弯半径
        results['radius_violation_rate'] = np.mean(curvatures > 1.0/3.0)
        
        return results
    
    def _calculate_curvatures_for_constraints(self, trajectories):
        """计算用于约束检查的曲率"""
        curvatures_list = []
        
        for traj in trajectories:
            # 计算一阶和二阶导数
            dx = np.gradient(traj[:, 0])
            dy = np.gradient(traj[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            # 计算曲率
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
            curvatures_list.extend(curvature)
        
        return np.array(curvatures_list)
    
    def temporal_error_analysis(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """
        时序误差分析
        
        Args:
            predictions: 预测轨迹 [N, T, 2]
            ground_truth: 真实轨迹 [N, T, 2]
            
        Returns:
            时序分析结果
        """
        # 定义时间窗口
        time_windows = {
            'early': (0, 4),      # 前期预测 (0-1.6秒)
            'middle': (4, 8),     # 中期预测 (1.6-3.2秒)
            'late': (8, 12),      # 后期预测 (3.2-4.8秒)
        }
        
        results = {}
        
        for window_name, (start_t, end_t) in time_windows.items():
            end_t = min(end_t, predictions.shape[1])  # 确保不超出范围
            
            window_pred = predictions[:, start_t:end_t, :]
            window_gt = ground_truth[:, start_t:end_t, :]
            
            if window_pred.shape[1] > 0:  # 确保窗口不为空
                results[window_name] = {
                    'ade': self.calculate_ade(window_pred, window_gt),
                    'fde': self.calculate_fde(window_pred, window_gt),
                    'heading_error': self.calculate_heading_error(window_pred, window_gt),
                    'velocity_error': self.calculate_velocity_error(window_pred, window_gt)
                }
        
        # 计算每个时间步的误差
        timestep_errors = self.calculate_timestep_errors(predictions, ground_truth)
        results['timestep_errors'] = timestep_errors
        
        return results
    
    def comprehensive_evaluation(self, predictions: np.ndarray, ground_truth: np.ndarray,
                               intent_predictions: Optional[np.ndarray] = None,
                               intent_ground_truth: Optional[np.ndarray] = None) -> Dict:
        """
        综合评价
        
        Args:
            predictions: 预测轨迹 [N, T, 2]
            ground_truth: 真实轨迹 [N, T, 2]
            intent_predictions: 意图预测 [N] (可选)
            intent_ground_truth: 真实意图 [N] (可选)
            
        Returns:
            综合评价结果
        """
        results = {}
        
        # 1. 轨迹预测精度
        results['trajectory_accuracy'] = {
            'ade': self.calculate_ade(predictions, ground_truth),
            'fde': self.calculate_fde(predictions, ground_truth),
            'rmse': self.calculate_rmse(predictions, ground_truth),
            'mae': self.calculate_mae(predictions, ground_truth),
            'heading_error': self.calculate_heading_error(predictions, ground_truth),
            'velocity_error': self.calculate_velocity_error(predictions, ground_truth),
            'acceleration_error': self.calculate_acceleration_error(predictions, ground_truth),
            'curvature_error': self.calculate_curvature_error(predictions, ground_truth)
        }
        
        # 2. 意图分类性能
        if intent_predictions is not None and intent_ground_truth is not None:
            results['intent_classification'] = self.evaluate_intent_classification(
                intent_predictions, intent_ground_truth)
        
        # 3. 物理约束检查
        results['physical_constraints'] = self.evaluate_physical_constraints(predictions)
        
        # 4. 时序分析
        results['temporal_analysis'] = self.temporal_error_analysis(predictions, ground_truth)
        
        # 5. 统计信息
        results['statistics'] = {
            'num_samples': len(predictions),
            'prediction_length': predictions.shape[1],
            'time_horizon': predictions.shape[1] * self.time_step
        }
        
        return results

class PerformanceBenchmark:
    """
    性能基准测试
    """
    
    def __init__(self):
        self.baseline_results = {}
        self.performance_thresholds = {
            'safety_critical': {
                'ade': 0.3,
                'fde': 0.5,
                'intent_accuracy': 0.95,
                'max_latency': 0.05  # 50ms
            },
            'comfort_optimization': {
                'ade': 0.8,
                'fde': 1.5,
                'intent_accuracy': 0.85,
                'max_latency': 0.2   # 200ms
            },
            'traffic_management': {
                'ade': 1.0,
                'fde': 2.0,
                'intent_accuracy': 0.80,
                'max_latency': 0.5   # 500ms
            }
        }
    
    def add_baseline(self, name: str, results: Dict):
        """添加基线方法结果"""
        self.baseline_results[name] = results
    
    def compare_with_baselines(self, our_results: Dict) -> Dict:
        """
        与基线方法比较
        
        Args:
            our_results: 我们方法的结果
            
        Returns:
            比较结果
        """
        comparison = {}
        
        for baseline_name, baseline_results in self.baseline_results.items():
            comparison[baseline_name] = {}
            
            # 轨迹预测指标比较
            if 'trajectory_accuracy' in both_results(our_results, baseline_results):
                traj_comparison = {}
                our_traj = our_results['trajectory_accuracy']
                baseline_traj = baseline_results['trajectory_accuracy']
                
                for metric in ['ade', 'fde', 'rmse', 'mae']:
                    if metric in our_traj and metric in baseline_traj:
                        our_score = our_traj[metric]
                        baseline_score = baseline_traj[metric]
                        # 对于误差指标，越小越好
                        improvement = (baseline_score - our_score) / baseline_score * 100
                        traj_comparison[metric] = {
                            'our_method': our_score,
                            'baseline': baseline_score,
                            'improvement_percent': improvement
                        }
                
                comparison[baseline_name]['trajectory_prediction'] = traj_comparison
            
            # 意图分类指标比较
            if 'intent_classification' in both_results(our_results, baseline_results):
                intent_comparison = {}
                our_intent = our_results['intent_classification']
                baseline_intent = baseline_results['intent_classification']
                
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in our_intent and metric in baseline_intent:
                        our_score = our_intent[metric]
                        baseline_score = baseline_intent[metric]
                        # 对于准确率指标，越大越好
                        improvement = (our_score - baseline_score) / baseline_score * 100
                        intent_comparison[metric] = {
                            'our_method': our_score,
                            'baseline': baseline_score,
                            'improvement_percent': improvement
                        }
                
                comparison[baseline_name]['intent_classification'] = intent_comparison
        
        return comparison
    
    def evaluate_application_readiness(self, results: Dict, application_type: str = 'safety_critical') -> Dict:
        """
        评价应用就绪程度
        
        Args:
            results: 评价结果
            application_type: 应用类型
            
        Returns:
            就绪程度评价
        """
        if application_type not in self.performance_thresholds:
            raise ValueError(f"未知的应用类型: {application_type}")
        
        thresholds = self.performance_thresholds[application_type]
        readiness = {}
        
        # 检查轨迹预测精度
        if 'trajectory_accuracy' in results:
            traj_results = results['trajectory_accuracy']
            readiness['ade_ready'] = traj_results.get('ade', float('inf')) <= thresholds['ade']
            readiness['fde_ready'] = traj_results.get('fde', float('inf')) <= thresholds['fde']
        
        # 检查意图分类精度
        if 'intent_classification' in results:
            intent_results = results['intent_classification']
            readiness['intent_ready'] = intent_results.get('accuracy', 0) >= thresholds['intent_accuracy']
        
        # 计算总体就绪程度
        ready_count = sum(readiness.values())
        total_count = len(readiness)
        readiness['overall_readiness'] = ready_count / total_count if total_count > 0 else 0
        readiness['application_type'] = application_type
        
        return readiness

def both_results(results1, results2):
    """辅助函数：检查两个结果字典是否都包含某个键"""
    def check_key(key):
        return key in results1 and key in results2
    return check_key

class EvaluationVisualizer:
    """
    评价结果可视化
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_trajectory_errors(self, timestep_errors: np.ndarray, save_path: str = None):
        """
        绘制时间步误差曲线
        """
        plt.figure(figsize=self.figsize)
        
        timesteps = np.arange(1, len(timestep_errors) + 1)
        time_seconds = timesteps * 0.4  # 转换为秒
        
        plt.plot(time_seconds, timestep_errors, 'o-', linewidth=2, markersize=6, color='blue')
        plt.fill_between(time_seconds, timestep_errors, alpha=0.3, color='blue')
        
        plt.xlabel('Prediction Time (seconds)', fontsize=12)
        plt.ylabel('Average Displacement Error (m)', fontsize=12)
        plt.title('Trajectory Prediction Error vs Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加性能阈值线
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Acceptable Threshold')
        plt.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_distributions(self, predictions: np.ndarray, ground_truth: np.ndarray, 
                               save_path: str = None):
        """
        绘制误差分布图
        """
        # 计算各种误差
        displacement_errors = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=2))
        ade_errors = np.mean(displacement_errors, axis=1)
        fde_errors = displacement_errors[:, -1]
        x_errors = (predictions - ground_truth)[:, :, 0].flatten()
        y_errors = (predictions - ground_truth)[:, :, 1].flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ADE分布
        axes[0, 0].hist(ade_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(np.mean(ade_errors), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(ade_errors):.3f}m')
        axes[0, 0].set_xlabel('ADE (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('ADE Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # FDE分布
        axes[0, 1].hist(fde_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(np.mean(fde_errors), color='red', linestyle='--',
                          label=f'Mean: {np.mean(fde_errors):.3f}m')
        axes[0, 1].set_xlabel('FDE (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('FDE Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # X方向误差分布
        axes[1, 0].hist(x_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(x_errors), color='red', linestyle='--',
                          label=f'Mean: {np.mean(x_errors):.3f}m')
        axes[1, 0].set_xlabel('X Direction Error (m)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('X Direction Error Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Y方向误差分布
        axes[1, 1].hist(y_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(np.mean(y_errors), color='red', linestyle='--',
                          label=f'Mean: {np.mean(y_errors):.3f}m')
        axes[1, 1].set_xlabel('Y Direction Error (m)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Y Direction Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_intent_classification_results(self, intent_results: Dict, save_path: str = None):
        """
        绘制意图分类结果
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 混淆矩阵
        cm = intent_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Non-Left Turn', 'Left Turn'],
                   yticklabels=['Non-Left Turn', 'Left Turn'])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # ROC曲线
        fpr, tpr = intent_results['fpr'], intent_results['tpr']
        roc_auc = intent_results['roc_auc']
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        
        # PR曲线
        precision_curve, recall_curve = intent_results['precision_curve'], intent_results['recall_curve']
        pr_auc = intent_results['pr_auc']
        axes[2].plot(recall_curve, precision_curve, color='darkorange', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[2].set_xlim([0.0, 1.0])
        axes[2].set_ylim([0.0, 1.05])
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].legend(loc="lower left")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, comparison_results: Dict, save_path: str = None):
        """
        绘制性能比较图
        """
        # 提取比较数据
        methods = list(comparison_results.keys())
        metrics = ['ade', 'fde', 'accuracy', 'f1_score']
        
        # 准备数据
        data = []
        for method in methods:
            method_data = {'Method': method}
            
            # 轨迹预测指标
            if 'trajectory_prediction' in comparison_results[method]:
                traj_data = comparison_results[method]['trajectory_prediction']
                for metric in ['ade', 'fde']:
                    if metric in traj_data:
                        method_data[metric] = traj_data[metric]['improvement_percent']
            
            # 意图分类指标
            if 'intent_classification' in comparison_results[method]:
                intent_data = comparison_results[method]['intent_classification']
                for metric in ['accuracy', 'f1_score']:
                    if metric in intent_data:
                        method_data[metric] = intent_data[metric]['improvement_percent']
            
            data.append(method_data)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 绘制比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                bars = axes[i].bar(df['Method'], df[metric], color=colors[i], alpha=0.7)
                axes[i].set_title(f'{metric.upper()} Improvement (%)')
                axes[i].set_ylabel('Improvement (%)')
                axes[i].grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}%', ha='center', va='bottom')
                
                # 添加零线
                axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

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

def main():
    """
    主函数：演示评价系统的使用
    """
    print("车辆左转轨迹预测评价系统演示")
    print("=" * 50)
    
    # 创建示例数据
    predictions, ground_truth, intent_predictions, intent_ground_truth = create_sample_data()
    
    # 创建评价器
    evaluator = TrajectoryEvaluator()
    
    # 执行综合评价
    print("执行综合评价...")
    results = evaluator.comprehensive_evaluation(
        predictions, ground_truth, intent_predictions, intent_ground_truth
    )
    
    # 打印结果
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
    
    # 创建可视化器
    visualizer = EvaluationVisualizer()
    
    # 生成可视化结果
    print("\n生成可视化结果...")
    
    # 时间步误差曲线
    timestep_errors = results['temporal_analysis']['timestep_errors']
    visualizer.plot_trajectory_errors(timestep_errors)
    
    # 误差分布图
    visualizer.plot_error_distributions(predictions, ground_truth)
    
    # 意图分类结果
    visualizer.plot_intent_classification_results(results['intent_classification'])
    
    # 性能基准测试
    print("\n执行性能基准测试...")
    benchmark = PerformanceBenchmark()
    
    # 添加模拟的基线结果
    baseline_results = {
        'trajectory_accuracy': {
            'ade': 0.65,
            'fde': 1.20,
            'rmse': 0.75,
            'mae': 0.58
        },
        'intent_classification': {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }
    }
    benchmark.add_baseline('LSTM_Baseline', baseline_results)
    
    # 比较结果
    comparison = benchmark.compare_with_baselines(results)
    print("\n与基线方法比较:")
    for baseline_name, comp_results in comparison.items():
        print(f"\n{baseline_name}:")
        if 'trajectory_prediction' in comp_results:
            for metric, values in comp_results['trajectory_prediction'].items():
                print(f"  {metric}: {values['improvement_percent']:+.1f}%")
    
    # 应用就绪程度评价
    readiness = benchmark.evaluate_application_readiness(results, 'safety_critical')
    print(f"\n安全关键应用就绪程度: {readiness['overall_readiness']:.1%}")
    
    print("\n" + "=" * 50)
    print("评价系统演示完成！")

if __name__ == "__main__":
    main()