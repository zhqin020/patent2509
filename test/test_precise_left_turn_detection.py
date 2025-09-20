#!/usr/bin/env python3
"""
基于NGSIM真实movement标签的左转检测验证脚本
使用movement=2的真实左转数据来验证和改进预测算法
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_ngsim_data():
    """加载NGSIM数据"""
    data_paths = [
        "../data/peachtree_filtered_data.csv",
        "data/peachtree_filtered_data.csv", 
        "../data/peachtree_trajectory.csv",
        "data/peachtree_trajectory.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"📁 找到数据文件: {path}")
            try:
                data = pd.read_csv(path)
                print(f"✅ 数据加载成功: {len(data)} 条记录, {len(data['vehicle_id'].unique())} 辆车")
                return data, path
            except Exception as e:
                print(f"❌ 数据加载失败: {e}")
                continue
    
    print("❌ 未找到有效的NGSIM数据文件")
    return None, None

def analyze_movement_labels(data):
    """分析movement标签分布"""
    print("" + "="*60)
    print("📊 NGSIM数据中的movement标签分析")
    print("="*60)
    
    # 统计movement分布
    movement_counts = data['movement'].value_counts().sort_index()
    total_records = len(data)
    
    movement_names = {
        1: "直行 (Straight)",
        2: "左转 (Left Turn)", 
        3: "右转 (Right Turn)"
    }
    
    print("Movement标签分布:")
    for movement, count in movement_counts.items():
        name = movement_names.get(movement, f"未知({movement})")
        percentage = count / total_records * 100
        print(f"  {name}: {count:,} 条记录 ({percentage:.2f}%)")
    
    # 按路口分析
    print(f"路口分布:")
    intersection_counts = data['int_id'].value_counts().sort_index()
    for int_id, count in intersection_counts.items():
        percentage = count / total_records * 100
        print(f"  路口 {int_id}: {count:,} 条记录 ({percentage:.2f}%)")
    
    return movement_counts

def extract_true_left_turns(data):
    """提取真实的左转数据 (movement=2)"""
    print("" + "="*60)
    print("🔍 提取真实左转数据 (movement=2)")
    print("="*60)
    
    # 筛选左转数据
    left_turn_data = data[data['movement'] == 2].copy()
    print(f"找到 {len(left_turn_data)} 条左转记录")
    
    if len(left_turn_data) == 0:
        print("❌ 没有找到左转数据 (movement=2)")
        return None
    
    # 按路口和车辆分组分析
    left_turn_vehicles = left_turn_data['vehicle_id'].unique()
    print(f"涉及 {len(left_turn_vehicles)} 辆车辆")
    
    # 按路口分组
    by_intersection = left_turn_data.groupby('int_id').agg({
        'vehicle_id': 'nunique',
        'frame_id': 'count'
    }).rename(columns={'vehicle_id': 'vehicles', 'frame_id': 'records'})
    
    print(f"各路口的左转情况:")
    for int_id, row in by_intersection.iterrows():
        print(f"  路口 {int_id}: {row['vehicles']} 辆车, {row['records']} 条记录")
    
    return left_turn_data

def analyze_left_turn_trajectories(data, left_turn_data):
    """分析左转车辆的完整轨迹"""
    print("" + "="*60)
    print("🚗 分析左转车辆的完整轨迹")
    print("="*60)
    
    left_turn_vehicles = left_turn_data['vehicle_id'].unique()
    trajectory_analysis = []
    
    for vehicle_id in left_turn_vehicles[:10]:  # 分析前10辆车
        # 获取该车辆的完整轨迹
        vehicle_traj = data[data['vehicle_id'] == vehicle_id].sort_values('frame_id')
        
        if len(vehicle_traj) < 10:
            continue
            
        # 分析轨迹特征
        start_pos = (vehicle_traj.iloc[0]['local_x'], vehicle_traj.iloc[0]['local_y'])
        end_pos = (vehicle_traj.iloc[-1]['local_x'], vehicle_traj.iloc[-1]['local_y'])
        
        # 计算直线距离和路径长度
        straight_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        dx = vehicle_traj['local_x'].diff().fillna(0)
        dy = vehicle_traj['local_y'].diff().fillna(0)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        
        # 计算航向角变化
        headings = np.degrees(np.arctan2(dy, dx))
        heading_change = headings.iloc[-1] - headings.iloc[1] if len(headings) > 1 else 0
        
        # 标准化角度
        while heading_change > 180:
            heading_change -= 360
        while heading_change < -180:
            heading_change += 360
        
        # 统计movement=2的帧数
        left_turn_frames = len(vehicle_traj[vehicle_traj['movement'] == 2])
        total_frames = len(vehicle_traj)
        
        analysis = {
            'vehicle_id': vehicle_id,
            'int_id': vehicle_traj.iloc[0]['int_id'],
            'total_frames': total_frames,
            'left_turn_frames': left_turn_frames,
            'left_turn_ratio': left_turn_frames / total_frames,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'straight_distance': straight_distance,
            'path_length': path_length,
            'curvature_ratio': path_length / straight_distance if straight_distance > 0 else float('inf'),
            'heading_change': heading_change,
            'avg_speed': vehicle_traj['v_vel'].mean(),
            'max_speed': vehicle_traj['v_vel'].max()
        }
        
        trajectory_analysis.append(analysis)
        
        print(f"--- 车辆 {vehicle_id} (路口 {analysis['int_id']}) ---")
        print(f"总帧数: {total_frames}, 左转帧数: {left_turn_frames} ({left_turn_frames/total_frames*100:.1f}%)")
        print(f"起点: ({start_pos[0]:.1f}, {start_pos[1]:.1f}), 终点: ({end_pos[0]:.1f}, {end_pos[1]:.1f})")
        print(f"直线距离: {straight_distance:.2f}m, 路径长度: {path_length:.2f}m")
        print(f"曲率比: {analysis['curvature_ratio']:.2f}, 航向角变化: {heading_change:.1f}°")
        print(f"平均速度: {analysis['avg_speed']:.2f}m/s, 最大速度: {analysis['max_speed']:.2f}m/s")
    
    return trajectory_analysis

def compare_with_prediction_algorithm(data, left_turn_data):
    """对比预测算法与真实标签的差异"""
    print("" + "="*60)
    print("🔬 对比预测算法与真实movement=2标签")
    print("="*60)
    
    try:
        # 尝试导入我们的预测算法
        sys.path.append('../src')
        from 改进的左转检测算法 import PreciseLeftTurnDetector
        
        # 创建临时数据文件
        temp_file = "temp_ngsim_data.csv"
        data.to_csv(temp_file, index=False)
        
        # 运行预测算法
        detector = PreciseLeftTurnDetector(temp_file)
        detector.load_data()
        detector.run_precise_classification()
        
        # 获取预测结果
        predicted_left_turns = detector.maneuver_stats.get("left_turn", [])
        true_left_turns = left_turn_data['vehicle_id'].unique()
        
        print(f"真实左转车辆 (movement=2): {len(true_left_turns)} 辆")
        print(f"预测左转车辆: {len(predicted_left_turns)} 辆")
        
        # 计算重叠
        true_set = set(true_left_turns)
        pred_set = set(predicted_left_turns)
        
        correct_predictions = true_set & pred_set
        false_positives = pred_set - true_set
        false_negatives = true_set - pred_set
        
        print(f"预测准确性分析:")
        print(f"正确预测: {len(correct_predictions)} 辆")
        print(f"误报 (预测为左转但实际不是): {len(false_positives)} 辆")
        print(f"漏报 (实际左转但未预测): {len(false_negatives)} 辆")
        
        if len(predicted_left_turns) > 0:
            precision = len(correct_predictions) / len(predicted_left_turns)
            print(f"精确率 (Precision): {precision:.3f}")
        
        if len(true_left_turns) > 0:
            recall = len(correct_predictions) / len(true_left_turns)
            print(f"召回率 (Recall): {recall:.3f}")
        
        if len(correct_predictions) > 0:
            f1 = 2 * precision * recall / (precision + recall)
            print(f"F1分数: {f1:.3f}")
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return {
            'true_positives': len(correct_predictions),
            'false_positives': len(false_positives), 
            'false_negatives': len(false_negatives),
            'precision': precision if len(predicted_left_turns) > 0 else 0,
            'recall': recall if len(true_left_turns) > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ 预测算法对比失败: {e}")
        return None

def analyze_by_intersection(data, left_turn_data):
    """按路口分别分析"""
    print("" + "="*60)
    print("🏢 按路口分别分析左转数据")
    print("="*60)
    
    intersections = left_turn_data['int_id'].unique()
    
    for int_id in intersections:
        print(f"--- 路口 {int_id} 分析 ---")
        
        # 该路口的所有数据
        int_data = data[data['int_id'] == int_id]
        int_left_turns = left_turn_data[left_turn_data['int_id'] == int_id]
        
        total_vehicles = len(int_data['vehicle_id'].unique())
        left_turn_vehicles = len(int_left_turns['vehicle_id'].unique())
        
        print(f"总车辆数: {total_vehicles}")
        print(f"左转车辆数: {left_turn_vehicles}")
        print(f"左转比例: {left_turn_vehicles/total_vehicles*100:.2f}%")
        
        # 分析该路口左转车辆的特征
        if len(int_left_turns) > 0:
            avg_speed = int_left_turns['v_vel'].mean()
            avg_acc = int_left_turns['v_acc'].mean()
            
            print(f"左转时平均速度: {avg_speed:.2f} m/s")
            print(f"左转时平均加速度: {avg_acc:.2f} m/s²")
            
            # 分析位置分布
            x_range = int_left_turns['local_x'].max() - int_left_turns['local_x'].min()
            y_range = int_left_turns['local_y'].max() - int_left_turns['local_y'].min()
            print(f"左转区域范围: X轴 {x_range:.1f}m, Y轴 {y_range:.1f}m")

def visualize_left_turn_analysis(data, left_turn_data, trajectory_analysis):
    """可视化左转分析结果"""
    print("" + "="*60)
    print("📊 生成左转分析可视化图表")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Movement分布饼图
    movement_counts = data['movement'].value_counts()
    movement_labels = {1: '直行', 2: '左转', 3: '右转'}
    labels = [movement_labels.get(m, f'其他({m})') for m in movement_counts.index]
    
    axes[0, 0].pie(movement_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('NGSIM数据中的Movement分布')
    
    # 2. 各路口左转车辆数
    int_left_turns = left_turn_data.groupby('int_id')['vehicle_id'].nunique()
    axes[0, 1].bar(int_left_turns.index.astype(str), int_left_turns.values)
    axes[0, 1].set_title('各路口的左转车辆数')
    axes[0, 1].set_xlabel('路口ID')
    axes[0, 1].set_ylabel('左转车辆数')
    
    # 3. 左转车辆速度分布
    axes[0, 2].hist(left_turn_data['v_vel'], bins=30, alpha=0.7, color='orange')
    axes[0, 2].set_title('左转时的速度分布')
    axes[0, 2].set_xlabel('速度 (m/s)')
    axes[0, 2].set_ylabel('频次')
    
    # 4. 轨迹特征分析
    if trajectory_analysis:
        curvature_ratios = [t['curvature_ratio'] for t in trajectory_analysis if t['curvature_ratio'] < 10]
        axes[1, 0].hist(curvature_ratios, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('左转车辆曲率比分布')
        axes[1, 0].set_xlabel('曲率比 (路径长度/直线距离)')
        axes[1, 0].set_ylabel('频次')
    
    # 5. 航向角变化分布
    if trajectory_analysis:
        heading_changes = [t['heading_change'] for t in trajectory_analysis]
        axes[1, 1].hist(heading_changes, bins=20, alpha=0.7, color='red')
        axes[1, 1].set_title('左转车辆航向角变化分布')
        axes[1, 1].set_xlabel('航向角变化 (度)')
        axes[1, 1].set_ylabel('频次')
    
    # 6. 左转轨迹示例
    if len(left_turn_data) > 0:
        # 选择一个左转车辆绘制轨迹
        sample_vehicle = left_turn_data['vehicle_id'].iloc[0]
        sample_traj = data[data['vehicle_id'] == sample_vehicle].sort_values('frame_id')
        
        # 区分左转和非左转部分
        left_turn_part = sample_traj[sample_traj['movement'] == 2]
        other_part = sample_traj[sample_traj['movement'] != 2]
        
        if len(other_part) > 0:
            axes[1, 2].plot(other_part['local_x'], other_part['local_y'], 'b-', alpha=0.6, label='其他阶段')
        if len(left_turn_part) > 0:
            axes[1, 2].plot(left_turn_part['local_x'], left_turn_part['local_y'], 'r-', linewidth=3, label='左转阶段')
        
        axes[1, 2].set_title(f'车辆 {sample_vehicle} 轨迹示例')
        axes[1, 2].set_xlabel('Local X (m)')
        axes[1, 2].set_ylabel('Local Y (m)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axis('equal')
    
    plt.tight_layout()
    plt.savefig('ngsim_left_turn_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 可视化图表已保存为: ngsim_left_turn_analysis.png")

def main():
    """主函数"""
    print("="*60)
    print("🎯 基于NGSIM真实标签的左转检测验证")
    print("="*60)
    
    # 1. 加载数据
    data, data_path = load_ngsim_data()
    if data is None:
        return
    
    # 2. 分析movement标签
    movement_counts = analyze_movement_labels(data)
    
    # 3. 提取真实左转数据
    left_turn_data = extract_true_left_turns(data)
    if left_turn_data is None:
        return
    
    # 4. 分析左转轨迹
    trajectory_analysis = analyze_left_turn_trajectories(data, left_turn_data)
    
    # 5. 按路口分析
    analyze_by_intersection(data, left_turn_data)
    
    # 6. 对比预测算法
    comparison_results = compare_with_prediction_algorithm(data, left_turn_data)
    
    # 7. 可视化分析
    visualize_left_turn_analysis(data, left_turn_data, trajectory_analysis)
    
    print("" + "="*60)
    print("🎉 基于真实标签的左转分析完成！")
    print("="*60)

def test_precise_detection():
    """精确检测测试函数"""
    print("🔍 开始精确检测测试...")
    
    try:
        # 1. 加载数据
        data, data_path = load_ngsim_data()
        if data is None:
            print("⚠️ 无法加载NGSIM数据，创建模拟数据进行测试")
            create_mock_data()
            if os.path.exists("mock_data.csv"):
                data = pd.read_csv("mock_data.csv")
                # 添加必要的列
                data['movement'] = 1  # 默认为直行
                data['int_id'] = 1    # 默认路口ID
                # 为前5辆车设置为左转
                data.loc[data['vehicle_id'].isin([1, 2, 3, 4, 5]), 'movement'] = 2
            else:
                print("❌ 无法创建测试数据")
                return False
        
        # 2. 分析movement标签
        movement_counts = analyze_movement_labels(data)
        
        # 3. 提取真实左转数据
        left_turn_data = extract_true_left_turns(data)
        if left_turn_data is None:
            print("❌ 无法提取左转数据")
            return False
        
        # 4. 分析左转轨迹
        trajectory_analysis = analyze_left_turn_trajectories(data, left_turn_data)
        
        # 5. 按路口分析
        analyze_by_intersection(data, left_turn_data)
        
        print("✅ 所有测试步骤完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_mock_data():
    """创建模拟测试数据"""
    print("📊 创建模拟测试数据...")
    
    # 创建不同类型的车辆轨迹
    mock_data = []
    
    # 1. 真正的左转车辆
    for vid in range(1, 6):
        frames = range(100 + vid * 1000, 200 + vid * 1000)
        for i, frame in enumerate(frames):
            # 模拟左转轨迹：从东向北
            progress = i / len(frames)
            if progress < 0.3:  # 直行阶段
                x = 100 + progress * 100
                y = 100
            elif progress < 0.7:  # 转弯阶段
                angle = (progress - 0.3) * np.pi / 2 / 0.4
                radius = 50
                x = 130 + radius * np.cos(angle)
                y = 100 + radius * np.sin(angle)
            else:  # 直行阶段
                x = 130
                y = 150 + (progress - 0.7) * 100
            
            mock_data.append({
                'vehicle_id': vid,
                'frame_id': frame,
                'local_x': x + np.random.normal(0, 1),  # 添加噪声
                'local_y': y + np.random.normal(0, 1),
                'v_vel': 10 + np.random.normal(0, 2),
                'v_acc': np.random.normal(0, 1)
            })
    
    # 2. 掉头车辆
    for vid in range(10, 13):
        frames = range(100 + vid * 1000, 200 + vid * 1000)
        for i, frame in enumerate(frames):
            # 模拟掉头轨迹：180度转向，净位移小
            progress = i / len(frames)
            angle = progress * np.pi
            radius = 20
            x = 200 + radius * np.cos(angle)
            y = 200 + radius * np.sin(angle)
            
            mock_data.append({
                'vehicle_id': vid,
                'frame_id': frame,
                'local_x': x + np.random.normal(0, 1),
                'local_y': y + np.random.normal(0, 1),
                'v_vel': 5 + np.random.normal(0, 1),
                'v_acc': np.random.normal(0, 0.5)
            })
    
    # 3. 直行车辆
    for vid in range(20, 30):
        frames = range(100 + vid * 1000, 200 + vid * 1000)
        for i, frame in enumerate(frames):
            # 模拟直行轨迹
            progress = i / len(frames)
            x = 300 + progress * 200
            y = 300 + np.random.normal(0, 2)  # 轻微横向偏移
            
            mock_data.append({
                'vehicle_id': vid,
                'frame_id': frame,
                'local_x': x,
                'local_y': y,
                'v_vel': 15 + np.random.normal(0, 3),
                'v_acc': np.random.normal(0, 1)
            })
    
    # 保存模拟数据
    df = pd.DataFrame(mock_data)
    df.to_csv("mock_data.csv", index=False)
    print(f"✅ 模拟数据创建完成，包含 {len(df)} 条记录，{len(df['vehicle_id'].unique())} 辆车辆")

def main():
    """主测试函数"""
    print("="*60)
    print("🎯 精确左转检测算法测试")
    print("="*60)
    
    success = test_precise_detection()
    
    print("\n" + "="*60)
    if success:
        print("🎉 测试完成！算法工作正常")
    else:
        print("⚠️ 测试发现问题，请检查算法实现")
    print("="*60)

if __name__ == "__main__":
    main()