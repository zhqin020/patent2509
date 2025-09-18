#!/usr/bin/env python3
"""
NGSIM数据格式兼容性测试程序
测试修复后的代码是否能正确处理NGSIM数据
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

def test_ngsim_data_loading():
    """测试NGSIM数据加载"""
    print("=== NGSIM数据加载测试 ===")
    
    data_path = "../data/peachtree_filtered_data.csv"
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在")
        return False
    
    try:
        # 加载数据
        data = pd.read_csv(data_path)
        print(f"数据加载成功: {len(data)} 条记录")
        print(f"数据列: {list(data.columns)}")
        print(f"车辆数量: {len(data['Vehicle_ID'].unique())}")
        
        # 检查关键列是否存在
        required_columns = ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"警告: 缺少列 {missing_columns}")
        else:
            print("所有必需的列都存在")
        
        return True
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False

def test_multimodal_dataset():
    """测试MultiModalDataset类"""
    print("\n=== MultiModalDataset测试 ===")
    
    try:
        from 代码实现框架 import MultiModalDataset
        
        # 创建数据集
        dataset = MultiModalDataset("../data/peachtree_filtered_data.csv")
        print(f"数据集创建成功: {len(dataset)} 个样本")
        
        # 测试获取单个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print("样本数据结构:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} (tensor)")
                else:
                    print(f"  {key}: {type(value)}")
            
            return True
        else:
            print("数据集为空")
            return False
            
    except Exception as e:
        print(f"MultiModalDataset测试失败: {e}")
        return False

def test_left_turn_analysis():
    """测试左转数据分析脚本"""
    print("\n=== 左转数据分析测试 ===")
    
    try:
        from 左转数据分析脚本 import LeftTurnAnalyzer
        
        # 创建分析器
        analyzer = LeftTurnAnalyzer("../data/peachtree_filtered_data.csv")
        
        # 加载数据
        if analyzer.load_data():
            print("数据加载成功")
            
            # 识别左转车辆
            if analyzer.identify_left_turn_vehicles():
                print("左转车辆识别成功")
                
                # 选择样例车辆
                if analyzer.select_sample_vehicles(num_samples=3):
                    print("样例车辆选择成功")
                    return True
        
        return False
        
    except Exception as e:
        print(f"左转数据分析测试失败: {e}")
        return False

def test_prediction_comparison():
    """测试预测对比脚本"""
    print("\n=== 预测对比测试 ===")
    
    try:
        from 预测样例对比脚本 import PredictionComparator
        
        # 创建对比器
        comparator = PredictionComparator("../data/peachtree_filtered_data.csv")
        
        # 加载数据
        if comparator.load_test_data():
            print("测试数据加载成功")
            
            # 选择预测样例
            if comparator.select_prediction_samples(num_samples=3):
                print("预测样例选择成功")
                return True
        
        return False
        
    except Exception as e:
        print(f"预测对比测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n=== 模型创建测试 ===")
    
    try:
        from 代码实现框架 import LeftTurnPredictor
        
        # 创建模型
        model = LeftTurnPredictor()
        print("模型创建成功")
        
        # 测试前向传播
        batch_size = 2
        visual_feat = torch.randn(batch_size, 64)
        motion_feat = torch.randn(batch_size, 40)
        traffic_feat = torch.randn(batch_size, 32)
        
        intent_pred, traj_pred = model(visual_feat, motion_feat, traffic_feat)
        
        print(f"意图预测输出形状: {intent_pred.shape}")
        print(f"轨迹预测输出形状: {traj_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"模型创建测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始NGSIM数据格式兼容性测试...")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("数据加载", test_ngsim_data_loading()))
    test_results.append(("MultiModalDataset", test_multimodal_dataset()))
    test_results.append(("左转数据分析", test_left_turn_analysis()))
    test_results.append(("预测对比", test_prediction_comparison()))
    test_results.append(("模型创建", test_model_creation()))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "通过" if result else "失败"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("所有测试通过! NGSIM数据格式兼容性修复成功")
    else:
        print("部分测试失败，需要进一步修复")
    
    print("=" * 50)

if __name__ == "__main__":
    main()