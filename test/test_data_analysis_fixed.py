#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析功能测试程序（修复版）
测试左转数据分析和预测对比功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_left_turn_analysis():
    """测试左转数据分析功能"""
    try:
        print("=" * 50)
        print("测试左转数据分析功能")
        print("=" * 50)
        
        # 导入左转数据分析模块
        from 左转数据分析脚本 import LeftTurnAnalyzer
        
        # 创建分析器实例
        analyzer = LeftTurnAnalyzer('../data/peachtree_filtered_data.csv')
        print("[OK] 左转数据分析器创建成功")
        
        # 测试数据加载
        analyzer.load_data()
        print("[OK] 数据加载成功")
        
        # 测试左转车辆识别
        left_turn_vehicles = analyzer.identify_left_turn_vehicles()
        print(f"[OK] 识别到 {len(left_turn_vehicles)} 个左转车辆")
        
        # 测试特征提取
        if len(left_turn_vehicles) > 0:
            sample_vehicles = left_turn_vehicles[:3]  # 取前3个作为样例
            features = analyzer.extract_features(sample_vehicles)
            print(f"[OK] 成功提取 {len(features)} 个车辆的特征")
        
        print("[OK] 左转数据分析功能测试通过")
        return True
        
    except Exception as e:
        print(f"[FAIL] 左转数据分析功能测试失败: {e}")
        return False

def test_prediction_comparison():
    """测试预测对比功能"""
    try:
        print("=" * 50)
        print("测试预测对比功能")
        print("=" * 50)
        
        # 导入预测对比模块
        from 预测样例对比脚本 import PredictionComparator
        
        # 创建对比器实例
        comparator = PredictionComparator('../data/peachtree_filtered_data.csv')
        print("[OK] 预测对比器创建成功")
        
        # 测试数据加载
        comparator.load_data()
        print("[OK] 数据加载成功")
        
        # 测试样例选择
        sample_vehicles = comparator.select_sample_vehicles(n_samples=3)
        print(f"[OK] 选择了 {len(sample_vehicles)} 个样例车辆")
        
        # 测试预测生成
        if len(sample_vehicles) > 0:
            predictions = comparator.generate_predictions(sample_vehicles)
            print(f"[OK] 生成了 {len(predictions)} 个预测结果")
        
        print("[OK] 预测对比功能测试通过")
        return True
        
    except Exception as e:
        print(f"[FAIL] 预测对比功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始数据分析功能测试...")
    
    # 测试结果
    results = []
    
    # 测试左转数据分析
    results.append(test_left_turn_analysis())
    
    # 测试预测对比
    results.append(test_prediction_comparison())
    
    # 输出总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] 所有数据分析功能测试通过!")
        return True
    else:
        print("[WARNING] 部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)