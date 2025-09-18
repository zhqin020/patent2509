#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试程序（修复版）
测试代码实现框架中的模型是否能正常工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_model_import():
    """测试模型导入"""
    try:
        print("测试1: 模型导入...")
        from 代码实现框架 import LeftTurnPredictor, MockDataset
        print("[OK] 成功导入LeftTurnPredictor和MockDataset")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    try:
        print("测试2: 模型创建...")
        from 代码实现框架 import LeftTurnPredictor
        
        # 创建模型实例
        model = LeftTurnPredictor(
            visual_dim=64,
            motion_dim=40,
            traffic_dim=32
        )
        print("[OK] 模型创建成功")
        return True
    except Exception as e:
        print(f"[FAIL] 模型创建失败: {e}")
        return False

def test_dataset_creation():
    """测试数据集创建"""
    try:
        print("测试3: 数据集创建...")
        from 代码实现框架 import MockDataset
        
        # 创建数据集实例
        dataset = MockDataset(100)
        print(f"[OK] 数据集创建成功，包含 {len(dataset)} 个样本")
        return True
    except Exception as e:
        print(f"[FAIL] 数据集创建失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    try:
        print("测试4: 前向传播...")
        import torch
        from 代码实现框架 import LeftTurnPredictor, MockDataset
        
        # 创建模型和数据
        model = LeftTurnPredictor(
            visual_dim=64,
            motion_dim=40,
            traffic_dim=32
        )
        
        dataset = MockDataset(100)
        visual_feat, motion_feat, traffic_feat, intent_label, traj_label = dataset[0]
        
        # 添加batch维度
        visual_feat = visual_feat.unsqueeze(0)
        motion_feat = motion_feat.unsqueeze(0)
        traffic_feat = traffic_feat.unsqueeze(0)
        
        # 前向传播
        with torch.no_grad():
            intent_pred, traj_pred = model(visual_feat, motion_feat, traffic_feat)
        
        print(f"[OK] 前向传播成功")
        print(f"     意图预测形状: {intent_pred.shape}")
        print(f"     轨迹预测形状: {traj_pred.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] 前向传播失败: {e}")
        return False

def test_training_step():
    """测试训练步骤"""
    try:
        print("测试5: 训练步骤...")
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from 代码实现框架 import LeftTurnPredictor, MockDataset
        
        # 创建模型、数据和优化器
        model = LeftTurnPredictor(
            visual_dim=64,
            motion_dim=40,
            traffic_dim=32
        )
        
        dataset = MockDataset(20)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        intent_criterion = nn.CrossEntropyLoss()
        traj_criterion = nn.MSELoss()
        
        # 执行一个训练步骤
        model.train()
        for batch_idx, (visual_feat, motion_feat, traffic_feat, intent_label, traj_label) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播
            intent_pred, traj_pred = model(visual_feat, motion_feat, traffic_feat)
            
            # 计算损失
            intent_loss = intent_criterion(intent_pred, intent_label)
            traj_loss = traj_criterion(traj_pred, traj_label)
            total_loss = intent_loss + traj_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            print(f"[OK] 训练步骤成功，损失: {total_loss.item():.4f}")
            break  # 只测试一个batch
        
        return True
    except Exception as e:
        print(f"[FAIL] 训练步骤失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("车辆左转轨迹预测模型测试")
    print("=" * 60)
    
    # 测试列表
    tests = [
        test_model_import,
        test_model_creation,
        test_dataset_creation,
        test_forward_pass,
        test_training_step
    ]
    
    # 运行所有测试
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            if not result:
                print("测试失败，停止后续测试")
                break
        except Exception as e:
            print(f"[ERROR] 测试执行异常: {e}")
            results.append(False)
            break
        print()
    
    # 输出结果
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] 所有模型功能测试通过!")
        return True
    else:
        print("[WARNING] 部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)