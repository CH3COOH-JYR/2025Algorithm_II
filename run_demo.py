#!/usr/bin/env python3
"""
图结构挖掘系统演示启动脚本
"""
import os
import sys
import subprocess

def check_requirements():
    """检查依赖是否安装"""
    try:
        import networkx
        import matplotlib
        import numpy
        import tkinter
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_test_data():
    """检查测试数据是否存在"""
    if os.path.exists("test_datasets") and os.listdir("test_datasets"):
        print("✓ 测试数据已存在")
        return True
    else:
        print("⚠ 测试数据不存在，正在生成...")
        try:
            subprocess.run([sys.executable, "generate_test_data.py"], check=True)
            print("✓ 测试数据生成完成")
            return True
        except subprocess.CalledProcessError:
            print("✗ 测试数据生成失败")
            return False

def main():
    """主函数"""
    print("=" * 60)
    print("图结构挖掘系统 - 算法课期末大作业")
    print("=" * 60)
    
    print("\n正在检查环境...")
    
    # 检查依赖
    if not check_requirements():
        return
    
    # 检查测试数据
    if not check_test_data():
        return
    
    print("\n选择运行模式:")
    print("1. 交互式可视化界面 (推荐)")
    print("2. 命令行批处理模式")
    print("3. 小图测试")
    print("4. 退出")
    
    while True:
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == "1":
            print("\n启动交互式可视化界面...")
            try:
                subprocess.run([sys.executable, "interactive_visualization.py"])
            except KeyboardInterrupt:
                print("\n程序已退出")
            break
            
        elif choice == "2":
            print("\n启动命令行批处理模式...")
            try:
                subprocess.run([sys.executable, "graph_system_no_gui.py"])
            except KeyboardInterrupt:
                print("\n程序已退出")
            break
            
        elif choice == "3":
            print("\n运行小图测试...")
            try:
                subprocess.run([sys.executable, "test_small_graph.py"])
            except KeyboardInterrupt:
                print("\n程序已退出")
            break
            
        elif choice == "4":
            print("退出程序")
            break
            
        else:
            print("无效选择，请输入 1-4")

if __name__ == "__main__":
    main() 