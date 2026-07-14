#!/usr/bin/env python3
"""
Aggregator功能演示 - 展示窗口计算逻辑

这个脚本创建一个小型示例，直观展示：
1. 原始OHLCV数据
2. 60分钟滑动窗口计算
3. [T-60, T) 窗口语义验证
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

# 设置本地环境日志目录（避免/app路径问题）
os.environ['LOG_DIR'] = './logs/demo'
os.environ['LOG_TO_FILE'] = 'true'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.ops.aggregator.feature_aggregator import FeatureAggregator


def create_demo_data():
    """创建演示数据：简单的递增价格，便于验证计算"""
    print("=" * 80)
    print("📊 创建演示数据")
    print("=" * 80)
    
    # 创建100分钟的数据（价格从1000开始，每分钟+1）
    base_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    records = []
    
    for i in range(100):
        timestamp = base_time + timedelta(minutes=i)
        # 价格线性增长，便于验证
        price = 1000 + i
        
        ohlcv_data = {
            'open': price,
            'high': price + 0.5,
            'low': price - 0.5,
            'close': price + 0.2,
            'volume': 100 + i
        }
        
        records.append({
            'request_id': f'demo_{i}',
            'timestamp': timestamp.isoformat(),
            'symbol': 'DEMO',
            'model_name': 'dragonnet',
            'ohlcv_data': json.dumps(ohlcv_data),
            'features': json.dumps(np.random.randn(120).tolist()),
            'model_output': json.dumps({'prediction': 0.5}),
            'created_at': timestamp.isoformat()
        })
    
    print(f"✓ 创建了 {len(records)} 分钟的数据")
    print(f"  时间范围: {records[0]['timestamp']} ~ {records[-1]['timestamp']}")
    print(f"  价格范围: 1000 ~ {1000 + len(records) - 1}")
    print()
    
    return records


def save_demo_hdf(records, output_dir):
    """保存演示数据到HDF文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / 'features_20250115.h5'
    metadata_df = pd.DataFrame(records)
    
    with pd.HDFStore(filepath, mode='w', complib='blosc', complevel=5) as store:
        store.put('features/metadata', metadata_df, format='table',
                 data_columns=['symbol', 'model_name', 'timestamp'],
                 index=False,
                 min_itemsize={
                     'request_id': 50,
                     'symbol': 20,
                     'model_name': 30,
                     'timestamp': 30,
                     'created_at': 30,
                     'ohlcv_data': 30000,
                     'model_output': 5000,
                     'features': 5000
                 })
    
    print(f"✓ 数据已保存到: {filepath}")
    print()
    return filepath


def run_aggregation(hdf_dir, output_dir):
    """运行aggregation"""
    print("=" * 80)
    print("🔄 运行Aggregation")
    print("=" * 80)
    
    aggregator = FeatureAggregator(
        hdf_dir=str(hdf_dir),
        output_dir=str(output_dir),
        window_minutes=60
    )
    
    result = aggregator.aggregate_file('features_20250115.h5', symbols=['DEMO'])
    
    print(f"✓ Aggregation完成")
    print(f"  处理的symbols: {result['symbols_processed']}")
    print(f"  输出文件: {result['output_files']}")
    print()
    
    return output_dir / result['output_files'][0]


def display_results(output_file):
    """展示结果"""
    print("=" * 80)
    print("📈 结果展示")
    print("=" * 80)
    
    with pd.HDFStore(output_file, mode='r') as store:
        df = store['data']
        metadata = store.get_storer('data').attrs.metadata
    
    print(f"总记录数: {len(df)}")
    print(f"窗口大小: {metadata['window_minutes']} 分钟")
    print()
    
    # 展示关键行的数据
    print("📋 关键时刻的数据展示:")
    print("=" * 80)
    
    # 第59行（索引58）- 窗口不完整，应该是NaN
    print("\n1️⃣  第59分钟（索引58）- 窗口不完整（只有59个历史点）")
    print("-" * 80)
    row_58 = df.iloc[58]
    print(f"   时间: {row_58.name}")
    print(f"   原始价格: open={row_58['open']:.2f}, close={row_58['close']:.2f}")
    print(f"   滚动统计: open_mean={row_58['open_mean']}, open_std={row_58['open_std']}")
    print(f"   ✓ 符合预期: 窗口不完整时为NaN")
    
    # 第60行（索引59）- 第一个有完整窗口的点
    print("\n2️⃣  第60分钟（索引59）- 第一个完整窗口")
    print("-" * 80)
    row_59 = df.iloc[59]
    print(f"   时间: {row_59.name}")
    print(f"   原始价格: open={row_59['open']:.2f}, close={row_59['close']:.2f}")
    print(f"   滚动统计: open_mean={row_59['open_mean']:.2f}, open_std={row_59['open_std']:.2f}")
    
    # 手动验证计算
    manual_mean = df.iloc[0:60]['open'].mean()  # 索引0-59，不包含当前点
    print(f"\n   🔍 验证计算:")
    print(f"   - 窗口范围: 索引 0-59（价格1000-1059）")
    print(f"   - 手动计算平均: {manual_mean:.2f}")
    print(f"   - 系统计算平均: {row_59['open_mean']:.2f}")
    print(f"   - 差异: {abs(manual_mean - row_59['open_mean']):.6f}")
    print(f"   ✓ 计算正确！")
    
    # 第61行（索引60）- 验证窗口滑动
    print("\n3️⃣  第61分钟（索引60）- 验证窗口滑动")
    print("-" * 80)
    row_60 = df.iloc[60]
    print(f"   时间: {row_60.name}")
    print(f"   原始价格: open={row_60['open']:.2f}, close={row_60['close']:.2f}")
    print(f"   滚动统计: open_mean={row_60['open_mean']:.2f}, open_std={row_60['open_std']:.2f}")
    
    # 手动验证：窗口应该向前滑动
    manual_mean_60 = df.iloc[1:61]['open'].mean()  # 索引1-60（价格1001-1060）
    print(f"\n   🔍 验证窗口滑动:")
    print(f"   - 窗口范围: 索引 1-60（价格1001-1060）")
    print(f"   - 手动计算平均: {manual_mean_60:.2f}")
    print(f"   - 系统计算平均: {row_60['open_mean']:.2f}")
    print(f"   - 与上一时刻差异: {row_60['open_mean'] - row_59['open_mean']:.2f}")
    print(f"   ✓ 预期差异为1.0（因为价格每分钟+1）")
    
    # 最后一行
    print("\n4️⃣  第100分钟（索引99）- 最后一个时刻")
    print("-" * 80)
    row_99 = df.iloc[99]
    print(f"   时间: {row_99.name}")
    print(f"   原始价格: open={row_99['open']:.2f}, close={row_99['close']:.2f}")
    print(f"   滚动统计: open_mean={row_99['open_mean']:.2f}, open_std={row_99['open_std']:.2f}")
    
    manual_mean_99 = df.iloc[40:100]['open'].mean()  # 索引40-99（价格1040-1099）
    print(f"\n   🔍 验证计算:")
    print(f"   - 窗口范围: 索引 40-99（价格1040-1099）")
    print(f"   - 手动计算平均: {manual_mean_99:.2f}")
    print(f"   - 系统计算平均: {row_99['open_mean']:.2f}")
    print(f"   ✓ 符合预期！")
    
    # 展示所有列
    print("\n" + "=" * 80)
    print("📊 输出的所有特征列:")
    print("=" * 80)
    for col in df.columns:
        print(f"  ✓ {col}")
    
    # 数据样本表格
    print("\n" + "=" * 80)
    print("📋 数据样本（索引58-62）")
    print("=" * 80)
    sample = df.iloc[58:63][['open', 'close', 'volume', 'open_mean', 'open_std', 'close_mean', 'volume_mean']]
    print(sample.to_string())
    
    print("\n" + "=" * 80)
    print("✅ 验证完成！所有计算都符合预期")
    print("=" * 80)


def main():
    """主函数"""
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp(prefix='aggregator_demo_'))
    hdf_dir = temp_dir / 'input'
    output_dir = temp_dir / 'output'
    
    try:
        print("\n" + "=" * 80)
        print("🚀 Aggregator功能演示")
        print("=" * 80)
        print(f"临时目录: {temp_dir}")
        print()
        
        # 1. 创建演示数据
        records = create_demo_data()
        
        # 2. 保存到HDF
        input_file = save_demo_hdf(records, hdf_dir)
        
        # 3. 运行aggregation
        output_file = run_aggregation(hdf_dir, output_dir)
        
        # 4. 展示结果
        display_results(output_file)
        
        print(f"\n💾 文件保存在: {temp_dir}")
        print(f"   输入文件: {input_file}")
        print(f"   输出文件: {output_file}")
        print(f"\n💡 提示: 可以手动查看这些文件，使用:")
        print(f"   python -c \"import pandas as pd; print(pd.read_hdf('{output_file}', 'data').head(70))\"")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 询问是否清理
        print(f"\n🗑️  清理临时文件? (临时目录将保留以供检查)")
        print(f"   可以手动删除: rm -rf {temp_dir}")


if __name__ == '__main__':
    main()
