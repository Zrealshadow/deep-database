#!/bin/bash
# TP-BERTa 预训练模型下载脚本

set -e  # 遇到错误立即退出

TPBERTA_DIR="../tp-berta"
CHECKPOINTS_DIR="$TPBERTA_DIR/checkpoints"

echo "=========================================="
echo "TP-BERTa 预训练模型下载"
echo "=========================================="

# 创建目录
mkdir -p "$CHECKPOINTS_DIR"
cd "$CHECKPOINTS_DIR"

# 检查是否已存在
if [ -f "tp-joint/pytorch_models/best/pytorch_model.bin" ]; then
    echo "✅ 模型已存在，跳过下载"
    exit 0
fi

# 检查 gdown 是否安装
if ! command -v gdown &> /dev/null; then
    echo "📦 安装 gdown..."
    pip install gdown
fi

# 下载模型
echo "📥 下载 TP-BERTa 联合预训练模型..."
gdown https://drive.google.com/uc?id=1ArjkOAblGPErmxUyVIfpiM0IztnjjYxq -O tp-joint.tar.gz

# 解压
echo "📦 解压模型文件..."
tar -xzf tp-joint.tar.gz

# 重命名目录（如果需要）
if [ -d "tp-joint"* ] && [ ! -d "tp-joint" ]; then
    mv tp-joint* tp-joint
fi

# 清理
rm -f tp-joint.tar.gz

# 验证
if [ -f "tp-joint/pytorch_models/best/pytorch_model.bin" ]; then
    echo "✅ 下载完成！"
    echo "   模型位置: $(pwd)/tp-joint/"
    ls -lh tp-joint/pytorch_models/best/pytorch_model.bin
else
    echo "❌ 下载失败，请检查"
    exit 1
fi

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
