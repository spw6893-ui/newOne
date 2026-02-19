#!/bin/bash
set -e

echo "=========================================="
echo "AlphaGen Crypto Factor Mining - One-Click"
echo "=========================================="
echo ""

# 训练预设（可选）
# - baseline: 保持脚本/代码默认值（最接近“原始能跑通”的行为）
# - explore20: 特征降维(Top20 IC) + 加大训练强度 + 放宽 pool（用于提升探索效率）
# - explore20_icir: 在 explore20 基础上，使用 ICIR 目标(MeanStdAlphaPool) + 动态阈值 + 长度惩罚
# - explore20_faststable: 在 explore20 基础上，偏“更快 + 更稳 + 更好泛化”的组合（推荐日常跑）
# - explore20_lcb: 走“思路2”：Pool 目标改为 LCB(mean - beta*std)，更偏泛化稳定性（冲更高 OOS IC）
# - explore20_ucblcb: 参考 AlphaQCM 的“方差引导探索”直觉：beta 从负到正（先 UCB 探索，后 LCB 稳健）
PRESET="${ALPHAGEN_PRESET:-baseline}"
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "用法:"
    echo "  ./run_training.sh [baseline|explore20|explore20_icir|explore20_faststable|explore20_lcb|explore20_ucblcb]"
    echo ""
    echo "示例:"
    echo "  ./run_training.sh"
    echo "  ./run_training.sh explore20"
    echo "  ./run_training.sh explore20_icir"
    echo "  ./run_training.sh explore20_faststable"
    echo "  ./run_training.sh explore20_lcb"
    echo "  ./run_training.sh explore20_ucblcb"
    echo ""
    echo "说明:"
    echo "  也可以用环境变量覆盖任意 ALPHAGEN_* 参数（例如 ALPHAGEN_TOTAL_TIMESTEPS）。"
    exit 0
fi
if [ -n "${1:-}" ]; then
    PRESET="$1"
fi

export_default () {
    # 仅当变量未设置时写入默认值（方便用户外部覆盖）
    local name="$1"
    local value="$2"
    if [ -z "${!name+x}" ]; then
        export "$name=$value"
    fi
}

# 可选：指定 Python 解释器（强烈建议用 venv/conda 的 python）
PYTHON="${PYTHON:-python3}"
PIP="$PYTHON -m pip"

# 确保 alphagen 子模块已拉取（否则会出现 `No module named alphagen.rl.env`）
if [ ! -f "alphagen/requirements.txt" ]; then
    echo "检测到 alphagen 子模块未就绪，正在初始化..."
    git submodule update --init --recursive
    echo "✓ alphagen 子模块就绪"
    echo ""
fi

# 配置参数
TRAIN_END="2024-01-01 00:00:00+00:00"
VAL_END="2024-07-01 00:00:00+00:00"
# 允许外部覆盖（例如你只复原到了 final_dataset/ 目录）
INPUT_DIR="${INPUT_DIR:-AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
# 训练 CSV 的 schema 策略：
# - per-file：保持每个币自己的列（默认，兼容旧输出）
# - union：全币种并集（“全因子”）
# - intersection：全币种交集（NaN 最少，适合先跑通）
SCHEMA_MODE="${SCHEMA_MODE:-per-file}"

# Step 0: 选择训练预设（只设置“未显式 export”的变量）
echo "训练预设: $PRESET"
case "$PRESET" in
    baseline)
        # 不强行设置任何 ALPHAGEN_*，完全跟随 train_alphagen_crypto.py 默认/你外部 export 的值
        ;;
    explore20)
        # 方案A：特征降维 + 增强探索（你发的建议）
        export_default ALPHAGEN_FEATURES_MAX 20
        export_default ALPHAGEN_FEATURES_PRUNE_CORR 0.95
        export_default ALPHAGEN_TOTAL_TIMESTEPS 800000
        export_default ALPHAGEN_BATCH_SIZE 256
        export_default ALPHAGEN_N_STEPS 4096
        export_default ALPHAGEN_POOL_CAPACITY 20
        export_default ALPHAGEN_IC_LOWER_BOUND 0.005
        export_default ALPHAGEN_POOL_L1_ALPHA 0.001
        # 避免频繁 KL early-stop 把更新截断；如需更稳可改成 0.1/0.2
        export_default ALPHAGEN_TARGET_KL none
        ;;
    explore20_icir)
        # 核心：reward 引导 + pool 目标从 IC 改为 ICIR + 动态阈值（先松后紧）
        export_default ALPHAGEN_FEATURES_MAX 20
        export_default ALPHAGEN_FEATURES_PRUNE_CORR 0.95
        export_default ALPHAGEN_TOTAL_TIMESTEPS 800000
        export_default ALPHAGEN_BATCH_SIZE 256
        export_default ALPHAGEN_N_STEPS 4096
        export_default ALPHAGEN_POOL_CAPACITY 20
        export_default ALPHAGEN_POOL_TYPE meanstd
        # MeanStdAlphaPool: 默认 lcb_beta=none => optimize ICIR(mean/std)
        export_default ALPHAGEN_POOL_LCB_BETA none
        export_default ALPHAGEN_POOL_L1_ALPHA 0.001
        # 动态阈值：初期更容易入池，后期更严格
        export_default ALPHAGEN_IC_LOWER_BOUND_START 0.005
        export_default ALPHAGEN_IC_LOWER_BOUND_END 0.02
        export_default ALPHAGEN_IC_LOWER_BOUND_UPDATE_EVERY 2048
        # 表达式长度惩罚（鼓励简洁/更早 SEP）
        export_default ALPHAGEN_REWARD_PER_STEP -0.001
        export_default ALPHAGEN_TARGET_KL none
        ;;
    explore20_faststable)
        # 目标：在不牺牲泛化的前提下，显著缓解“后期越跑越慢”，并降低 PPO 抖动（clip_fraction/approx_kl 偏高）
        export_default ALPHAGEN_FEATURES_MAX 20
        export_default ALPHAGEN_FEATURES_PRUNE_CORR 0.95
        export_default ALPHAGEN_TOTAL_TIMESTEPS 800000

        # PPO：大 batch + 大 rollout + 较少 epoch（更快也更稳）
        export_default ALPHAGEN_BATCH_SIZE 512
        export_default ALPHAGEN_N_STEPS 8192
        export_default ALPHAGEN_N_EPOCHS 10
        export_default ALPHAGEN_LEARNING_RATE 0.0001
        export_default ALPHAGEN_CLIP_RANGE 0.2
        export_default ALPHAGEN_TARGET_KL none

        # Pool：不把阈值抬到过高（避免后期入池停滞），并限制优化步数避免“200k 后慢到爆”
        export_default ALPHAGEN_POOL_TYPE mse
        export_default ALPHAGEN_POOL_CAPACITY 20
        export_default ALPHAGEN_IC_LOWER_BOUND_START 0.005
        export_default ALPHAGEN_IC_LOWER_BOUND_END 0.02
        export_default ALPHAGEN_IC_LOWER_BOUND_UPDATE_EVERY 10000
        export_default ALPHAGEN_POOL_L1_ALPHA 0.005
        export_default ALPHAGEN_POOL_OPT_MAX_STEPS 1000
        export_default ALPHAGEN_POOL_OPT_TOLERANCE 100

        # 防止策略学会超早 SEP（导致评估次数爆炸）：
        # - 冷启动先允许短表达式（学会“能生成合法表达式 + 会 SEP”）
        # - 训练中后期逐步抬高最小长度（降低评估频率，显著缓解越跑越慢）
        export_default ALPHAGEN_MIN_EXPR_LEN_START 1
        export_default ALPHAGEN_MIN_EXPR_LEN_END 8
        export_default ALPHAGEN_REWARD_PER_STEP 0

        # 子表达式库（突破平台期的关键手段之一）：让 agent 直接选择常用子结构再组合
        # 子表达式库加大一些（配合 train_alphagen_crypto.py 的“结构优先”库构建策略，能更明显突破平台期）
        export_default ALPHAGEN_SUBEXPRS_MAX 80
        export_default ALPHAGEN_SUBEXPRS_WINDOWS "5,10,20,40"
        export_default ALPHAGEN_SUBEXPRS_DTS "1,2,4,8"

        # 缓存与周期评估（评估太频繁会拖慢整体）
        export_default ALPHAGEN_ALPHA_CACHE_SIZE 128
        export_default ALPHAGEN_EVAL_EVERY_STEPS 50000
        export_default ALPHAGEN_EVAL_TEST 1
        ;;
    explore20_lcb)
        # 思路2：Pool objective 从“均值 IC”切到“更稳的下置信界”(LCB = mean - beta * std)
        # 目的：更偏向 OOS（val/test）稳定性，通常能把平台期的泛化能力往上抬。
        #
        # 注意：这是“训练信号/优化目标”的变化，不是纯调参；reward 会变成 LCB/ICIR，而不是单纯 IC。
        export_default ALPHAGEN_FEATURES_MAX 20
        export_default ALPHAGEN_FEATURES_PRUNE_CORR 0.95
        export_default ALPHAGEN_TOTAL_TIMESTEPS 800000

        # PPO：尽量稳（避免抖动导致 val/test flat）
        export_default ALPHAGEN_BATCH_SIZE 512
        export_default ALPHAGEN_N_STEPS 8192
        export_default ALPHAGEN_N_EPOCHS 10
        export_default ALPHAGEN_LEARNING_RATE 0.0001
        export_default ALPHAGEN_CLIP_RANGE 0.2
        export_default ALPHAGEN_TARGET_KL none
        export_default ALPHAGEN_ENT_COEF 0

        # Pool：meanstd + LCB
        export_default ALPHAGEN_POOL_TYPE meanstd
        export_default ALPHAGEN_POOL_CAPACITY 30
        export_default ALPHAGEN_POOL_L1_ALPHA 0.001
        # LCB beta：越大越保守（更压 std），一般 0.3~1.0 之间试；可外部覆盖
        export_default ALPHAGEN_POOL_LCB_BETA 0.5

        # 动态阈值：先松后紧（避免早期卡死）
        export_default ALPHAGEN_IC_LOWER_BOUND_START 0.005
        export_default ALPHAGEN_IC_LOWER_BOUND_END 0.02
        export_default ALPHAGEN_IC_LOWER_BOUND_UPDATE_EVERY 10000

        # 速度/可训练性：冷启动短，后期加长减少评估频率
        export_default ALPHAGEN_STACK_GUARD 1
        export_default ALPHAGEN_MIN_EXPR_LEN_START 1
        export_default ALPHAGEN_MIN_EXPR_LEN_END 10
        export_default ALPHAGEN_MIN_EXPR_LEN_UPDATE_EVERY 20000
        export_default ALPHAGEN_REWARD_PER_STEP 0

        # 子表达式库：配合“结构优先”构建策略，突破平台期
        export_default ALPHAGEN_SUBEXPRS_MAX 80
        export_default ALPHAGEN_SUBEXPRS_RAW_MAX 10
        export_default ALPHAGEN_SUBEXPRS_WINDOWS "5,10,20,40"
        export_default ALPHAGEN_SUBEXPRS_DTS "1,2,4,8"

        # cache & eval
        export_default ALPHAGEN_ALPHA_CACHE_SIZE 256
        export_default ALPHAGEN_EVAL_EVERY_STEPS 50000
        export_default ALPHAGEN_EVAL_TEST 1
        ;;
    explore20_ucblcb)
        # 参考 AlphaQCM：把“不确定性”当作探索红利。
        # 这里用 LCB beta < 0 的阶段近似 UCB(mean + |beta|*std) 来鼓励探索高方差表达式，
        # 再逐步把 beta 调到 >0（LCB）以追求更稳健的 OOS IC。
        export_default ALPHAGEN_FEATURES_MAX 20
        export_default ALPHAGEN_FEATURES_PRUNE_CORR 0.95
        export_default ALPHAGEN_TOTAL_TIMESTEPS 800000

        export_default ALPHAGEN_BATCH_SIZE 512
        export_default ALPHAGEN_N_STEPS 8192
        export_default ALPHAGEN_N_EPOCHS 10
        export_default ALPHAGEN_LEARNING_RATE 0.0001
        export_default ALPHAGEN_CLIP_RANGE 0.2
        export_default ALPHAGEN_TARGET_KL none
        export_default ALPHAGEN_ENT_COEF 0

        export_default ALPHAGEN_POOL_TYPE meanstd
        export_default ALPHAGEN_POOL_CAPACITY 30
        export_default ALPHAGEN_POOL_L1_ALPHA 0.001

        # beta schedule：UCB -> LCB（关键）
        export_default ALPHAGEN_POOL_LCB_BETA_START -0.5
        export_default ALPHAGEN_POOL_LCB_BETA_END 0.5
        export_default ALPHAGEN_POOL_LCB_BETA_UPDATE_EVERY 10000

        export_default ALPHAGEN_IC_LOWER_BOUND_START 0.005
        export_default ALPHAGEN_IC_LOWER_BOUND_END 0.02
        export_default ALPHAGEN_IC_LOWER_BOUND_UPDATE_EVERY 10000

        export_default ALPHAGEN_STACK_GUARD 1
        export_default ALPHAGEN_MIN_EXPR_LEN_START 1
        export_default ALPHAGEN_MIN_EXPR_LEN_END 10
        export_default ALPHAGEN_MIN_EXPR_LEN_UPDATE_EVERY 20000
        export_default ALPHAGEN_REWARD_PER_STEP 0

        export_default ALPHAGEN_SUBEXPRS_MAX 80
        export_default ALPHAGEN_SUBEXPRS_RAW_MAX 10
        export_default ALPHAGEN_SUBEXPRS_WINDOWS "5,10,20,40"
        export_default ALPHAGEN_SUBEXPRS_DTS "1,2,4,8"

        export_default ALPHAGEN_ALPHA_CACHE_SIZE 256
        export_default ALPHAGEN_EVAL_EVERY_STEPS 50000
        export_default ALPHAGEN_EVAL_TEST 1
        ;;
    *)
        echo "❌ 未知 PRESET: $PRESET（支持 baseline / explore20 / explore20_icir / explore20_faststable / explore20_lcb / explore20_ucblcb）"
        exit 1
        ;;
esac

# 检查数据目录（Vision 基底 + metrics 覆盖完整 85 币种）
if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A "$INPUT_DIR"/*_final.csv 2>/dev/null)" ]; then
    echo "❌ 最终宽表目录不存在或为空：$INPUT_DIR"
    echo "   请先生成 Vision 基底数据集（final_dataset_vision_metrics85）"
    exit 1
fi

echo "✓ 训练数据已就绪"
echo ""

# Step 1: 准备AlphaGen训练数据
echo "=========================================="
echo "Step 1: 准备AlphaGen训练数据"
echo "=========================================="

INPUT_DIR_FOR_PREP="$INPUT_DIR"
if [[ "$INPUT_DIR_FOR_PREP" == AlphaQCM/* ]]; then
    INPUT_DIR_FOR_PREP="${INPUT_DIR_FOR_PREP#AlphaQCM/}"
fi

if [ "$FORCE_REBUILD" = "1" ] || [ ! -d "AlphaQCM/AlphaQCM_data/alphagen_ready" ] || [ -z "$(ls -A AlphaQCM/AlphaQCM_data/alphagen_ready 2>/dev/null)" ]; then
    echo "准备训练数据..."
    $PYTHON AlphaQCM/data_collection/prepare_alphagen_training_data.py \
        --input-dir "$INPUT_DIR_FOR_PREP" \
        --output-dir AlphaQCM_data/alphagen_ready \
        --horizon-hours 1 \
        --filter-quality \
        --impute ffill \
        --ffill-limit 24 \
        --schema "$SCHEMA_MODE" \
        --train-end "$TRAIN_END" \
        --val-end "$VAL_END"
    echo "✓ 数据准备完成"
else
    echo "✓ AlphaGen训练数据已存在，跳过准备步骤"
fi

echo ""

# Step 2: 检查依赖
echo "=========================================="
echo "Step 2: 检查依赖"
echo "=========================================="

if ! $PYTHON -c "import torch, gymnasium, stable_baselines3, sb3_contrib" 2>/dev/null; then
    echo "安装依赖（最小集，避免 alphagen/requirements.txt 的过老版本）..."
    $PIP install -q -U pip setuptools wheel
    $PIP install -q -r requirements_alphagen_train.txt
    echo "✓ 依赖安装完成"
else
    echo "✓ 依赖已安装"
fi

echo ""

# Step 3: 开始训练
echo "=========================================="
echo "Step 3: 开始AlphaGen训练"
echo "=========================================="
echo "训练配置:"
echo "  - 训练集: 2020-01-01 -> $TRAIN_END"
echo "  - 验证集: $TRAIN_END -> $VAL_END"
echo "  - 测试集: $VAL_END -> 2025-02-15"
echo ""
echo "AlphaGen 关键参数（可通过环境变量覆盖）:"
echo "  - ALPHAGEN_FEATURES_MAX=${ALPHAGEN_FEATURES_MAX:-0}"
echo "  - ALPHAGEN_TOTAL_TIMESTEPS=${ALPHAGEN_TOTAL_TIMESTEPS:-100000}"
echo "  - ALPHAGEN_BATCH_SIZE=${ALPHAGEN_BATCH_SIZE:-128}"
echo "  - ALPHAGEN_N_STEPS=${ALPHAGEN_N_STEPS:-2048}"
echo "  - ALPHAGEN_POOL_CAPACITY=${ALPHAGEN_POOL_CAPACITY:-10}"
echo "  - ALPHAGEN_IC_LOWER_BOUND=${ALPHAGEN_IC_LOWER_BOUND:-0.01}"
echo "  - ALPHAGEN_POOL_TYPE=${ALPHAGEN_POOL_TYPE:-mse}"
echo "  - ALPHAGEN_IC_LOWER_BOUND_START=${ALPHAGEN_IC_LOWER_BOUND_START:-${ALPHAGEN_IC_LOWER_BOUND:-0.01}}"
echo "  - ALPHAGEN_IC_LOWER_BOUND_END=${ALPHAGEN_IC_LOWER_BOUND_END:-${ALPHAGEN_IC_LOWER_BOUND:-0.01}}"
echo "  - ALPHAGEN_REWARD_PER_STEP=${ALPHAGEN_REWARD_PER_STEP:-0}"
echo "  - ALPHAGEN_ALPHA_CACHE_SIZE=${ALPHAGEN_ALPHA_CACHE_SIZE:-64}"
echo "  - ALPHAGEN_MIN_EXPR_LEN=${ALPHAGEN_MIN_EXPR_LEN:-1}"
echo "  - ALPHAGEN_SUBEXPRS_MAX=${ALPHAGEN_SUBEXPRS_MAX:-0}"
echo "  - ALPHAGEN_POOL_OPT_LR=${ALPHAGEN_POOL_OPT_LR:-5e-4}"
echo "  - ALPHAGEN_POOL_OPT_MAX_STEPS=${ALPHAGEN_POOL_OPT_MAX_STEPS:-10000}"
echo "  - ALPHAGEN_POOL_OPT_TOLERANCE=${ALPHAGEN_POOL_OPT_TOLERANCE:-500}"
echo "  - ALPHAGEN_EVAL_EVERY_STEPS=${ALPHAGEN_EVAL_EVERY_STEPS:-0}"
echo "  - ALPHAGEN_EVAL_TEST=${ALPHAGEN_EVAL_TEST:-1}"
echo "  - ALPHAGEN_POOL_L1_ALPHA=${ALPHAGEN_POOL_L1_ALPHA:-0.005}"
echo "  - ALPHAGEN_TARGET_KL=${ALPHAGEN_TARGET_KL:-none}"
echo ""
echo "监控训练进度:"
echo "  tensorboard --logdir=./alphagen_output/tensorboard"
echo ""

# libgomp 对 OMP_NUM_THREADS 要求是正整数；有些平台会注入非整数值导致告警
if [ -n "${OMP_NUM_THREADS:-}" ] && ! [[ "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
    echo "⚠ OMP_NUM_THREADS=${OMP_NUM_THREADS} 不是整数，已 unset（避免 libgomp 警告）"
    unset OMP_NUM_THREADS
fi

$PYTHON train_alphagen_crypto.py

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  - 模型: ./alphagen_output/model_final.zip"
echo "  - 因子池: ./alphagen_output/alpha_pool.json"
echo "  - 验证结果: ./alphagen_output/validation_results.json"
echo ""
echo "查看结果:"
echo "  cat ./alphagen_output/alpha_pool.json"
echo "  cat ./alphagen_output/validation_results.json"
