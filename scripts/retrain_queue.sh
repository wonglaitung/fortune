#!/bin/bash
# 生产模型重训 + 校验队列（严格串行，避免 OOM）
cd /data/fortune
L=output/train_retrain

# 等 20d 重训结束
while tmux has-session -t jtr20 2>/dev/null; do sleep 30; done

# 1) 5d / 1d CatBoost 重训（保持原训练窗口：start_date 默认=近4年，只修对齐 bug）
echo "[$(date '+%F %T')] 5d 重训开始" >> $L/queue.log
python3 ml_services/ml_trading_model.py --mode train --horizon 5 --model-type catboost \
        --use-feature-selection >> $L/5d.log 2>&1
echo "[$(date '+%F %T')] 5d 完成 rc=$?" >> $L/queue.log

echo "[$(date '+%F %T')] 1d 重训开始" >> $L/queue.log
python3 ml_services/ml_trading_model.py --mode train --horizon 1 --model-type catboost \
        --use-feature-selection >> $L/1d.log 2>&1
echo "[$(date '+%F %T')] 1d 完成 rc=$?" >> $L/queue.log

# 2) 用修复后代码重跑绝对标签 walk-forward → 新 CSV（GATE_SNAPSHOT 数据源）
for h in 20 5 1; do
  extra=""
  [ "$h" = "20" ] && extra="--learner lightgbm"
  echo "[$(date '+%F %T')] abs${h}d walk-forward 开始" >> $L/queue.log
  bash scripts/run_walk_forward.sh --model-type catboost --horizon $h \
       --use-feature-selection $extra --label-mode absolute \
       --start-date 2019-06-01 --end-date 2026-07-31 \
       > output/wf${h}d_abs.log 2>&1
  echo "[$(date '+%F %T')] abs${h}d 完成 rc=$?" >> $L/queue.log
done

echo "[$(date '+%F %T')] 队列全部完成" >> $L/queue.log
echo QUEUE_DONE
