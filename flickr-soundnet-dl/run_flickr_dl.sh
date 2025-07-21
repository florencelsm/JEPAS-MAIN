#!/usr/bin/env bash
# ------------------------------------------------------------
# run_flickr_dl.sh  (single-thread, 30-s progress)
# ------------------------------------------------------------

set -euo pipefail

##################### 必填参数 ##############################
URLS_FILE=${1:? "第 1 个参数：URL 列表文件"}
OUT_DIR=${2:? "第 2 个参数：输出目录"}
############################################################

LOG_FILE=flickr_dl.log
FAIL_LIST=failed_urls.txt

TOTAL=$(wc -l < "$URLS_FILE")
echo ">>> URL 总数: $TOTAL"

: > "$LOG_FILE"
: > "$FAIL_LIST"

# ----------------- 启动下载（单线程） ----------------------
python flickr.py \
  --ffmpeg  "$(which ffmpeg)" \
  --ffprobe "$(which ffprobe)" \
  --num-workers 1 \
  --num-retries 5 \
  --video-frame-rate 8 \
  "$URLS_FILE" "$OUT_DIR" >>"$LOG_FILE" 2>&1 &

DL_PID=$!

trap 'echo -e "\n收到中断信号，终止下载…"; kill "$DL_PID" 2>/dev/null; exit 130' INT

# ------------------- 进度监控循环 ------------------------
while kill -0 "$DL_PID" 2>/dev/null; do
  # 成功数 = 已下载 MP4 数
  success=$(ls "$OUT_DIR"/video/*.mp4 2>/dev/null | wc -l)

  # 失败数 = 日志中无法下载的 ID 去重计数
  grep -Eo 'Could not download video with Flickr ID [0-9]+' "$LOG_FILE" \
      | awk '{print $NF}' | sort -u > "$FAIL_LIST"
  fail=$(wc -l < "$FAIL_LIST")

  processed=$((success + fail))
  rate=$(( processed == 0 ? 0 : success * 100 / processed ))

  printf "\r成功: %d | 成功率: %d%%" "$success" "$rate"

  sleep 30
done

wait "$DL_PID" || true   # 捕获退出码不让脚本崩

# ------------------- 结束统计 ----------------------------
success=$(ls "$OUT_DIR"/video/*.mp4 2>/dev/null | wc -l)
grep -Eo 'Could not download video with Flickr ID [0-9]+' "$LOG_FILE" \
    | awk '{print $NF}' | sort -u > "$FAIL_LIST"
fail=$(wc -l < "$FAIL_LIST")
processed=$((success + fail))
rate=$(( processed == 0 ? 0 : success * 100 / processed ))

echo -e "\n\n>>> 下载完成！"
echo "成功: $success"
echo "成功率: $rate%"
echo "日志文件: $LOG_FILE"
echo "失败列表: $FAIL_LIST"
