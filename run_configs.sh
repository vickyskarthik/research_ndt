#!/bin/bash
# ============================================================
# PW-NDT C4/C5 Sequential Run Script
# Usage: bash run_configs.sh [SEQUENCE_NAME]
# Example: bash run_configs.sh 2019-01-10-11-46-21-radar-oxford-10k
# ============================================================

WS=/home/mcw/Desktop/MAY16PhD/PW-ndt/PW-ndt/catkin_ws
DATA_ROOT=/home/mcw/Desktop/MAY16PhD/PW-ndt/PW-ndt/dataset
POSES_DIR=${WS}/src/poses
RESULTS_DIR=${WS}/src/results/result/data

SEQ=${1:-2019-01-10-12-32-52-radar-oxford-10k}
RADAR_DIR=${DATA_ROOT}/${SEQ}/radar
GT_FILE=${POSES_DIR}/${SEQ}_gt.txt

echo "===================================================="
echo "  Sequence: ${SEQ}"
[ ! -d "${RADAR_DIR}" ] && echo "[ERROR] No radar dir: ${RADAR_DIR}" && exit 1
[ ! -f "${GT_FILE}"   ] && echo "[ERROR] No GT file:  ${GT_FILE}"   && exit 1

mkdir -p "${RESULTS_DIR}"
source /opt/ros/noetic/setup.bash
source ${WS}/devel/setup.bash
export OMP_NUM_THREADS=8

COMMON="__ns:=radar_odometry
  _directory:=${RADAR_DIR} _gt_directory:=${GT_FILE}
  _save_directory:=${RESULTS_DIR} _bag_name:=${SEQ}
  _global_viz:=0 _global_threshold:=70 _global_bias:=70 _global_mask_flag:=0
  _grid_size1:=9 _grid_size2:=6 _grid_size3:=4
  _eps1:=0.003 _eps2:=0.003 _eps3:=0.0003
  _max_step_size1:=0.03 _max_step_size2:=0.02 _max_step_size3:=0.01
  _epsilon_reg:=0.000001"

echo "[$(date +%H:%M:%S)] Config A — Baseline"
rosrun pw_ndt_radar_scan_matching radar_scan_matching ${COMMON} \
  _res_file_name:=_res_configA.txt _save_res:=1 \
  _log_covariance:=false _log_degeneracy:=false _use_constrained_update:=false _tau_degeneracy:=0.0 \
  >> /tmp/${SEQ}_configA.log 2>&1
echo "[$(date +%H:%M:%S)] Config A done — $(wc -l < ${RESULTS_DIR}/${SEQ}_res_configA.txt 2>/dev/null || echo 0) frames"

echo "[$(date +%H:%M:%S)] Config C — Logging only"
rosrun pw_ndt_radar_scan_matching radar_scan_matching ${COMMON} \
  _res_file_name:=_res_configC.txt _save_res:=0 \
  _log_covariance:=true _log_degeneracy:=true _use_constrained_update:=false _tau_degeneracy:=0.0 \
  >> /tmp/${SEQ}_configC.log 2>&1
echo "[$(date +%H:%M:%S)] Config C done — $(wc -l < ${RESULTS_DIR}/${SEQ}_configC_cov.csv 2>/dev/null || echo 0) cov rows"

echo "[$(date +%H:%M:%S)] B1 — Computing tau from lambda_min 5th percentile..."
TAU=$(python3 -c "
import csv,numpy as np
rows=list(csv.DictReader(open('${RESULTS_DIR}/${SEQ}_configC_deg.csv')))
pos=[float(r['lambda_min']) for r in rows if float(r['lambda_min'])>0]
print(f'{float(np.percentile(pos,5)) if pos else 0:.2f}')
")
echo "[$(date +%H:%M:%S)] tau_degeneracy = ${TAU}"

echo "[$(date +%H:%M:%S)] Config D — C5 constrained (tau=${TAU})"
rosrun pw_ndt_radar_scan_matching radar_scan_matching ${COMMON} \
  _res_file_name:=_res_configD.txt _save_res:=1 \
  _log_covariance:=true _log_degeneracy:=true _use_constrained_update:=true _tau_degeneracy:=${TAU} \
  >> /tmp/${SEQ}_configD.log 2>&1
echo "[$(date +%H:%M:%S)] Config D done — $(wc -l < ${RESULTS_DIR}/${SEQ}_res_configD.txt 2>/dev/null || echo 0) frames"

echo "===================================================="
echo "  ALL DONE: ${SEQ}"
echo "  Results: ${RESULTS_DIR}"
echo "===================================================="
touch /tmp/${SEQ}_all_configs_done.log
