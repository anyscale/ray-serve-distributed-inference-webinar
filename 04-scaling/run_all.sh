#!/usr/bin/env bash
# Checks for the scaling demos, one suite per job (entry point of job.yaml):
#   SUITE=ab       routing reuse A/B
#   SUITE=stall    stall check at 2/4/6/8 with the affinity router, then the default router
#   SUITE=itl      decode ITL, collocated vs disaggregated
#   SUITE=configs  config checks and the ai-dynamo import
#   SUITE=kvaware  KVAwareRouter: A/B, follow-ups and the stall check
#   SUITE=kvaware-evidence  KVAwareRouter follow-ups with INFO logs, to name the routing
#                  process. Submit both kvaware suites with
#                  --env RAY_SERVE_ENABLE_HA_PROXY=1 --env RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1
#                  --env RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY=1, so the
#                  Serve controller and its proxies start with them.
# Each step has a time limit, and a failed step does not stop the rest. Result
# lines go to one file that is printed last, after the Ray log noise.
set -u
cd "$(dirname "$0")"
SUITE=${SUITE:-ab}
RESULTS=/tmp/scaling-results.txt
: > "$RESULTS"
FAILED=0

step() {  # step <name> <limit-seconds> <command...>
  local name=$1 limit=$2; shift 2
  echo; echo "===== $name ($(date -u +%H:%M:%SZ))"
  timeout "$limit" "$@"; local rc=$?
  echo "$name rc=$rc" >> "$RESULTS"; [ "$rc" -ne 0 ] && FAILED=1
  return 0
}

# Log lines that show whether KV-aware routing was really active.
KV_EVIDENCE=("Direct streaming enabled" "Dynamo SelectionService created" "No KVTokenTracker"
  "ai-dynamo is not installed" "KV-aware routing is unavailable"
  "Could not derive a routing key from the request body" "Failed to get queue length")

kv_evidence() {  # kv_evidence <label>: per-node counts, appended to the results
  echo "  log evidence after $1:" >> "$RESULTS"
  python cluster_tools.py grep "${KV_EVIDENCE[@]}" >> "$RESULTS" 2>/dev/null
}

count_queue_warnings() {  # head-node Serve logs only; replicas on the GPU node are not counted
  local n
  n=$(cat /tmp/ray/session_latest/logs/serve/*.log 2>/dev/null | grep -c "Failed to get queue length")
  echo "  'Failed to get queue length' lines in head-node Serve logs so far: $n" >> "$RESULTS"
}

python -c "import ray, vllm; print('ray', ray.__version__, '| vllm', vllm.__version__)" \
  2>/dev/null | tee -a "$RESULTS"

case "$SUITE" in
  configs)
    step check-configs 600 bash -c "cd configs && python check_configs.py | tee -a $RESULTS"
    step ai-dynamo 600 bash -c 'python -m venv --system-site-packages /tmp/dyn \
      && /tmp/dyn/bin/pip install -q --no-deps "ai-dynamo-runtime>=1.4.0" \
      && /tmp/dyn/bin/python -c "import importlib.metadata as m; from dynamo.llm import SelectionService; print(\"dynamo.llm ok, ai-dynamo-runtime\", m.version(\"ai-dynamo-runtime\"))"' \
      | tee -a "$RESULTS"
    ;;
  ab)
    step wait-gpus 2800 python wait_for_gpus.py --gpus 4 --timeout-s 2700
    step routing-ab 1800 bash -c "cd routing && python verify.py --skip-stall --results-file $RESULTS"
    ;;
  stall)
    step wait-gpus 2800 python wait_for_gpus.py --gpus 4 --timeout-s 2700
    for router in affinity default; do
      step "stall-$router" 1500 bash -c \
        "cd routing && python verify.py --skip-reuse --stall-router $router --results-file $RESULTS"
      count_queue_warnings
    done
    ;;
  itl)
    step wait-gpus 2800 python wait_for_gpus.py --gpus 4 --timeout-s 2700
    for which in collocated disaggregated; do
      step "itl-$which" 1200 bash -c \
        "cd disaggregation && python deploy.py $which && python itl.py --label $which --results-file $RESULTS"
      serve shutdown -y >/dev/null 2>&1; sleep 5
    done
    ;;
  kvaware)
    for var in RAY_SERVE_ENABLE_HA_PROXY RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING \
               RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY; do
      echo "  $var=${!var:-unset}" >> "$RESULTS"
    done
    step wait-gpus 2800 python wait_for_gpus.py --gpus 4 --timeout-s 2700
    step ai-dynamo-all-nodes 600 bash -c \
      "python cluster_tools.py pip --no-deps 'ai-dynamo-runtime>=1.4.0' | tee -a $RESULTS"
    KV="cd routing && python verify.py --router kvaware --results-file $RESULTS"
    step kv-ab 1500 bash -c "$KV --skip-stall"
    kv_evidence kv-ab
    step kv-followups 1500 bash -c "$KV --skip-reuse --skip-stall --followups 8"
    step kv-stall 1500 bash -c "$KV --skip-reuse"
    step kv-stall-1slot 1500 bash -c "$KV --skip-reuse --max-ongoing-requests 1"
    kv_evidence kv-stall
    ;;
  kvaware-evidence)  # same env as kvaware; INFO logs show which process routes
    step wait-gpus 2800 python wait_for_gpus.py --gpus 4 --timeout-s 2700
    step ai-dynamo-all-nodes 600 bash -c \
      "python cluster_tools.py pip --no-deps 'ai-dynamo-runtime>=1.4.0' | tee -a $RESULTS"
    step kv-followups 1500 bash -c "cd routing && python verify.py --router kvaware \
      --skip-reuse --skip-stall --followups 8 --serve-log-level INFO --results-file $RESULTS"
    kv_evidence kv-followups-info
    ;;
  *) echo "unknown SUITE=$SUITE"; exit 2 ;;
esac

echo; echo "===== results (SUITE=$SUITE)"; cat "$RESULTS"
exit $FAILED
