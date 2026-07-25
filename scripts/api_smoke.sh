#!/usr/bin/env bash
# M9 live smoke: boot the REST server (real backend + Ollama) and walk the acceptance
# path over HTTP. Unlike tests/test_api.py (fakes, deterministic), this proves the
# process actually serves — uvicorn startup, the lifespan backend build, real routing.
#
#   bash scripts/api_smoke.sh
#
# Runs the exmem backend offline (local Ollama only). Asserts HTTP success and response
# shape, not model-dependent supersession — a local model's predicate choice is its own
# variable (see the M8 notes), so this smokes the REST layer, not extraction quality.
set -euo pipefail

HOST=127.0.0.1
PORT="${MEMORY_API_PORT:-8077}"
BASE="http://${HOST}:${PORT}"
SCOPE="session:api-smoke-$$"

export MEMORY_BACKEND="${MEMORY_BACKEND:-exmem}"
export MEMORY_OFFLINE=1
export MEMORY_API_PORT="$PORT"

echo "== booting ${MEMORY_BACKEND} REST server on ${BASE} =="
uv run uvicorn local_infra.api.app:app --host "$HOST" --port "$PORT" --log-level warning &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for the lifespan to build the backend and the server to accept requests.
for _ in $(seq 1 60); do
  if curl -sf "${BASE}/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -sf "${BASE}/health" | grep -q '"ok":true' || { echo "FAIL: /health never came up"; exit 1; }
echo "[health] up"

echo "[1] POST /remember (real extraction — may pause on a cold model)"
R=$(curl -sf -X POST "${BASE}/remember" -H 'content-type: application/json' \
  -d "{\"content\":\"I work at Acme as a backend engineer.\",\"scope\":\"${SCOPE}\",\"extract\":\"sync\"}")
echo "  ${R}"
echo "${R}" | grep -q '"episode_id"' || { echo "FAIL: no episode_id in /remember response"; exit 1; }

echo "[2] GET /recall"
R=$(curl -sf --get "${BASE}/recall" --data-urlencode "q=where do I work" --data-urlencode "scope=${SCOPE}")
echo "  ${R}"

echo "[3] POST /improve"
R=$(curl -sf -X POST "${BASE}/improve" -H 'content-type: application/json' \
  -d "{\"scope\":\"${SCOPE}\",\"ops\":[\"reindex\"]}")
echo "  ${R}"
echo "${R}" | grep -q '"ops"' || { echo "FAIL: /improve response malformed"; exit 1; }

echo "[4] GET /health"
curl -sf "${BASE}/health" | grep -q "\"backend\":\"${MEMORY_BACKEND}\"" || { echo "FAIL: /health backend mismatch"; exit 1; }

echo "[5] POST /remember extract=async, then poll /recall until the worker extracts"
ASCOPE="session:api-smoke-async-$$"
R=$(curl -sf -X POST "${BASE}/remember" -H 'content-type: application/json' \
  -d "{\"content\":\"I work at Globex as a data engineer.\",\"scope\":\"${ASCOPE}\",\"extract\":\"async\"}")
echo "  ${R}"
echo "${R}" | grep -q '"deferred":true' || { echo "FAIL: async remember not deferred"; exit 1; }
echo "${R}" | grep -Eq '"job_id":"[0-9a-f]' || { echo "FAIL: async remember returned no job_id"; exit 1; }
HITS="[]"
for _ in $(seq 1 60); do
  HITS=$(curl -sf --get "${BASE}/recall" --data-urlencode "q=where do I work" --data-urlencode "scope=${ASCOPE}")
  if [ -n "${HITS}" ] && [ "${HITS}" != "[]" ]; then break; fi
  sleep 1
done
[ "${HITS}" != "[]" ] || { echo "FAIL: background worker never produced a recallable fact"; exit 1; }
echo "  worker drained the job: $(printf '%s' "${HITS}" | head -c 200)"

echo "== api smoke OK =="
