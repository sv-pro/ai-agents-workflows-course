#!/usr/bin/env bash

bench() {
  local m="$1"
  curl -s http://localhost:11434/api/generate \
    -d "{\"model\":\"$m\",\"prompt\":\"Benchmark me long enough.\",\"stream\":false,\
        \"options\":{\"num_ctx\":1536,\"num_predict\":512}}" \
  | jq -r '[.eval_count,.eval_duration] | @tsv' \
  | awk -v m="$m" 'BEGIN{FS="\t"} {printf "%-22s TPS: %.1f tok/s\n", m, $1/($2/1e9)}'
}

for m in gemma3:1b-it-qat gemma3:4b-it-qat llama3:latest gemma3n:e2b; do bench "$m"; done
