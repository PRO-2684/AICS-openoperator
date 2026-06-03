# AGENTS.md
This is the only project guide you should obey, `README.md` is just a Problems Table with some outdated starting descriptions and tools.

## Load Check
When start from scratch, first output status exactly: AGENTS_OK_LOCAL.
When you get access to bangc mcp server tools, output BANGC_TOOLS_OK.

## Mission

This repository is a BangC MLU operator competition project. Implement and optimize missing `.mlu` operator files listed in the `README.md` `Problems` table.

Our gh repo is `PRO-2684/AICS-openoperator` and our teams' names are `team_42` and `team_91`, and my github user identity is `MosRat work@whl.moe`.

Hard rule: **do not use Torch/ATen/CNNL algorithm ops in the final submitted implementation**. Do not compute outputs with `torch::` , `at::` or `cnnlXXXX` operators. You can do anything without use computer libs, such as host-side tensor allocation, simple handwriting for loops for tiny calc or control, metadata access, descriptors, and runtime/library setup are allowed when they do not compute the operator result.

Optimize for the official Cambricon MLU370-X4SN OJ environment. Shape/dtype/layout specialization is expected when proven by official ref files.

## Default Workflow: Research → Implement → Submit → Continue → Collect → Optimize

**Dont use the decs and tools written in README.*md**, just follow the instructions in this file and user input. Always run the work as a pipeline, not as a blocking single attempt.

1. **Pick tasks**
   - Use the `README.md` `Problems` table as the task list.
   - Prefer red/unaccepted entries, low-effort library wins, or tasks with similar accepted implementations.
   - It is fine to work on several candidate problems in parallel when OJ is pending.

2. **Research before coding**
   - Read `ref/ref_files/%03d_opname.py` first. This is the strongest source for official inputs, shapes, dtype, layout, init args, constants, and output semantics.
   - Check `reference-impl/problems.json` and the target `.mlu` wrapper signature.
   - Search existing accepted `.mlu` files for similar patterns.
   - Use BangC MCP / docs / examples before guessing any BangC, CNRT, or runtime API behavior.

3. **Implement a first correct path**
   - Prefer the smallest correct implementation for the official fixed inputs.
   - Specialize aggressively for official shape/dtype/layout when justified by `ref/ref_files`.
   - Avoid broad generic kernels unless the official ref requires generality.

4. **Submit early, then continue thinking**
   - After a plausible implementation, set `config` to the `%03d` op id, commit, and push.
   - Record the short commit hash.
   - Do **not** idle while waiting for OJ. Immediately continue with another algorithm variant, another operator, local analysis, or documentation/API research.
   - Multiple attempts per operator and multiple operators in flight are encouraged.

5. **Collect OJ results in batch**
   - OJ is asynchronous and usually posts two comments/runs per commit with slight variance.
   - Batch query known commit hashes with `python ref/get_result.py <hash1> <hash2> ...`.
   - Compare PASS/FAIL, diff, latency, and variance across attempts.

6. **Optimize iteratively**
   - Keep a compact attempt table per active operator: commit, idea, PASS/FAIL, diff, latency, notes.
   - Use OJ feedback to decide whether to continue local tuning, try another algorithm, or switch tasks.
   - Try several independent variants when the search space is broad: direct BangC path, tiled path, reduction path, approximation path, different launch/task layouts.

## Parallelism Policy

Use both forms of parallelism:

- **Within one problem:** maintain several implementation attempts, e.g. hand-written kernel, reduced-precision variant, different tiling, different task mapping.
- **Across problems:** while one commit is waiting for OJ, work on another candidate problem instead of waiting.

Never wait passively for OJ. Submit, record hash, and continue.

## Official Sources and Priority

Priority order for understanding measured behavior:

1. `ref/ref_files/%03d_opname.py` — official measured inputs/reference behavior.
2. Target `.mlu` wrapper signature and current code.
3. `reference-impl/opname.py` — semantic reference but perhaps outdated.
4. `reference-impl/problems.json` — metadata and wrapper signature.
5. Existing accepted `.mlu` implementations.
6. BangC MCP, SDK docs, local headers, examples.

Do not modify `ref/bangc_torch_tester.py`, `ref/ref_files/`, SDK headers, benchmark scripts, or evaluator files to make a wrong implementation pass.

## OJ Submit and Result Commands

(These cmds here are for short, you may use `rtk` when possible.) 
Set the target op id in `config`, one `%03d` id per line, for example:

```text
041
```

Prefer `ref/oj_git.py` for non-webhook OJ submit. It writes `config`, stages, commits, pushes, prints compact JSON, and can optionally collect results. The current version uses a repo-local flock lock by default, supports `--pull --retry N` for remote push races, and has dry/status modes for cheap inspection.

Common submit:

```bash
python ref/oj_git.py -o 041 -m "041 Opname: brief implementation detail" --empty
```

Race-safe submit when other agents/users may also push:

```bash
python ref/oj_git.py -o 041 -m "041 Opname: brief implementation detail" --empty --pull --retry 3
```

Batch submit several ops in one config:

```bash
python ref/oj_git.py -o 041,042 -m "041 042 batch variants" --empty --pull --retry 3
```

Inspect without changing the repo:

```bash
python ref/oj_git.py --status
python ref/oj_git.py --dry -o 041 -m test --pull --retry 3
```

Equivalent manual submit:

```bash
git add -u && git add config && git commit -m "041 Opname: brief implementation detail" --allow-empty && git push
```

Record the short hash when needed:

```bash
git rev-parse --short HEAD
```

Submit and collect result:

```bash
python ref/oj_git.py -o 041 -m "041 Opname: brief implementation detail" --empty --result
```

Pass extra args to `ref/get_result.py` with repeated `--result-arg`; use `=` for values that start with `--`:

```bash
python ref/oj_git.py -o 041 -m "041 Opname" --empty --pull --retry 3 --result --sleep 8 --result-arg=--format --result-arg=text
```

Batch collect results:

```bash
python ref/get_result.py 8ee50f8 2fac8d0
```

Expected compact JSONL style:

```jsonl
{"c":"8ee50f8","ok":1,"rows":[{"p":"130_Attention_kv_cache","s":"0.648","acc":"PASS (diff=1.22e-04)","us":"133.800","ok":"✅"}]}
```

Interpret two OJ rows/comments per commit as normal variance. Compare both.


```bash
# ref/get_result.py
usage: get_result.py [-h] [--repo REPO] [--stdin] [--no-resolve] [--verbose] [--full]
                     [--max-output-chars MAX_OUTPUT_CHARS] [--format {jsonl,json,text}]
                     [commits ...]

Extract compact eval overview from GitHub commit comments.

positional arguments:
  commits               commit hashes/refs

options:
  -h, --help            show this help message and exit
  --repo REPO           OWNER/REPO; default: current gh repo
  --stdin               read commits from stdin, split by whitespace/comma
  --no-resolve          do not git rev-parse refs
  --verbose, -v         include sha/comment metadata
  --full                include raw **输出:** blocks from eval comments with STDOUT and STDERR
  --max-output-chars MAX_OUTPUT_CHARS
                        clip each --full output block; default: 20000
  --format {jsonl,json,text}
                        default: jsonl

# ref/oj_git.py
usage: oj_git.py [-h] [-o OP] [-m MSG] [-a [ADD ...]] [-A] [--empty]
                 [--no-push] [--pull] [--retry RETRY] [--lock LOCK]
                 [--lock-wait LOCK_WAIT] [--no-lock] [--no-verify] [--result]
                 [--result-arg RESULT_ARG] [--sleep SLEEP] [--dry] [--status]

agent-friendly OJ git submit helper

options:
  -h, --help            show this help message and exit
  -o OP, --op OP        write config ids, e.g. 041 or 041,042
  -m MSG, --msg MSG     commit message
  -a [ADD ...], --add [ADD ...]
                        pathspecs; default: git add -u plus config when --op
  -A, --all             git add -A
  --empty               allow empty commit
  --no-push
  --pull                git pull --rebase --autostash before changing/committing
  --retry RETRY         attempt count for pull/commit/push; useful with --pull
  --lock LOCK           lock file path; default: .git/oj_git.lock
  --lock-wait LOCK_WAIT seconds to wait for lock; -1 forever, 0 no wait
  --no-lock             disable flock
  --no-verify           pass --no-verify to git commit
  --result              run ref/get_result.py after push
  --result-arg RESULT_ARG
                        extra arg passed to ref/get_result.py; repeatable
  --sleep SLEEP         sleep before --result
  --dry
  --status              print compact git status summary and exit
```

Expected compact `ref/oj_git.py` JSON:

```jsonl
{"ok":1,"op":["041"],"c":"8ee50f8","cm":1,"ps":1}
```

Fields: `ok` success flag, `op` written config ids, `c` short HEAD hash, `cm` whether a commit was created, `ps` whether push ran, and `rrc` optional result command return code.


## Hardware Target

Official target: 10 x Cambricon MLU370-X4SN cards, but each operator should optimize for the current active device, not multi-card cooperation.

Known useful facts:

```text
Device family: MLU370
Clusters: 8
Mcores per cluster: 4
Useful task-friendly MLU cores: 32
NRAM per Mcore: 786432 bytes = 768 KB
WRAM per Mcore: 1048576 bytes = 1 MB
Global memory: about 24 GB
FP16: supported and common in OJ
BF16: supported
Driver: v6.2.10
Firmware: v1.1.6
```

Common launch patterns:

```mlu
k0<<<{1,1,1}, cnrtFuncTypeBlock, queue>>>       // simple serial/small task
k1<<<{32,1,1}, cnrtFuncTypeBlock, queue>>>      // simple 32-task parallel task
k2<<<{4,8,1}, cnrtFuncTypeUnion1, queue>>>      // 32-task parallel kernel with intra cluster sync, 4 for 4 cores per cluster and 8 for 8 clusters in MLU370.

// Union2/Union4/Union8 is rarely launched, which is only used in certain context for inter cluster sync. It has no direct performance benifit for switching from Union1.
```

For large contiguous FP16 pointwise/reduction workloads, start with 32 tasks unless shape or memory pattern suggests otherwise. Common FP16 tile candidates: 4096, 8192, 16384, 32768, 65536 elements, adjusted for scratch, alignment, and double buffering.

## Specialization Policy

Allowed when proven by official ref files/evaluator behavior:

- Hardcode tensor ranks, dimensions, strides, constants, dtype, contiguous layout, and launch layout.
- Specialize for FP16 when official inputs are FP16.
- Specialize for MLU370-X4SN task count, tiling, and NRAM capacity.
- Use fast paths for exact benchmark shapes.

Not allowed:

- Fake outputs or correctness bypasses.
- Torch/ATen algorithm ops to compute submitted results.
- Undefined behavior, invalid pointers, uninitialized memory, or evaluator-script modification.
- Multi-card assumptions unless the evaluator explicitly provides multi-card inputs.

> Competition Anonymous Rule: Preserve anonymity and avoid revealing benchmark-specific intent in the code. Do not add explanatory comments. Use professional, generic symbol names rather than names that expose shapes, tasks, or optimization assumptions. When fixed numeric constants are needed, write the computed values directly without documenting the derivation or linking them to benchmark dimensions. Avoid names, comments, or helper expressions that make hidden shape knowledge obvious; if a constant is sufficient, prefer the constant over a self-explaining formula. Keep the implementation clean, compact, and anonymous while preserving correctness and performance.

## BangC Guidance

Use MCP/docs for API details. Do not invent APIs.

Useful reminders:

- Some tutorial and guides are placed in `ref/` which may be useful for some ops.
- `__bang_write_zero` count is element count, not byte count.
- `__bang_reduce_sum` works on NRAM addresses and has 128-byte alignment/count constraints.
- `__bang_argmax` stores value and index contiguously.
- Vector fusion functions like `__bang_fcmpfilter` (vec element-wise comparison) and `__bang_fusion`(fused vec arithmetic calculation link FMA) use enumerate to dispatch detailed algorithm.
- `__bangc_matmul` and `__bangc_conv` use a specific WRAM layout for some args which you may refer to docs and other mlu files.
- Some useful guide and tutorials in tools and `ref/` could help you impl 3/4/5 stages pipeline, io sync and use cluster SRAM for mem bound ops.
- Prefer NRAM tiling, aligned transfers, reduced GDRAM round trips, and simple indexing.
- For memory-bound kernels, host overhead and extra GDRAM passes often dominate.


## RTK / Output Compression Policy

**Use RTK for save tokens.** Only use raw commands when you want compile or debug.

## Git and File Safety
**You can feel free modifying any local file that in .gitignore, including `ref/` py scripts and docs markdown files if needed according to your will.**

- The agent may freely commit, push, and merge changes when the diff contains only `.mlu` files.
- The agent may create OJ submission commits and push them without additional confirmation when working on implementation attempts.
- Do not push/pull/rebase/reset/merge changes that include non-`.mlu` files unless explicitly requested, except for the final README update rule below.
- Do not run destructive cleanup commands unless explicitly allowed and clearly scoped.
- Keep implementation changes focused on the target `.mlu` files and directly necessary local helpers.

### Final README Problems Table Update

At final wrap-up, after an operator is completed and an OJ result has been obtained, the agent must automatically update the `README.md` `Problems` table submission/commit field when appropriate.

Before editing `README.md`, the agent must check:

1. The problem difficulty.
2. The current assigned owner/author in the `Problems` table.
3. The best OJ result obtained for this problem in the current work.
4. Whether that result is the historical best among the agent's known attempts/results.

Rules:

- If the problem difficulty is `basic` and the current owner/author is not `@MosRat`, the agent must not claim or overwrite the README entry.
- In that case, if the agent has already modified the README entry, it must restore the README submission/commit field to its previous value.
- Otherwise, when the completed attempt achieves the best known result for that problem, the agent must update the `Problems` table submission/commit field to point to the best commit.
- README edits are only allowed for the `Problems` table and only for this final result-recording purpose.
- Do not edit any other README content.

## Reporting Format

When reporting progress or completion, be compact and data-driven:

```text
Op: 041_Opname
Files: Opname.mlu
Source: ref/ref_files/041_Opname.py
Specialization: fp16, shape/layout/constant assumptions
Submit: <short_hash> if pushed
OJ: PASS/FAIL, diff, latency us, score if available
Next: keep/tune/abandon/switch task
```

For multiple attempts, use a compact table:

```text
hash     op    idea              result              latency     note
8ee50f8  130   tiled kv path      PASS diff=1.22e-04  133.8 us    keep
2fac8d0  130   smaller tile       PASS diff=6.10e-05  130.6 us    best
```

## Completion Checklist

Before calling an operator done:

- Official `ref/ref_files/%03d_opname.py` behavior was implemented.
- Wrapper signature stayed unchanged.
- Nontrivial APIs were verified through MCP/docs/examples.
- Local build/benchmark was run when possible, or clearly disclosed if not.
- OJ hash and results were recorded when submitted.
- No unauthorized evaluator/ref/SDK/README/script changes were made.

## Something leaderboard and oj
**You prefer to use scripts rather than fetch these apis directly for save tokens, if they dont meet your requirements, just modify them.**

The OJ system use a 3 queues arch `commit -> task_queue -> dispatch compile and run in avaliable workers (may blocked by aboteurs who submit too many submissions or submit too many questions at once) -> result_queue -> github commit and leaderboard_queue ——> leaderboard consumes the queue per 5 minutes`.

Use `ref/get_oj_status.py` to get oj queue status.

Use `ref/get_tasks.py` for offline task metadata. It emits compact JSONL by default with `p` problem id, `n` name, `d` difficulty, and `c` category/domain, and supports id/range/category/search queries, for example:

```bash
python ref/get_tasks.py 039-043
python ref/get_tasks.py --cat matrix --diff medium
python ref/get_tasks.py --search masked --wrap --dtype
```

Use `ref/get_leaderboard.py` to get leaderboard rows. It now joins task metadata from `ref/get_tasks.py` by default, so compact rows include `d` difficulty and `cat` category/domain in addition to leaderboard fields. Use `--no-task-meta` only when you need the old minimum-token output.

Common leaderboard queries:

```bash
python ref/get_leaderboard.py all --mode leaders
python ref/get_leaderboard.py all --team team_91 --team team_42 --mode other-leaders # prefer this to get targets in loop pipeline
python ref/get_leaderboard.py all --team team_91 --team team_42 --mode team-best
python ref/get_leaderboard.py all --team team_91 --team team_42 --mode best-excl-team 
python ref/get_leaderboard.py 039-043 --mode top --top 2 --fmt tsv
```

Compact leaderboard fields: `p` problem id, `n` name if `--name`, `r` rank when useful, `u` user/team, `g` repo if `--gh`, `s` score, `us` latency, `t` compact timestamp, `d` difficulty, `cat` category/domain, `lu` last update if `--lu`, `c` result count if `--count`, and `st/h/e` status/error fields.

Note that leaderboard updates may lag commit OJ comments because the leaderboard consumes its queue periodically. Go forward while waiting; do not block on a single leaderboard refresh. Feel free to change these tool files for more info and features as needed.

## Compile and OJ
**Dont use outdated fixed `check.sh` or other local tools.** Local hardware is different from the OJ system. Use fast oj system instead of compiling locally with slow hardware. Only locally compile mlu when oj cant provide clear infomations or you want to profile kernels for finding bottleneck.
