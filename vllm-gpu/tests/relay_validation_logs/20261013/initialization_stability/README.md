Repeated fresh-process initialization, real checkpoints
======================================================

CPU TP2 packed BF16 weights, CPU first 8 layers, concurrent AMX + H2D +
GPU model construction/warmup; V1 SHM eager FA3, 128 input / 8 output
tokens, chunked prefill budget 4096. Qwen3-32B GPU TP2 devices 0,4;
Qwen2.5-72B GPU TP4 devices 0,1,4,5. Each role uses its repository .venv.
The partition is diagnostic, not planner selected.

Measurement is verifier lazy_init_end minus lazy_init_start, including
RPC overhead. It excludes process/context startup and CPU weight packing;
weights are not required to finish loading before readiness. TTFT is from
the same model-creation call start to first returned token.

Root cause and fix
------------------
The shared shell LD_PRELOAD contains the CPU libiomp5.so runtime (twice).
With that preload, a GPU worker main thread was restricted to CPUs 0,96,
the siblings of one physical core. Linux thread schedstat measured
77.296 / 137.325 ms runqueue waiting during architecture creation, versus
108.504 / 108.623 ms thread CPU work. Wall times were 187.165 / 247.268 ms.
Removing only libiomp entries from the GPU child environment restored its
available affinity and reduced runqueue waits to microseconds. CPU AMX
preloads and binding remain unchanged; other GPU preloads are preserved.
Resident GPU launching was fixed in sllm commit 8e313e0. The validation
launcher applies the same environment fix. No CUDA fence or load ordering
was removed. Scheduler and warmup delays were also reduced.

32B baseline four call durations: 1.084929, 0.751159, 0.467612, 0.463526 s.
Fixed four: 0.264253, 0.188627, 0.176712, 0.190177 s. Baseline mean
0.691806 s, population standard deviation 0.255168 s, range 0.621403 s.
Fixed mean 0.204942 s, standard deviation 0.034637 s, range 0.087542 s.
The remaining difference has not been eliminated; four observations do
not establish a tail-latency guarantee.

Quiet-logging diagnostic: 0.598960 s; not adopted as a root-cause fix.
KMP_AFFINITY=disabled diagnostic crashed in Intel OpenMP before lazy init
(assertion kmp_affinity.cpp:4610); not adopted. Its logs are retained.
Older diagnostic command records lack inherited LD_PRELOAD/logging env;
subsequent records capture those fields explicitly.

Correctness
-----------
All successful relay runs compare 8 tokens against a full GPU prefill
on the same fully loaded SHM weights. This verifies relay continuity,
not an independent safetensors loader. No cross-chunk request is tested.
GPU focused tests: 54 passed. Resident client/controller: 48 passed,
including GPU preload isolation, parent environment and failure cleanup.
Raw role logs, exact commands, revisions and tracked patch snapshots are
alongside measurements.json. 72B clean-preload repetitions are recorded
in that JSON as they complete.

72B fixed two repetitions: lazy 0.317391 / 0.243950 s; TTFT
0.812046 / 0.739817 s. Both roles exit 0 and relay tokens match.
