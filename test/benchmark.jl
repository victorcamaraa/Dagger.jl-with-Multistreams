# ══════════════════════════════════════════════════════════════════════════════
#  Unified Dagger.jl multi-stream scheduling benchmark  (CUDA + ROCm, one file)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Times ONLY the `spawn_datadeps` region; allocation and `collect` stay outside.
#  Writes median/min/IQR wall, busy_fraction (gpu_work/wall; >1 ⇒ streams overlap),
#  gflops and cross-stream event counts to
#  test/benchresults/unified_<backend>_<ts>.csv.
#
#  Run:  julia --project=test test/benchmark.jl
#        DAGGER_BENCH_BACKEND=CUDA|ROC|both selects the backend (default :auto).
# ══════════════════════════════════════════════════════════════════════════════

# Needed before AMDGPU loads on gfx1032; leave off for natively-supported archs.
get!(ENV, "HSA_OVERRIDE_GFX_VERSION", "10.3.0")
get!(ENV, "ROCBLAS_DEVICE_MEMORY_SIZE", "0")

const BACKEND = Symbol(get(ENV, "DAGGER_BENCH_BACKEND", "auto"))

# `:both` relaunches once per backend in separate processes, so each run's
# launch-bound timings are free of the other's host contention. The env guard
# stops the child from re-entering the driver.
if BACKEND === :both && !(get(ENV, "DAGGER_BENCH_BACKEND", "") in ("CUDA", "ROC"))
    proj = dirname(Base.active_project())
    for b in ("CUDA", "ROC")
        println("\n" * "█"^78 * "\n██  launching $b backend\n" * "█"^78)
        p = run(ignorestatus(addenv(`$(Base.julia_cmd()) --project=$proj $(@__FILE__)`,
                                    "DAGGER_BENCH_BACKEND" => b)))
        success(p) || @warn "$b backend exited with failure (continuing to next)"
    end
    exit(0)
end

using Dagger, LinearAlgebra, Statistics, Printf, Dates
import LinearAlgebra: BLAS

# ─── Load & verify the chosen backend ─────────────────────────────────────────
const _BK = Ref{Symbol}(:none)
if BACKEND in (:auto, :CUDA)
    try; @eval using CUDA;   CUDA.functional()   && (_BK[] = :CUDA); catch e; BACKEND === :CUDA && rethrow(e); end
end
if _BK[] === :none && BACKEND in (:auto, :ROC)
    try; @eval using AMDGPU; AMDGPU.functional() && (_BK[] = :ROC);  catch e; BACKEND === :ROC && rethrow(e); end
end
_BK[] === :none && error("No functional GPU backend for BACKEND=:$BACKEND (use :auto, :CUDA, :ROC, or :both).")

const B   = _BK[]
const GPU = B === :CUDA ? CUDA : AMDGPU                                   # only the taken side is evaluated
const EXT = B === :CUDA ? Base.get_extension(Dagger, :CUDAExt) : Base.get_extension(Dagger, :ROCExt)

# Function bodies resolve module names lazily, so the untaken branch is safe.
# Only the @elapsed MACRO must be built per-backend — hence the @eval below.
if B === :CUDA
    scope_for(d) = Dagger.scope(worker = 1, cuda_gpu = d + 1)   # CUDA scope keys are 1-based
    dev_id(p)    = p.device
    new_stream() = CUDA.CuStream()
    n_cus()      = Int(CUDA.attribute(CUDA.device(),
                       CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
else
    scope_for(d) = Dagger.scope(worker = 1, rocm_gpu = d)       # ROCm scope keys are raw device_id
    dev_id(p)    = p.device_id
    new_stream() = AMDGPU.HIPStream()
    n_cus()      = Int(AMDGPU.HIP.properties(AMDGPU.device()).multiProcessorCount)
end
@eval gpu_elapsed(f) = $(B === :CUDA ? :(CUDA.@elapsed f()) : :(AMDGPU.@elapsed f()))  # seconds

const PROC  = first(sort(collect(filter(p -> p isa Dagger.gpu_processor(Val(B)),
                                        Dagger.get_processors(Dagger.OSProc()))); by = dev_id))
const DEVID = dev_id(PROC)
const SCOPE = scope_for(DEVID)
const GPU_LABEL = replace(Dagger.short_name(PROC), ',' => ';')   # keep commas out of the CSV

gpu_sync() = (Dagger.gpu_synchronize(PROC); GPU.synchronize())
set_strategy!(s) = EXT.stream_strategy!(s)
clear_syncdeps!() = lock(EXT.SYNCDEPS) do m; empty!(m); end
read_events() = isdefined(EXT, :_EVENT_COUNT) ? getfield(EXT, :_EVENT_COUNT)[] : missing

# Rebuild the stream pool. Clears the producer→stream map first: stale
# (dev, stream_idx) entries could index past a now-shorter pool.
const _CUR_STREAMS = Ref(0)
function set_stream_count!(n::Int)
    n == _CUR_STREAMS[] && return n
    clear_syncdeps!()
    GPU.device!(EXT.DEVICES[DEVID])
    EXT.STREAMS[DEVID]       = [new_stream() for _ in 1:n]
    EXT.STREAM_QUEUES[DEVID] = [Threads.Atomic{Int}(0) for _ in 1:n]
    EXT.ROUNDROBIN[DEVID]    = Threads.Atomic{Int}(1)
    _CUR_STREAMS[] = n
    return n
end

# ═══ Configuration — H200 / MI300A stream-scheduling profile ══════════════════
# Auto-scaled from the device, so the same settings hold on an H200 (132 SMs), an
# MI300A (228 CUs) or a laptop card. ~5 min/backend. On an RTX 5060 Ti: roundrobin
# 776 → 206 → 148 ms across N=1/8/32, busy 0.97 → 5.10, locality flat at ~510 ms.
#
# :chainlink separates the policies (deep chain), :saturate gives the stream-count
# ceiling (no dependencies), :cholesky is the real-workload control — it runs
# Dagger's own factorization through real BLAS and so ignores TASK_OP entirely.
# For a real-gemm run: TASK_OP = :gemm and RUN_TILE_SWEEP = true.
const MATRIX        = 4096                       # square side
const TILE_SIZES    = [512, 1024, 2048]          # only used when RUN_TILE_SWEEP
const STREAM_COUNTS = [1, 2, 8, 32]              # 1 = single-stream (upstream) baseline
const STRATEGIES    = [:roundrobin, :random, :sdq, :locality]   # the comparison IS the point
const SHAPES        = [:linear, :diamond, :chainlink, :tangled, :cholesky] # [:matmul, :linear, :diamond, :chainlink, :tangled, :cholesky]
const SATURATE_K    = 16                         # K independent matmuls ⇒ 128 tasks at tile=2048,
                                                 # enough width to keep 32 streams busy

# ─── Per-task kernel ──────────────────────────────────────────────────────────
# :spin is an occupancy-controlled stand-in for a tile gemm — same DAG, same
# In/InOut dependencies, same events — using only SPIN_THREADS/256 workgroups.
# It exists because cuBLAS splits every gemm across all SMs, so no tile size
# leaves room for a second stream: on a 5060 Ti, 8 streams vs 1 gave 0.89x
# (1024²), 0.97x (2048²), 1.08x (512×4096) — never above ~1 — while a
# 4-workgroup kernel overlaps 7.98x. With :gemm no sweep setting can show a
# stream effect on hardware this size.
const TASK_OP      = :spin            # :gemm | :spin

# Tune these TOGETHER: SPIN_DIVISOR sets how many tasks *can* share the GPU,
# SPIN_TARGET_MS how many the host can keep in flight (overlap needs
# duration > streams / host_rate; the host submits ~5k tasks/s, so 32 streams
# need ≳6.4 ms). Occupancy sweep on 36 SMs: 4 workgroups 7.98x, 8 → 6.93x,
# 16 → 4.03x, 36 (full GPU) → 1.94x. Use 9 / 2.5 for a faster run topping out
# near 8 streams.
const SPIN_DIVISOR   = 32             # each task gets 1/SPIN_DIVISOR of the GPU
const SPIN_TARGET_MS = 8.0            # per-task duration to calibrate to
const SPIN_THREADS   = Ref(1024)
const SPIN_ITERS     = Ref(1_200_000)

# `A`/`B` are unused — they are here so the datadeps annotations match :gemm.
spin_op(x::Float32, iters::Int) = begin
    acc = x
    for _ in 1:iters
        acc = muladd(acc, 1.0000001f0, 1f-6)
    end
    acc
end
function spin_tile!(A, B, C)
    n, iters = SPIN_THREADS[], SPIN_ITERS[]
    v = view(vec(C), 1:n)
    v .= spin_op.(v, iters)
    return
end

# Size one spin task to SPIN_TARGET_MS on whatever GPU we are on.
function calibrate_spin!()
    TASK_OP === :spin || return
    GPU.device!(EXT.DEVICES[DEVID])
    cus = n_cus()
    SPIN_THREADS[] = 256 * max(1, cld(cus, SPIN_DIVISOR))
    probe = GPU.zeros(Float32, SPIN_THREADS[])
    SPIN_ITERS[] = 200_000
    spin_tile!(probe, probe, probe); GPU.synchronize()          # compile
    ms = minimum(gpu_elapsed(() -> (spin_tile!(probe, probe, probe); nothing)) for _ in 1:3) * 1000
    SPIN_ITERS[] = max(1, round(Int, 200_000 * SPIN_TARGET_MS / ms))
    GPU.unsafe_free!(probe)
    @printf("  :spin calibrated — %d CUs, %d workgroups (1/%d of GPU), %d iters ⇒ %.2f ms/task\n",
            cus, SPIN_THREADS[] ÷ 256, SPIN_DIVISOR, SPIN_ITERS[], SPIN_TARGET_MS)
end

const SAMPLES       = 20
const WARMUP        = 5                            # ≥5 tames the post-stream-rebuild median noise
const CHAIN_LEN     = 4                            # nodes in :linear, diamonds in :chainlink
const MAX_TASKS     = 20000                        # skip tile-512 DAGs (≥1536 tasks); keeps chainlink-1024 (576)

const RUN_TILE_SWEEP   = false                     # pointless under :spin (task cost is SPIN_ITERS,
                                                   # not tile); turn on with :gemm
const FIXED_STREAMS    = 8
const RUN_STREAM_SWEEP = true                      # streams × strategy @ FIXED_TILE (incl. saturate)
const FIXED_TILE       = 2048                      # ⇒ chainlink 96 / saturate 128 tasks

# ═══ DAG shapes — per-tile gemm into the current datadeps region ═══════════════
function dag_gemm!(C, A, B)                         # C := A·B, tile by tile (→ CUBLAS/rocBLAS)
    Ac, Bc, Cc = A.chunks, B.chunks, C.chunks
    Ant = size(Ac, 2)
    for n in axes(Cc, 2), m in axes(Cc, 1), k in 1:Ant
        if TASK_OP === :spin
            Dagger.@spawn spin_tile!(Dagger.In(Ac[m, k]), Dagger.In(Bc[k, n]),
                                     Dagger.InOut(Cc[m, n]))
        else
            beta = k == 1 ? 0f0 : 1f0
            Dagger.@spawn BLAS.gemm!('N', 'N', 1f0,
                Dagger.In(Ac[m, k]), Dagger.In(Bc[k, n]), beta, Dagger.InOut(Cc[m, n]))
        end
    end
end

# (nodes, critical-path in nodes, max independent nodes). :cholesky counts one
# node per panel step; its potrf chain is exactly that long.
function topo(shape, tile)
    p = cld(MATRIX, tile)
    shape === :matmul    ? (1, 1, 1) :
    shape === :linear    ? (CHAIN_LEN, CHAIN_LEN, 1) :
    shape === :diamond   ? (3, 2, 2) :
    shape === :chainlink ? (3 * CHAIN_LEN, 2 * CHAIN_LEN, 2) :
    shape === :tangled   ? (5, 3, 2) :
    shape === :saturate  ? (SATURATE_K, 1, SATURATE_K) :
    shape === :cholesky  ? (p, p, max(p - 1, 1)) :
    error("unknown shape $shape")
end

# gemm shapes are `nodes × (tiles/side)³`. :cholesky's right-looking loop
# (src/array/cholesky.jl) instead sums to binomial(p+2,3): p=4 ⇒ 20, p=8 ⇒ 120.
function n_tasks_for(shape, tile)
    p = cld(MATRIX, tile)
    shape === :cholesky && return binomial(p + 2, 3)
    return topo(shape, tile)[1] * p^3
end

# Useful FLOPs: 2·tile³ per gemm task, n³/3 for a Cholesky regardless of tiling.
# :cholesky never goes through `dag_gemm!`, so TASK_OP does not apply to it — it
# keeps the real gemm accounting even in a :spin run.
uses_spin(shape) = TASK_OP === :spin && shape !== :cholesky

flops_for(shape, tile) =
    uses_spin(shape)     ? n_tasks_for(shape, tile) * 2.0 * SPIN_THREADS[] * SPIN_ITERS[] :
    shape === :cholesky  ? MATRIX^3 / 3 :
                           n_tasks_for(shape, tile) * 2.0 * tile^3

# :cholesky seed — all-ones with a dominant diagonal: symmetric by construction
# (no cross-tile mirroring, no X[i,j]/X[j,i] aliasing) and positive definite.
# Values don't affect BLAS timing, but the factorization MUST complete: `_chol!`
# swallows PosDefException and returns early, silently timing an empty region.
fill_tile!(X, ondiag, boost) =
    (fill!(X, 1f0); ondiag && (view(X, diagind(X)) .+= boost); nothing)

function seed_spd!(A, boost)
    Ac = A.chunks
    Dagger.spawn_datadeps() do
        for n in axes(Ac, 2), m in axes(Ac, 1)
            Dagger.@spawn fill_tile!(Dagger.InOut(Ac[m, n]), m == n, boost)
        end
    end
    return A
end

# Pre-allocate everything OUTSIDE timing; return `run!` plus a `prep!` that
# restores any state `run!` destroys (also outside timing).
function make_shape(shape, tile)
    mats = DArray[]
    A!() = (x = rand(Blocks(tile, tile), Float32, MATRIX, MATRIX); push!(mats, x); x)
    prep! = () -> nothing

    run! =
        if shape === :matmul
            C, X, Y = A!(), A!(), A!()
            () -> Dagger.spawn_datadeps() do; dag_gemm!(C, X, Y) end
        elseif shape === :linear
            X = A!(); M = A!(); outs = [A!() for _ in 1:CHAIN_LEN]
            () -> Dagger.spawn_datadeps() do
                cur = X; for o in outs; dag_gemm!(o, cur, M); cur = o end
            end
        elseif shape === :diamond
            S = A!(); M1 = A!(); M2 = A!(); B1 = A!(); B2 = A!(); J = A!()
            () -> Dagger.spawn_datadeps() do
                dag_gemm!(B1, S, M1); dag_gemm!(B2, S, M2); dag_gemm!(J, B1, B2)
            end
        elseif shape === :chainlink
            S = A!(); M1 = A!(); M2 = A!()
            st = [(A!(), A!(), A!()) for _ in 1:CHAIN_LEN]
            () -> Dagger.spawn_datadeps() do
                cur = S
                for (b1, b2, sn) in st
                    dag_gemm!(b1, cur, M1); dag_gemm!(b2, cur, M2); dag_gemm!(sn, b1, b2); cur = sn
                end
            end
        elseif shape === :tangled
            S1 = A!(); S2 = A!(); T1 = A!(); T2 = A!(); U1 = A!(); U2 = A!(); J = A!()
            () -> Dagger.spawn_datadeps() do
                dag_gemm!(T1, S1, S2); dag_gemm!(T2, S2, S1)
                dag_gemm!(U1, T1, T2); dag_gemm!(U2, T2, T1); dag_gemm!(J, U1, U2)
            end
        elseif shape === :saturate
            tri = [(A!(), A!(), A!()) for _ in 1:SATURATE_K]
            () -> Dagger.spawn_datadeps() do
                for (C, X, Y) in tri; dag_gemm!(C, X, Y) end
            end
        elseif shape === :cholesky
            # Factorization overwrites its input and refactoring is not positive
            # definite, so `prep!` restores a pristine copy before every sample.
            A0 = seed_spd!(A!(), Float32(MATRIX))
            DA = A!()
            prep! = () -> copyto!(DA, A0)
            # `_chol!` is what `cholesky!` dispatches to after `ishermitian` — a
            # full O(n²) DArray pass on every call, which we must not time.
            () -> LinearAlgebra._chol!(DA, UpperTriangular)
        else
            error("unknown shape $shape")
        end

    free! = () -> foreach(Dagger.unsafe_free!, mats)
    return run!, free!, n_tasks_for(shape, tile), prep!
end

# ═══ Measurement ══════════════════════════════════════════════════════════════
const _TILE_MS = Dict{Tuple{Int,Bool},Float64}()
# Both kernels can be needed in one run, since :cholesky keeps gemm accounting.
function tile_kernel_ms(tile, spin::Bool)           # one standalone task, GPU time
    get!(_TILE_MS, (tile, spin)) do
        GPU.device!(EXT.DEVICES[DEVID])
        if spin
            s = GPU.ones(Float32, max(tile * tile, SPIN_THREADS[]))
            spin_tile!(s, s, s); GPU.synchronize()
            t = minimum(gpu_elapsed(() -> (spin_tile!(s, s, s); nothing)) for _ in 1:5) * 1000
            GPU.unsafe_free!(s)
            return t
        end
        a = GPU.ones(Float32, tile, tile); b = GPU.ones(Float32, tile, tile); c = GPU.ones(Float32, tile, tile)
        mul!(c, a, b); GPU.synchronize()            # compile
        t = minimum(gpu_elapsed(() -> (mul!(c, a, b); nothing)) for _ in 1:5) * 1000
        GPU.unsafe_free!(a); GPU.unsafe_free!(b); GPU.unsafe_free!(c)
        t
    end
end

iqr(x) = quantile(x, 0.75) - quantile(x, 0.25)

function time_region(run!, prep!)                   # times ONLY the datadeps region (+ device sync)
    for _ in 1:WARMUP; prep!(); run!(); gpu_sync() end
    ts = Float64[]
    for _ in 1:SAMPLES
        prep!(); gpu_sync()                         # restore destroyed inputs, untimed
        t0 = time_ns(); run!(); gpu_sync()
        push!(ts, (time_ns() - t0) / 1e6)           # ms
    end
    (median(ts), minimum(ts), iqr(ts))
end

# ═══ One measured config ══════════════════════════════════════════════════════
mutable struct Row
    sweep::String; shape::Symbol; tile::Int; streams::Int; strategy::Symbol
    nodes::Int; crit::Int; n_tasks::Int
    wall_med::Float64; wall_min::Float64; wall_iqr::Float64
    tile_ms::Float64; gpu_work::Float64; busy::Float64; gflops::Float64; gflops_min::Float64
    events::Union{Int,Missing}
end

function measure(sweep, shape, tile, streams, strat)
    nodes, crit, _ = topo(shape, tile)
    n_tasks = n_tasks_for(shape, tile)
    n_tasks > MAX_TASKS && return nothing           # skip pathological explosions (raise MAX_TASKS to include)

    set_stream_count!(streams); set_strategy!(strat); clear_syncdeps!()
    ev0 = read_events()

    med, mn, ir = Dagger.with_options(; scope = SCOPE) do
        run!, free!, _, prep! = make_shape(shape, tile)
        try; time_region(run!, prep!) finally free!() end
    end

    flops    = flops_for(shape, tile)
    tms      = tile_kernel_ms(tile, uses_spin(shape))
    # FLOPs → equivalent full-tile gemms, so one formula covers every shape
    # without inflating :cholesky, whose potrf/trsm/syrk tasks are cheaper.
    # :spin tasks are identical, so there `tms` already is the per-task cost.
    gpu_work = uses_spin(shape) ? tms * n_tasks : tms * flops / (2.0 * tile^3)
    busy     = gpu_work / med
    gflops     = flops / (med / 1000) / 1e9         # achieved FLOP/s at the median wall
    gflops_min = flops / (mn  / 1000) / 1e9         # peak FLOP/s at the best-case wall
    ev1      = read_events()
    # counter accrues over all WARMUP+SAMPLES region runs → normalize to per-region
    events   = (ev0 === missing || ev1 === missing) ? missing :
               round(Int, (ev1 - ev0) / (WARMUP + SAMPLES))

    Row(sweep, shape, tile, streams, strat, nodes, crit, n_tasks,
        med, mn, ir, tms, gpu_work, busy, gflops, gflops_min, events)
end

# ═══ Correctness (once) ═══════════════════════════════════════════════════════
function verify()
    set_stream_count!(FIXED_STREAMS); set_strategy!(:locality)
    if TASK_OP === :spin        # no product computed; cholesky check still applies
        println("  correctness (tiled matmul): SKIPPED — TASK_OP=:spin computes no product")
    end
    ok = TASK_OP === :spin ? true : Dagger.with_options(; scope = SCOPE) do
        A = rand(Blocks(512, 512), Float32, 1024, 1024)
        Bm = rand(Blocks(512, 512), Float32, 1024, 1024)
        C = rand(Blocks(512, 512), Float32, 1024, 1024)
        Ah, Bh = collect(A), collect(Bm)
        Dagger.spawn_datadeps() do; dag_gemm!(C, A, Bm) end
        r = collect(C) ≈ Ah * Bh
        foreach(Dagger.unsafe_free!, (A, Bm, C)); r
    end
    TASK_OP === :spin ||
        println("  correctness (:locality tiled matmul vs CPU): ", ok ? "PASS ✓" : "FAIL ✗")

    # A bad seed would look like a fast region, not an error — so check the factor.
    okc = Dagger.with_options(; scope = SCOPE) do
        DA = seed_spd!(rand(Blocks(128, 128), Float32, 512, 512), 512f0)
        H = ones(Float32, 512, 512); H[diagind(H)] .+= 512f0   # same matrix, host side
        # Full `cholesky!`, not `_chol!`: it also verifies the seed is hermitian.
        r = collect(cholesky!(DA).U) ≈ cholesky(H).U
        Dagger.unsafe_free!(DA); r
    end
    println("  correctness (:locality tiled cholesky vs CPU): ", okc ? "PASS ✓" : "FAIL ✗")
    ok && okc
end

# ═══ Output ═══════════════════════════════════════════════════════════════════
const CSV_HEADER = "backend,gpu,sweep,shape,matrix,tile,tiles_per_side,nodes,crit_path,n_tasks," *
                   "streams,strategy,samples,wall_median_ms,wall_min_ms,wall_iqr_ms," *
                   "tile_kernel_ms,gpu_work_ms,busy_fraction,gflops,gflops_min,events\n"

csv_line(r::Row) = @sprintf("%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%s,%d,%.4f,%.4f,%.4f,%.5f,%.2f,%.4f,%.2f,%.2f,%s\n",
    string(B), GPU_LABEL, r.sweep, r.shape, MATRIX, r.tile, cld(MATRIX, r.tile),
    r.nodes, r.crit, r.n_tasks, r.streams, r.strategy, SAMPLES,
    r.wall_med, r.wall_min, r.wall_iqr, r.tile_ms, r.gpu_work, r.busy, r.gflops, r.gflops_min,
    r.events === missing ? "" : string(r.events))

function print_row(i, n, r::Row)
    @printf("  [%3d/%3d] %-8s %-10s t=%-4d str=%-2d nodes=%-2d %-11s  %8.1f ms (min %7.1f)  busy=%5.2f  %8.1f GF%s\n",
        i, n, r.sweep, r.shape, r.tile, r.streams, r.nodes, r.strategy,
        r.wall_med, r.wall_min, r.busy, r.gflops,
        r.events === missing ? "" : @sprintf("  ev=%d", r.events))
    flush(stdout)                                   # show progress live even when redirected to a file
end

# How many configs will actually run (after MAX_TASKS skips) — denominator for i/N.
function total_configs()
    fits(shape, tile) = n_tasks_for(shape, tile) <= MAX_TASKS
    n = 0
    if RUN_TILE_SWEEP
        for shape in SHAPES, tile in TILE_SIZES, _ in STRATEGIES
            fits(shape, tile) && (n += 1)
        end
    end
    if RUN_STREAM_SWEEP
        for shape in vcat(SHAPES, :saturate), _ in STREAM_COUNTS, _ in STRATEGIES
            fits(shape, FIXED_TILE) && (n += 1)
        end
    end
    n
end

# ═══ Main ═════════════════════════════════════════════════════════════════════
function main()
    println("═"^90)
    println("  Dagger multi-stream benchmark  ·  backend=$B  ·  $(Dagger.short_name(PROC))")
    println("  Julia $(VERSION)  ·  Dagger $(pkgversion(Dagger))  ·  $(B == :CUDA ? "CUDA.jl" : "AMDGPU.jl") $(pkgversion(GPU))")
    println("  matrix=$(MATRIX)²  samples=$SAMPLES  warmup=$WARMUP  max_tasks=$MAX_TASKS  task_op=$TASK_OP")
    println("═"^90)
    calibrate_spin!()
    verify()

    mkpath(joinpath(@__DIR__, "benchresults"))
    path = joinpath(@__DIR__, "benchresults",
                    "unified_$(B)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv")
    io = open(path, "w"); write(io, CSV_HEADER)
    rows = Row[]
    total = total_configs()
    done = Ref(0)
    println("  running $total configs (any > $MAX_TASKS tasks are skipped)")
    emit(r) = (r === nothing && return; done[] += 1; push!(rows, r);
               write(io, csv_line(r)); flush(io); print_row(done[], total, r))

    if RUN_TILE_SWEEP
        println("\n── Tile-granularity sweep  (streams=$FIXED_STREAMS)  → host↔device crossover ──"); flush(stdout)
        for shape in SHAPES, tile in TILE_SIZES, strat in STRATEGIES
            emit(measure("tile", shape, tile, FIXED_STREAMS, strat))
        end
    end
    if RUN_STREAM_SWEEP
        println("\n── Stream-count sweep  (tile=$FIXED_TILE, incl. 1 = single-stream baseline) ──"); flush(stdout)
        for shape in vcat(SHAPES, :saturate), nstr in STREAM_COUNTS, strat in STRATEGIES
            emit(measure("streams", shape, FIXED_TILE, nstr, strat))
        end
    end

    close(io)
    println("\n", "═"^90)
    println("  $(length(rows)) configs measured.  CSV → $path")
    println("  Plot ideas: busy_fraction & gflops vs tile (crossover); gflops vs streams per strategy;")
    println("              gflops by strategy per shape.  busy>1 ⇒ overlap/device-bound, <1 ⇒ host-bound.")
    println("═"^90)
    return rows
end

main()
