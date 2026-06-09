# ==============================================================================
# refactored_parameter_homotopy.jl
# Riemannian random walk on the space of quintic threefolds, searching for
# parameters with a high number of real lines. Tracks all N_LINES = 2875
# complex solutions via homotopy continuation and certifies real counts.
# ==============================================================================

using Random, HomotopyContinuation, Serialization, LinearAlgebra, DelimitedFiles

# ==============================================================================
# --- CONSTANTS & CONFIGURATION ---
# ==============================================================================

# Each worker writes to its own CSV to avoid concurrent-write races.
# Merge output files with: cat master_lines_registry_*.csv > final_massive_registry.csv
# Strip duplicate headers with: awk 'NR==1 || !/^Real_Lines/' final_massive_registry.csv > clean_registry.csv
const WORKER_ID         = Random.randstring(RandomDevice(), 8)
const MASTER_CSV        = "master_lines_registry_$(WORKER_ID).csv"

const BETA              = 0.82   # Momentum retention in Riemannian step
const N_PARAMS          = 126    # Monomials of degree 5 in 5 variables
const N_LINES           = 2875   # Bézout-theoretic complex line count
const INITIAL_STEP      = 0.12
const MIN_STEP          = 5e-8
const STEP_DECAY        = 0.925
const STAGNATION_CAP    = 2500
const SEED_MAX_ATTEMPTS = 30     # Each attempt calls full_solve (~minutes); cap keeps hangs bounded

@enum StepOutcome success tracker_fail cert_fail parity_fail

# ==============================================================================
# --- RIEMANNIAN GEOMETRY ---
# ==============================================================================

full_solve(system, p) = solve(system; start_system=:total_degree, target_parameters=p, show_progress=true)

# Projects v onto the tangent space of the sphere at base.
# Returns nothing if the projection is degenerate (v nearly parallel to base).
function project_tangent(v, base)::Union{Vector{Float64}, Nothing}
    proj = v .- dot(v, base) .* base
    n    = norm(proj)
    return n < 1e-12 ? nothing : proj ./ n
end

# Resamples random tangent vectors until a non-degenerate one is found.
# Used wherever a valid tangent direction is unconditionally required.
function safe_project_tangent(base; max_attempts::Int=10)::Vector{Float64}
    for _ in 1:max_attempts
        r = project_tangent(randn(Float64, N_PARAMS), base)
        r !== nothing && return r
    end
    error("safe_project_tangent: failed after $max_attempts attempts — severe numerical problem.")
end

# Moves point p along the great circle in direction v by step_size.
# Re-normalizes output to prevent floating-point drift off the sphere.
function exp_map_sphere(p, v, step_size)
    norm_v = norm(v)
    norm_v < 1e-12 && return normalize(p)
    v_dir = v ./ norm_v
    theta = step_size * norm_v
    return normalize(p .* cos(theta) .+ v_dir .* sin(theta))
end

# Carries momentum vector m from the tangent space at p1 to the tangent space at p2.
# Re-projects the result onto the tangent space at p2 to correct for sphere curvature.
# Falls back to projecting the original m, then to a fresh draw, if transport is degenerate.
function parallel_transport_sphere(p1, p2, m)
    delta         = p2 .- p1
    norm_delta_sq = dot(delta, delta)
    norm_delta_sq < 1e-12 && return m
    transported = m .- (2.0 * dot(p2, m) / norm_delta_sq) .* delta
    result = project_tangent(transported, p2)
    if result === nothing
        @warn "parallel_transport_sphere: degenerate transport; falling back."
        fallback = project_tangent(m, p2)
        return fallback !== nothing ? fallback : safe_project_tangent(p2)
    end
    return result
end

# Computes a new sphere position and transported momentum using Riemannian SGD.
# The step direction blends current momentum (weight BETA) with fresh tangent noise.
function get_riemannian_step(current_params, momentum, current_step)
    noise = safe_project_tangent(current_params)
    blend = (BETA .* momentum) .+ ((1.0 - BETA) .* noise)
    v     = project_tangent(blend, current_params)
    if v === nothing
        @warn "get_riemannian_step: degenerate blend; falling back to pure noise."
        v = noise
    end
    new_params   = exp_map_sphere(current_params, v, current_step)
    new_momentum = parallel_transport_sphere(current_params, new_params, v)
    return new_params, new_momentum
end

# ==============================================================================
# --- SYSTEM & REGISTRY ---
# ==============================================================================

# Builds the polynomial system whose solutions are lines on a quintic threefold.
# A line is parametrized as u*v1 + v*v2 in P^4; substituting into the quintic
# and extracting [u,v] coefficients gives 6 equations in 6 unknowns (alpha).
# The 126 quintic coefficients (c) are the homotopy parameters.
function build_system()
    @var x[1:5] u v alpha[1:6] c[1:N_PARAMS]
    monoms    = monomials(x, 5:5)
    f_on_line = subs(
        sum(c[i] * monoms[i] for i in 1:length(monoms)),
        x => u .* [1, 0, alpha[1], alpha[2], alpha[3]] + v .* [0, 1, alpha[4], alpha[5], alpha[6]]
    )
    return System(coefficients(f_on_line, [u, v]); variables=alpha, parameters=c)
end

function log_to_master(real_count, params)
    open(MASTER_CSV, "a") do io
        writedlm(io, hcat(real_count, transpose(Float64.(real.(params)))), ',')
    end
end

function initialize_registry_if_needed()
    isfile(MASTER_CSV) && return
    println("Registry not found. Creating $MASTER_CSV with header...")
    @var x[1:5]
    clean_strs = replace.(
        string.(monomials(x, 5:5)),
        r"[₁₂₃₄₅]" => s -> Dict("₁"=>"1","₂"=>"2","₃"=>"3","₄"=>"4","₅"=>"5")[s]
    )
    open(MASTER_CSV, "w") do io
        writedlm(io, reshape(["Real_Lines"; clean_strs], 1, :), ',')
    end
end

# ==============================================================================
# --- HOMOTOPY TRACKING ---
# ==============================================================================

function track_homotopy_leg(system, start_sols, start_p, target_p, t_options)
    res = solve(
        system, start_sols;
        start_parameters  = start_p,
        target_parameters = target_p,
        show_progress     = false,
        tracker_options   = t_options
    )
    return [solution(r) for r in res if is_success(r)]
end

# Module-level constant: pure value derived from literals, no reason to reconstruct per call.
const TRACKER_OPTIONS = TrackerOptions(
    max_steps          = 50_000,
    extended_precision = true,
    parameters         = :conservative
)

# Tracks solutions from start_params to target_params via a complex lift stratagem.
# Lift tiers are sorted descending so each retry uses a smaller imaginary perturbation,
# stepping progressively closer to the real target slice.
function parameter_solve(system, start_solutions, start_params, target_params, base_lift_magnitude)
    lift_tiers = sort([base_lift_magnitude, 0.05, 0.002], rev=true)
    for current_lift in lift_tiers
        lift_dir            = safe_project_tangent(target_params)
        intermediate_params = target_params .+ (current_lift * im) .* lift_dir
        sols_inter = track_homotopy_leg(
            system, start_solutions, start_params, intermediate_params, TRACKER_OPTIONS
        )
        if length(sols_inter) == N_LINES
            final_sols = track_homotopy_leg(
                system, sols_inter, intermediate_params, target_params, TRACKER_OPTIONS
            )
            length(final_sols) == N_LINES && return final_sols
        end
    end
    return Vector{Vector{ComplexF64}}()
end

# ==============================================================================
# --- SEED MANAGEMENT ---
# ==============================================================================

# Re-certifies a loaded (params, sols) pair against the live system.
# Guards against seeds from prior buggy runs or version-mismatched checkpoints.
function verify_seed(system, params, sols)::Int
    length(sols) == N_LINES || error("Seed has $(length(sols)) solutions, expected $N_LINES.")
    cert        = certify(system, sols; target_parameters=params)
    n_certified = count(is_certified, certificates(cert))
    n_certified == N_LINES || error("Seed: only $n_certified/$N_LINES solutions certified.")
    real_count  = ndistinct_real_certified(cert)
    isodd(real_count) || error("Seed has even real count ($real_count); parity violated.")
    return real_count
end

function generate_certified_seed(system)
    for attempt in 1:SEED_MAX_ATTEMPTS
        println("Seed attempt $attempt/$SEED_MAX_ATTEMPTS...")
        params     = normalize(randn(Float64, N_PARAMS))
        result     = full_solve(system, params)
        valid_sols = solutions(result)
        if length(valid_sols) == N_LINES
            cert = certify(system, valid_sols; target_parameters=params)
            if count(is_certified, certificates(cert)) == N_LINES
                real_count = ndistinct_real_certified(cert)
                isodd(real_count) && return params, valid_sols, real_count
            end
        end
        println("  Failed. Re-rolling...")
    end
    error("Failed to generate a certified seed after $SEED_MAX_ATTEMPTS attempts.")
end

# Loads and re-verifies an existing checkpoint, or generates a fresh certified seed.
# On any deserialization or verification failure, the corrupt file is deleted and
# a fresh seed is generated rather than propagating bad state into the walk.
# Checkpoints are written atomically via serialize-to-tmp then rename.
function initialize_or_load_seed(system, filename)
    if isfile(filename)
        raw = try
            deserialize(filename)
        catch e
            @warn "Deserialization of $filename failed: $e — regenerating."
            rm(filename, force=true)
            nothing
        end

        if raw !== nothing
            params, sols, count_val = raw
            params = normalize(params)
            verified_count = try
                verify_seed(system, params, sols)
            catch e
                @warn "Seed re-verification failed: $e — regenerating."
                rm(filename, force=true)
                nothing
            end

            if verified_count !== nothing
                if verified_count != count_val
                    @warn "Stored count ($count_val) differs from certified ($verified_count); using certified."
                    count_val = verified_count
                end
                println("Loaded verified seed: $filename (real lines: $count_val).")
                return params, sols, count_val
            end
        end
    end

    println("Generating fresh certified seed for $filename...")
    params, valid_sols, real_count = generate_certified_seed(system)
    log_to_master(real_count, params)
    tmp_file = filename * ".tmp"
    serialize(tmp_file, (params, valid_sols, real_count))
    mv(tmp_file, filename, force=true)
    return params, valid_sols, real_count
end

# ==============================================================================
# --- STATE OBJECT ---
# ==============================================================================

"""
    SearchState

Full state of the Riemannian random walk on S^125 (unit sphere in R^126).

Invariants:
- `params` is always unit-normalized.
- `momentum` is always in the tangent space of `params`; enforced by
  parallel_transport_sphere and by in-place zeroing on failures/regressions.
- `step` decays on all failures; resets to INITIAL_STEP on any accepted step.
- `stagnation` counts all iterations since the last upward tick, including
  lateral drifts and failures, to prevent indefinite plateau wandering.
- All momentum updates use .= (in-place) to avoid heap allocation in the inner loop.
"""
mutable struct SearchState
    params    ::Vector{Float64}
    solutions ::Vector{Vector{ComplexF64}}
    real_count::Int
    momentum  ::Vector{Float64}
    step      ::Float64
    stagnation::Int
end

# ==============================================================================
# --- STEP LOGIC ---
# ==============================================================================

function propose_step(state::SearchState)
    new_params, new_momentum = get_riemannian_step(state.params, state.momentum, state.step)
    lift = max(0.01, state.step * 0.5)
    return new_momentum, new_params, lift
end

# Tracks to new_params, certifies all N_LINES solutions, and checks the parity invariant.
# start_params is passed explicitly to avoid a hidden closure dependency on state.
function track_and_verify(system, start_params, new_params, state_solutions, lift)
    sols = parameter_solve(system, state_solutions, start_params, new_params, lift)
    length(sols) != N_LINES && return tracker_fail, sols, nothing
    cert        = certify(system, sols; target_parameters=new_params)
    n_certified = count(is_certified, certificates(cert))
    n_certified != N_LINES && return cert_fail, sols, nothing
    real_count = ndistinct_real_certified(cert)
    !isodd(real_count) && return parity_fail, sols, real_count
    return success, sols, real_count
end

function attempt_step(system, state::SearchState)
    direction, new_params, lift = propose_step(state)
    outcome, sols, real_count  = track_and_verify(
        system, state.params, new_params, state.solutions, lift
    )
    return outcome, direction, new_params, sols, real_count
end

# ==============================================================================
# --- ACCEPTANCE / UPDATE POLICY ---
# ==============================================================================

# Accepts a step that is an upward tick or lateral drift.
# Stagnation is only reset on an upward tick; lateral drifts leave it running
# to prevent the walk from wandering a flat plateau indefinitely.
function accept_step!(state::SearchState, direction, new_params, new_solutions, new_real, filename)
    log_to_master(new_real, new_params)
    if new_real > state.real_count
        println("Upward tick: $(state.real_count) -> $new_real")
        state.stagnation = 0
    else
        println("Lateral drift: $new_real")
    end
    state.params     = new_params
    state.solutions  = new_solutions
    state.real_count = new_real
    state.momentum  .= direction   # in-place: reuses the pre-allocated buffer
    state.step       = INITIAL_STEP
    tmp_file = filename * ".tmp"
    serialize(tmp_file, (state.params, state.solutions, state.real_count))
    mv(tmp_file, filename, force=true)
end

function handle_regression!(state::SearchState, new_real)
    println("Regression: $new_real. Step rejected.")
    state.momentum .= 0.0
    state.step = max(MIN_STEP, state.step * STEP_DECAY)
end

# soften=true halves momentum rather than zeroing it, used for cert/parity failures
# where the direction may still be partially useful.
function reject_step!(state::SearchState; soften::Bool=false)
    soften ? (state.momentum .*= 0.5) : (state.momentum .= 0.0)
    state.step = max(MIN_STEP, state.step * STEP_DECAY)
end

# ==============================================================================
# --- SINGLE SEARCH LOOP ---
# ==============================================================================

function run_single_search(system, filename)
    params, sols, real_count = initialize_or_load_seed(system, filename)
    state = SearchState(params, sols, real_count, zeros(Float64, N_PARAMS), INITIAL_STEP, 0)
    iteration = 0

    while true
        iteration        += 1
        state.stagnation += 1
        println(
            "Iter: $iteration | " *
            "Since upward tick: $(state.stagnation) | " *
            "Real lines: $(state.real_count) | " *
            "Step: $(round(state.step, sigdigits=3))"
        )

        if state.stagnation >= STAGNATION_CAP
            println("Stagnation limit reached. Restarting.")
            return :restart
        end

        outcome, direction, new_params, new_solutions, new_real = attempt_step(system, state)

        if outcome == tracker_fail
            reject_step!(state)
            if state.step <= MIN_STEP
                println("Step collapsed to minimum. Restarting.")
                return :restart
            end
        elseif outcome == cert_fail
            println("Certification failed.")
            reject_step!(state, soften=true)
        elseif outcome == parity_fail
            println("Parity failure (even real count: $new_real).")
            reject_step!(state, soften=true)
        elseif outcome == success
            new_real >= state.real_count ?
                accept_step!(state, direction, new_params, new_solutions, new_real, filename) :
                handle_regression!(state, new_real)
        end
    end
end

# ==============================================================================
# --- TOP-LEVEL ORCHESTRATION ---
# ==============================================================================

function run_automated()
    initialize_registry_if_needed()
    system = build_system()
    # Seed filenames drawn from system entropy via RandomDevice, independent of the
    # global RNG, to prevent collisions across parallel workers.
    make_filename() = "quintic_search_seed_$(randstring(RandomDevice(), 12)).jls"
    current_file = (length(ARGS) > 0 && !isempty(ARGS[1])) ? ARGS[1] : make_filename()
    while true
        result = run_single_search(system, current_file)
        if result == :restart
            current_file = make_filename()
            println("--- NEW SEED FILE: $current_file ---")
        end
    end
end

run_automated()