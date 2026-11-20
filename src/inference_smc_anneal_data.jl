# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# included by ./Inference.jl

import Distributions
import LinearAlgebra

using Printf: @sprintf

# Return normalized particle weights
function compute_particle_weights(state::Gen.ParticleFilterState)
    log_normalized_weights = Gen.normalize_weights(state.log_weights)[2]
    return exp.(log_normalized_weights)
end

# Return normalized particle weights
function effective_sample_size(state::Gen.ParticleFilterState)
    log_normalized_weights = Gen.normalize_weights(state.log_weights)[2]
    return Gen.effective_sample_size(log_normalized_weights)
end

# Errors raised when a gradient step reaches parameters at which the
# covariance matrix cannot be constructed, evaluated or factorized.
const DivergenceError = Union{
    LinearAlgebra.PosDefException,DomainError,AssertionError}

"""
    structure_metric(trace, addresses; n_draws=32)

Diagonal metric (mass matrix) for HMC on the numeric choices at `addresses`:
their prior precision plus the Fisher information of the likelihood, averaged
over the prior. The squared score of the likelihood is averaged over `n_draws`
draws of the choices at `addresses` from their prior, each with data simulated
from the model.

The metric depends on the covariance structure and time points of `trace`,
but not on the current values at `addresses` or on the observed data, so HMC
with it leaves the posterior invariant.
"""
function structure_metric(trace::Gen.Trace, addresses; n_draws::Integer=32)
    selection = Gen.select(addresses...)
    resampled = Gen.select(addresses..., :xs)
    total = zeros(length(addresses))
    n = 0
    for _ in 1:n_draws
        try
            (simulated, _) = Gen.regenerate(trace, resampled)
            (_, values, gradients) = Gen.choice_gradients(simulated, selection)
            # The choices have normal(0, 1) priors, whose gradient is minus
            # the value: adding the values leaves the score of the likelihood.
            score = Gen.to_array(gradients, Float64) .+ Gen.to_array(values, Float64)
            all(isfinite, score) || continue
            total .+= score .^ 2
            n += 1
        catch err
            err isa DivergenceError || rethrow()
        end
    end
    return n == 0 ? ones(length(addresses)) : 1 .+ total ./ n
end

function rejuvenate_particle_parameters(
        trace::Gen.Trace,
        n_hmc::Integer;
        hmc_config=Dict(),
        verbose::Bool=false,
        check::Bool=false,
        observations::ChoiceMap=EmptyChoicemap())
    # Obtain addresses of numeric parameters.
    choices = Gen.get_choices(trace)
    submap = Gen.get_submap(choices, :tree)
    leaf_addrs = Any[
        (:tree => address)
        for (address, value) in Gen.get_values_shallow(submap)
        if address[2] != :node_type
    ]
    # Infer noise?
    infer_noise = !Gen.has_value(observations, :noise)
    # HMC configuration.
    metric_draws = get(hmc_config, :metric_draws, 0)
    L_param   = get(hmc_config, :L_param, 10)
    eps_param = get(hmc_config, :eps_param, metric_draws > 0 ? .1 : .02)
    L_noise   = get(hmc_config, :L_noise, 10)
    eps_noise = get(hmc_config, :eps_noise, .02)
    n_exit    = get(hmc_config, :n_exit, n_hmc)
    # HMC on numeric parameters.
    n_accept = 0
    n_reject = 0
    n_trial = 0
    @assert length(leaf_addrs) > 0
    # With a metric, kernel parameters and noise move jointly. The metric is
    # built once: it is the same for every state the moves below can reach.
    hmc_kwargs = (;)
    if metric_draws > 0
        # Gen up to 0.4.9 leaves the metric out of the leapfrog position step
        # (Gen.jl issue 572), which lowers acceptance instead of raising it.
        (pkgversion(Gen) > v"0.4.9" || isdefined(Gen, :value_update)) || error(
            "hmc_config[:metric_draws] needs a version of Gen after 0.4.9.")
        infer_noise && push!(leaf_addrs, :noise)
        metric = structure_metric(trace, leaf_addrs; n_draws=metric_draws)
        hmc_kwargs = (; metric=metric)
    end
    selection = Gen.select(leaf_addrs...)
    for i=1:n_hmc
        # A trajectory that diverges is rejected.
        trace, accepted = try
            Gen.hmc(trace, selection;
                L=L_param, eps=eps_param, check=check, observations=observations,
                hmc_kwargs...)
        catch err
            err isa DivergenceError || rethrow()
            (trace, false)
        end
        if infer_noise && metric_draws == 0
            trace, = Gen.hmc(trace, Gen.select(:noise);
                L=L_noise, eps=eps_noise, check=check, observations=observations)
        end
        n_trial += 1
        n_accept += Integer(accepted)
        n_reject = accepted ? 0 : n_reject + 1
        (n_reject == n_exit) && break
    end
    verbose && println("accepted HMC[$(n_accept)/$(n_trial)]")
    return trace, n_accept, n_trial
end

function rejuvenate_particle_structure(
        trace::Gen.Trace,
        n_mcmc::Integer,
        n_hmc::Integer,
        biased::Bool;
        hmc_config=Dict(),
        verbose::Bool=false,
        check::Bool=false,
        observations::ChoiceMap=EmptyChoicemap())

    stats = Dict(:mh=>0, :hmc=>0, :hmc_trials=> 0)
    for iter=1:n_mcmc
        trace, accepted = Gen.metropolis_hastings(
            trace,
            tree_rejuvenation_proposal,
            (biased,),
            tree_rejuvenation_involution;
            check=check,
            observations=observations)
        stats[:mh] += Integer(accepted)

        if accepted
            trace, n_accept, trials = rejuvenate_particle_parameters(
                trace, n_hmc;
                verbose=false,
                check=check,
                observations=observations,
                hmc_config=hmc_config)
            stats[:hmc] += n_accept
            stats[:hmc_trials] += trials
        end
    end

    # Report statistics
    if verbose
        print("accepted MCMC[$(stats[:mh])/$(n_mcmc)]"
            * " HMC[$(stats[:hmc])/$(stats[:hmc_trials])]"
            * "\n")
    end

    return trace, stats
end

# Custom implementation of particle_filter_step! in Gen
# that removes the error on !isempty(discard)
# In this case, the sequence of targets is p_n(C) = P(C | x[1:n])
# When performing an "update the target" move
#   simulate C_n ~ \delta(C_{n-1})
#   incremental weight p_n(C) / p_{n-1}(C) = p(x[n] | C, x[1:n-1])
function smc_step!(
        state::Gen.ParticleFilterState{U},
        (ts, config)::Tuple,
        observations::Gen.ChoiceMap) where {U}
    argdiffs = (Gen.UnknownChange(), Gen.NoChange())
    num_particles = length(state.traces)
    Threads.@threads for i=1:num_particles
        (state.new_traces[i], weight_diff, _, discard) = Gen.update(
            state.traces[i], (ts, config), argdiffs, observations)
        state.log_weights[i] += weight_diff
    end
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
end

function run_smc_anneal_data(
        ts::Vector{Float64},
        xs::Vector{Float64};
        config::GPConfig=GPConfig(),
        biased::Bool=false,
        n_particles::Integer=4,
        n_mcmc::Union{Integer,Vector{<:Integer}}=10,
        n_hmc::Union{Integer,Vector{<:Integer}}=10,
        hmc_config=Dict(),
        permutation::Vector{<:Integer}=collect(1:length(ts)),
        schedule::Vector{<:Integer}=range(1:length(ts)),
        adaptive_resampling::Bool=true,
        adaptive_rejuvenation::Bool=false,
        verbose::Bool=false,
        check::Bool=false,
        callback_fn::Function=(; kwargs...) -> nothing)

    # Initialize elapsed runtime.
    elapsed = 0.

    # Apply the permutation
    @assert sort(permutation) == 1:length(ts)
    ts = ts[permutation]
    xs = xs[permutation]

    # Determine inference schedule.
    @assert 1 <= schedule[1]
    @assert schedule[end] == length(ts)
    @assert all((schedule[2:end] .- schedule[1:end-1]) .> 0)

    # Obtain n_mcmc and n_hmc.
    isa(n_mcmc, Integer) && begin n_mcmc = repeat([n_mcmc], length(schedule)) end
    isa(n_hmc, Integer) && begin n_hmc = repeat([n_hmc], length(schedule)) end
    @assert length(n_mcmc) == length(schedule)
    @assert length(n_hmc) == length(schedule)

    # Initialize SMC particles from prior.
    @timeit elapsed begin
        observations = Gen.choicemap()
        if !isnothing(config.noise)
            observations[:noise] = Model.untransform_param(:noise, config.noise, config)
        end
        state = Gen.initialize_particle_filter(
                    model,
                    (Float64[], config),
                    observations,
                    n_particles)
    end

    # Invoke callback for step zero.
    callback_fn(
        state=state,
        ts=ts,
        xs=xs,
        permutation=permutation,
        schedule=schedule,
        step=0,
        elapsed=elapsed,
        rejuvenated=false,
        resampled=false,
        verbose=verbose)

    # Run inference.
    for (i, step) in enumerate(schedule)
        verbose && println("Running SMC round $(step)/$(schedule[end])")

        @timeit elapsed begin

            # Obtain dataset.
            ts_obs = ts[1:step]
            xs_obs = xs[1:step]
            observations[:xs] = xs_obs

            # Reweight step.
            smc_step!(state, (ts_obs, config), observations)

            # Report weights.
            verbose && begin
                w = compute_particle_weights(state)
                ess = effective_sample_size(state)
                wstr = replace(repr(map((x->(@sprintf "%1.2e" x)), w)),"\""=>"")
                println("Particle Weights: $(wstr)")
                println("Particle ESS: $(ess/length(state.traces))")
            end

            # Resample step.
            resampled = false
            if step < schedule[end]
                ess_threshold = adaptive_resampling ? n_particles/2 : n_particles
                resampled = Gen.maybe_resample!(state, ess_threshold=ess_threshold)
                verbose && println("resampled $(resampled)")
            end

            # Rejuvenate step.
            rejuvenated = false
            if !adaptive_rejuvenation || resampled
                rejuvenated = true
                Threads.@threads for p=1:n_particles
                    local trace = state.traces[p]
                    trace, = rejuvenate_particle_structure(
                        trace,
                        n_mcmc[i],
                        n_hmc[i],
                        biased;
                        hmc_config=hmc_config,
                        verbose=verbose,
                        check=check,
                        observations=observations)
                    state.traces[p] = trace
                end
            end

        end

        # Invoke callback for current step.
        callback_fn(
            state=state,
            ts=ts,
            xs=xs,
            permutation=permutation,
            schedule=schedule,
            step=step,
            elapsed=elapsed,
            rejuvenated=rejuvenated,
            resampled=resampled,
            verbose=verbose)
    end

    # Return final traces.
    return state
end
