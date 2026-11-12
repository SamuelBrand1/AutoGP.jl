# Copyright 2025 CMU Probabilistic Computing Systems Lab
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

using Test
using LinearAlgebra

using AutoGP
using AutoGP: GP
using AutoGP: Transforms
import AutoGP: Model

using Gen

# ---------------------------------------------------------------------------
# Test-only observation models (NOT shipped). Both are Gaussian in the
# observations, so they reuse the gradient of Gen.mvnormal.
# ---------------------------------------------------------------------------

# Same density as the default observation model: N(0, K + noise*I).
struct WrappedMvNormal <: Gen.Distribution{Vector{Float64}} end
obs_cov(::WrappedMvNormal, xs, K, noise) = K + noise * I

# Noise variance that grows with the magnitude of each observation.
struct HeteroNormal <: Gen.Distribution{Vector{Float64}} end
obs_cov(::HeteroNormal, xs, K, noise) = K + noise * I + Diagonal(0.05 .* abs.(xs))

const TestObservationModel = Union{WrappedMvNormal,HeteroNormal}

function Gen.logpdf(d::TestObservationModel, xs::AbstractVector, K::AbstractMatrix, noise::Real)
    isempty(xs) && return 0.0
    return Gen.logpdf(Gen.mvnormal, xs, zeros(length(xs)), obs_cov(d, xs, K, noise))
end

function Gen.logpdf_grad(d::TestObservationModel, xs::AbstractVector, K::AbstractMatrix, noise::Real)
    isempty(xs) && return (nothing, zeros(0, 0), 0.0)
    (_, _, dK) = Gen.logpdf_grad(Gen.mvnormal, xs, zeros(length(xs)), obs_cov(d, xs, K, noise))
    return (nothing, dK, tr(dK))
end

function Gen.random(d::TestObservationModel, K::AbstractMatrix, noise::Real)
    n = size(K, 1)
    n == 0 && return Float64[]
    return Gen.random(Gen.mvnormal, zeros(n), K + noise * I)
end

(d::TestObservationModel)(K, noise) = Gen.random(d, K, noise)
Gen.has_output_grad(::TestObservationModel) = false
Gen.has_argument_grads(::TestObservationModel) = (true, true)

@testset "test_observation_model" begin

    ds = collect(1.0:24.0)
    y = sin.(ds ./ 3) .+ 0.1 .* cos.(7 .* ds)
    scores(model) = [Gen.get_score(t) for t in model.pf_state.traces]

    # All continuous choices in a trace: kernel parameters and the noise.
    function parameter_selection(trace)
        submap = Gen.get_submap(Gen.get_choices(trace), :tree)
        addrs = Any[
            (:tree => address)
            for (address, value) in Gen.get_values_shallow(submap)
            if address[2] != :node_type]
        return Gen.select(addrs..., :noise)
    end

    @testset "default configuration" begin
        @test isnothing(GP.GPConfig().observation_model)
        model = AutoGP.GPModel(ds, y; n_particles=1)
        @test model.y_transform == Transforms.LinearTransform(y, 1)
    end

    @testset "y_transform keyword" begin
        identity_transform = Transforms.LinearTransform(1., 0.)
        model = AutoGP.GPModel(ds, y; n_particles=1, y_transform=identity_transform)
        @test model.y_transform == identity_transform
        @test model.pf_state.traces[1][:xs] == y
        AutoGP.add_data!(model, [25.0], [0.3])
        @test model.pf_state.traces[1][:xs] == vcat(y, 0.3)
    end

    @testset "wrapped mvnormal matches default scores and gradients" begin
        ts = collect(range(0, 1, length=15))
        xs = sin.(8 .* ts)
        config_default = GP.GPConfig()
        config_wrapped = GP.GPConfig(observation_model=WrappedMvNormal())
        seen = Set{Any}()
        for seed in 1:40
            AutoGP.seed!(seed)
            trace_default, = Gen.generate(
                Model.model, (ts, config_default), Gen.choicemap((:xs, xs)))
            trace_wrapped, = Gen.generate(
                Model.model, (ts, config_wrapped), Gen.get_choices(trace_default))
            push!(seen, typeof(trace_default[]))
            @test trace_wrapped[] == trace_default[]
            @test isapprox(
                Gen.get_score(trace_wrapped), Gen.get_score(trace_default); rtol=1e-12)
            selection = parameter_selection(trace_default)
            (_, _, grads_default) = Gen.choice_gradients(trace_default, selection)
            (_, _, grads_wrapped) = Gen.choice_gradients(trace_wrapped, selection)
            g_default = Gen.to_array(grads_default, Float64)
            g_wrapped = Gen.to_array(grads_wrapped, Float64)
            @test all(isfinite, g_default)
            @test isapprox(g_wrapped, g_default; rtol=1e-8, atol=1e-10)
        end
        # The seeds cover leaf and composite kernels, including changepoints.
        @test GP.ChangePoint in seen
        @test any(T -> T <: GP.LeafNode, seen)
    end

    @testset "wrapped mvnormal matches default inference" begin
        function fit(config)
            AutoGP.seed!(11)
            model = AutoGP.GPModel(ds, y; n_particles=2, config=config)
            AutoGP.fit_smc!(model; schedule=[8, 16, 24], n_mcmc=4, n_hmc=4)
            return model
        end
        model_default = fit(GP.GPConfig())
        model_wrapped = fit(GP.GPConfig(observation_model=WrappedMvNormal()))
        @test isapprox(
            model_wrapped.pf_state.log_weights,
            model_default.pf_state.log_weights; rtol=1e-6)
        @test isapprox(scores(model_wrapped), scores(model_default); rtol=1e-6)
        @test isapprox(
            AutoGP.observation_noise_variances(model_wrapped; reparameterize=false),
            AutoGP.observation_noise_variances(model_default; reparameterize=false);
            rtol=1e-6)
        kernels_default = AutoGP.covariance_kernels(model_default; reparameterize=false)
        kernels_wrapped = AutoGP.covariance_kernels(model_wrapped; reparameterize=false)
        @test typeof.(kernels_wrapped) == typeof.(kernels_default)
    end

    @testset "custom observation model inference" begin
        AutoGP.seed!(3)
        config = GP.GPConfig(observation_model=HeteroNormal())
        model = AutoGP.GPModel(ds, y; n_particles=2, config=config)
        AutoGP.fit_smc!(model; schedule=[8, 16, 24], n_mcmc=4, n_hmc=4)
        @test all(isfinite, model.pf_state.log_weights)
        @test all(isfinite, scores(model))
        @test length(AutoGP.covariance_kernels(model; reparameterize=false)) == 2

        # The noise accessor returns the scalar given to the observation model.
        noises = AutoGP.observation_noise_variances(model; reparameterize=false)
        expected = [
            Model.transform_param(:noise, t[:noise], config) + Model.JITTER
            for t in model.pf_state.traces]
        @test noises == expected

        # Rejuvenation moves run.
        AutoGP.mcmc_parameters!(model, 2)
        AutoGP.mcmc_structure!(model, 2, 2)
        AutoGP.fit_mcmc!(model; n_mcmc=2, n_hmc=2)
        @test all(isfinite, scores(model))

        # Adding data reweights each particle by its change in score.
        old_scores = scores(model)
        old_weights = copy(model.pf_state.log_weights)
        AutoGP.add_data!(model, [25.0, 26.0], [0.4, 0.5])
        @test isapprox(
            model.pf_state.log_weights .- old_weights,
            scores(model) .- old_scores; rtol=1e-10)
        AutoGP.remove_data!(model, [26.0])
        @test all(length(t[:xs]) == 25 for t in model.pf_state.traces)
        @test all(isfinite, model.pf_state.log_weights)

        # Dict round trip preserves particles.
        model2 = AutoGP.GPModel(Dict(model))
        @test model2.config.observation_model isa HeteroNormal
        @test isapprox(scores(model2), scores(model); rtol=1e-8)
        @test model2.pf_state.log_weights == model.pf_state.log_weights
    end

    @testset "fixed noise with custom observation model" begin
        AutoGP.seed!(4)
        config = GP.GPConfig(observation_model=HeteroNormal(), noise=0.05)
        model = AutoGP.GPModel(ds, y; n_particles=2, config=config)
        AutoGP.fit_smc!(model; schedule=[12, 24], n_mcmc=3, n_hmc=3)
        noises = AutoGP.observation_noise_variances(model; reparameterize=false)
        @test all(isapprox.(noises, 0.05 + Model.JITTER; rtol=1e-10))
    end

    @testset "Gaussian-only functions raise an error" begin
        config = GP.GPConfig(
            observation_model=HeteroNormal(), changepoints=false, max_depth=2)
        model = AutoGP.GPModel(ds, y; n_particles=1, config=config)
        ds_query = [25.0, 26.0]
        @test_throws ErrorException AutoGP.predict(model, ds_query)
        @test_throws ErrorException AutoGP.predict_mvn(model, ds_query)
        @test_throws ErrorException AutoGP.predict_quantile(model, ds_query, 0.5)
        @test_throws ErrorException AutoGP.predict_proba(model, ds_query, [0.1, 0.2])
        @test_throws ErrorException AutoGP.predict_sum(model, ds_query, GP.Linear)
        @test_throws ErrorException AutoGP.predict_mvn_sum(model, ds_query, GP.Linear)
        @test_throws ErrorException AutoGP.decompose(model)
        @test_throws ErrorException AutoGP.extract_kernel(model, GP.Linear)
        @test_throws ErrorException AutoGP.split_kernel_sop(model, GP.Linear)
        @test_throws ErrorException AutoGP.fit_greedy!(model)
    end

end
