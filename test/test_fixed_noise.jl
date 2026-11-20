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

using AutoGP
using AutoGP: GP
import AutoGP: Model

using Gen

@testset "test_fixed_noise" begin

    ds = collect(1.0:20.0)
    y = sin.(ds ./ 3) .+ 0.1 .* cos.(7 .* ds)
    scores(model) = [Gen.get_score(t) for t in model.pf_state.traces]

    @testset "fit_smc! and remove_data! with fixed noise" begin
        AutoGP.seed!(1)
        noise = 0.1
        config = GP.GPConfig(noise=noise)
        model = AutoGP.GPModel(ds, y; n_particles=2, config=config)
        AutoGP.fit_smc!(model; schedule=[10, 20], n_mcmc=3, n_hmc=3)
        noises = AutoGP.observation_noise_variances(model; reparameterize=false)
        @test all(isapprox.(noises, noise + Model.JITTER; rtol=1e-10))

        AutoGP.add_data!(model, [21.0, 22.0], [0.5, 0.6])
        AutoGP.remove_data!(model, [21.0])
        @test length(model.ds) == 21
        @test all(length(t[:xs]) == 21 for t in model.pf_state.traces)
        noises = AutoGP.observation_noise_variances(model; reparameterize=false)
        @test all(isapprox.(noises, noise + Model.JITTER; rtol=1e-10))
    end

    @testset "Dict round trip preserves noise and scores" begin
        for config in [GP.GPConfig(), GP.GPConfig(noise=0.1)]
            AutoGP.seed!(2)
            model1 = AutoGP.GPModel(ds, y; n_particles=2, config=config)
            AutoGP.fit_smc!(model1; schedule=[10, 20], n_mcmc=3, n_hmc=3)
            model2 = AutoGP.GPModel(Dict(model1))
            noises1 = AutoGP.observation_noise_variances(model1; reparameterize=false)
            noises2 = AutoGP.observation_noise_variances(model2; reparameterize=false)
            @test all(isapprox.(noises1, noises2; rtol=1e-10))
            @test all(isapprox.(scores(model1), scores(model2); rtol=1e-8))
            # Repeated round trips do not drift.
            model3 = AutoGP.GPModel(Dict(AutoGP.GPModel(Dict(model2))))
            noises3 = AutoGP.observation_noise_variances(model3; reparameterize=false)
            @test all(isapprox.(noises1, noises3; rtol=1e-10))
        end
    end

end
