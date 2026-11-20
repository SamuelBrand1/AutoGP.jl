# Copyright 2026 CMU Probabilistic Computing Systems Lab
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
using Random

using AutoGP: GP
using AutoGP: Inference
using AutoGP.Model: Gen

@testset "structure metric" begin

  config = GP.GPConfig()
  n = 40
  ts = collect(range(0, 1, length=n))
  Random.seed!(1)
  xs = sin.(2pi .* ts ./ 0.1) .+ 0.1 .* randn(n)

  # Two traces with the same structure and time points, but different
  # parameter values, noise and data.
  trace_a = Inference.node_to_trace(
    GP.Linear(0.1, 1.3, 0.7) + GP.Periodic(0.5, 0.1, 1.0), config, ts, xs, 0.01)
  trace_b = Inference.node_to_trace(
    GP.Linear(0.6, 0.2, 2.1) + GP.Periodic(1.7, 0.37, 0.2), config, ts, randn(n), 0.4)
  addresses = Inference.get_gp_parameter_addresses(trace_a)
  @test Set(addresses) == Set(Inference.get_gp_parameter_addresses(trace_b))

  # The metric does not depend on the current parameter values or the data.
  Random.seed!(2); metric_a = Inference.structure_metric(trace_a, addresses; n_draws=8)
  Random.seed!(2); metric_b = Inference.structure_metric(trace_b, addresses; n_draws=8)
  @test metric_a == metric_b
  @test length(metric_a) == length(addresses)
  @test all(>=(1), metric_a)

  # A different structure has a different metric.
  trace_c = Inference.node_to_trace(GP.Linear(0.1, 1.3, 0.7), config, ts, xs, 0.01)
  Random.seed!(2)
  metric_c = Inference.structure_metric(
    trace_c, Inference.get_gp_parameter_addresses(trace_c); n_draws=8)
  @test length(metric_c) == 4

  # HMC with the metric moves kernel parameters and noise jointly. It needs
  # a Gen whose leapfrog position step uses the metric (after 0.4.9).
  observations = Inference.get_observations_choicemap(trace_a)
  hmc_config = Dict(:metric_draws => 16)
  if pkgversion(Gen) > v"0.4.9" || isdefined(Gen, :value_update)
    Random.seed!(3)
    (new_trace, n_accept, n_trial) = Inference.rejuvenate_particle_parameters(
      trace_a, 10; hmc_config=hmc_config, observations=observations)
    @test n_trial == 10
    @test n_accept > 0
    @test all(new_trace[a] != trace_a[a] for a in addresses)
    @test new_trace[:xs] == trace_a[:xs]
  else
    @test_throws ErrorException Inference.rejuvenate_particle_parameters(
      trace_a, 10; hmc_config=hmc_config, observations=observations)
  end

end
