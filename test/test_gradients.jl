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
using AutoGP: Greedy
using AutoGP: Inference
using AutoGP.Model: Gen

# Central finite difference of the trace score in the choice at `address`.
function score_finite_difference(trace, address; h=1e-6)
  args = Gen.get_args(trace)
  argdiffs = map(_ -> Gen.NoChange(), args)
  score(z) = Gen.get_score(
    Gen.update(trace, args, argdiffs, Gen.choicemap((address, z)))[1])
  z = trace[address]
  return (score(z + h) - score(z - h)) / (2h)
end

@testset "choice gradients match finite differences" begin

  Random.seed!(1)
  # The default prior gives zero probability to some of the primitive kernels.
  config = GP.GPConfig(
    node_dist_leaf=fill(1/5, 5),
    node_dist_nocp=fill(1/7, 7),
    node_dist_cp=fill(1/8, 8))
  n = 30
  ts = collect(range(0, 1, length=n))
  # Repeated time points have zero distance off the diagonal.
  ts[end] = ts[end-1]
  xs = sin.(9 .* ts) .+ 0.3 .* randn(n)
  noise = 0.1

  leaves = [
    GP.Constant(0.5),
    GP.Linear(0.1, 1.3, 0.7),
    GP.SquaredExponential(0.47, 0.13),
    GP.GammaExponential(0.42, 0.58, 3.2),
    GP.GammaExponential(0.31, 1.7, 0.9),
    GP.Periodic(0.96, 0.21, 1.1),
  ]
  # The change point is within its scale (.001) of a time point, where
  # the score is sensitive to its location.
  location = ts[12] + 2e-4
  kernels = vcat(
    leaves,
    [GP.Plus(a, b) for a in leaves for b in leaves],
    [GP.Times(a, b) for a in leaves for b in leaves],
    [
      GP.ChangePoint(leaves[2], leaves[6], location, .001),
      GP.ChangePoint(leaves[4] + leaves[2], leaves[3] * leaves[6], location, .001),
      (leaves[2] + leaves[4]) * (leaves[6] + leaves[3] * leaves[1]),
    ])

  for kernel in kernels
    trace = Inference.node_to_trace(kernel, config, ts, xs, noise)
    addresses = Inference.get_gp_parameter_addresses(trace)
    # Every numeric parameter of the kernel, and the noise.
    n_params = sum(
      length(fieldnames(typeof(k))) for k in GP.unroll(kernel) if k isa GP.LeafNode)
    n_params += count(k -> k isa GP.ChangePoint, GP.unroll(kernel))
    @test length(addresses) == n_params + 1
    (_, _, gradients) = Gen.choice_gradients(trace, Gen.select(addresses...))
    for address in addresses
      expected = score_finite_difference(trace, address)
      @test isapprox(gradients[address], expected; rtol=1e-4, atol=1e-4)
    end
  end

end

@testset "divergent gradient steps do not throw" begin

  Random.seed!(1)
  n = 30
  ts = collect(range(0, 1, length=n))
  xs = sin.(9 .* ts) .+ 0.3 .* randn(n)
  kernel = GP.Linear(0.1, 1.3, 0.7) + GP.Periodic(0.96, 0.21, 1.1)
  trace = Inference.node_to_trace(kernel, GP.GPConfig(), ts, xs, 0.1)
  addresses = filter(!=(:noise), Inference.get_gp_parameter_addresses(trace))

  # HMC with an unstable step size rejects every trajectory.
  (new_trace, n_accept, n_trial) = Inference.rejuvenate_particle_parameters(
    trace, 5;
    hmc_config=Dict(:eps_param => 10.),
    observations=Inference.get_observations_choicemap(trace))
  @test n_trial == 5
  @test n_accept == 0
  @test all(new_trace[a] == trace[a] for a in addresses)

  # Gradient ascent backs off from an initial step that is far too large.
  new_trace = Greedy.map_optimize(
    trace, Gen.select(addresses...); max_step_size=1e6)
  @test Gen.get_score(new_trace) >= Gen.get_score(trace)

end
