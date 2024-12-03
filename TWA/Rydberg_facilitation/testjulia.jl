using DifferentialEquations, LinearAlgebra
using Statistics, Folds
using Plots, LaTeXStrings
using JLD2
using Random
BLAS.set_num_threads(1)

function sampleSpinZPlus()
    rr = rand()
    θ = acos(1/sqrt(3))
    ϕ = 2π * rr
    return θ, ϕ
end

function prob_func(prob, i, repeat)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    for n in 1:nAtoms
        θn, ϕn = sampleSpinZPlus()
        u0[n] = θn
        u0[nAtoms + n] = ϕn
    end
    remake(prob, u0=u0)
end

function get_neighbors(index, matrix_size)
    row = div(index - 1, matrix_size) + 1
    col = mod(index - 1, matrix_size) + 1

    neighbor_offsets = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
    ]
    
    neighbor_indices = []
    
    for (dr, dc) in neighbor_offsets
        neighbor_row = row + dr
        neighbor_col = col + dc
        if neighbor_row >= 1 && neighbor_row <= matrix_size &&
           neighbor_col >= 1 && neighbor_col <= matrix_size
            neighbor_index = (neighbor_row - 1) * matrix_size + neighbor_col
            push!(neighbor_indices, neighbor_index)
        end
    end
    
    return neighbor_indices
end

function precompute_neighbors(nAtoms, lattice_size)
    neighbors = Vector{Vector{Int}}(undef, nAtoms)
    for n in 1:nAtoms
        neighbors[n] = get_neighbors(n, lattice_size)
    end
    return neighbors
end

function drift!(du, u, p, t)
    Ω, Γ, γ, Δ, V, k_f, ϵ, nAtoms, rng = p
    θ = u[1:nAtoms]
    ϕ = u[nAtoms.+(1:nAtoms)]
    sqrt_3 = sqrt(3)
    neighbors = precompute_neighbors(nAtoms, Int(sqrt(nAtoms)))

    for n in 1:nAtoms
        dϕ_drift_sum = 0.0

        facilitated = false

        neighbor_indices = neighbors[n]


        random_values = rand(rng, length(neighbor_indices))
        rydberg_count = 0
  
        for (i, m) in enumerate(neighbor_indices)
            prob = (1 + cos(θ[m])) / 2

            # the condition that you are imposing here is always false, because the atoms if started
            # from exitec state, decay together, 
            # second, when the condition prob > 1 - epsilon is satisfied, it's always blockaded
            # because there's always more than 1 neighbor above 0.9
            if random_values[i] > prob && prob < 1 - ϵ
                rydberg_count += 1
                dϕ_drift_sum += 1 + sqrt_3 * cos(θ[m])
            end
        end
        

        facilitated = (rydberg_count == 1)

        dθ_drift = -2 * Ω * sin(ϕ[n]) + Γ * (cot(θ[n]) + csc(θ[n]) / sqrt(3))
        dϕ_drift = -2 * Ω * cot(θ[n]) * cos(ϕ[n]) + V / 2 * dϕ_drift_sum - Δ

        if facilitated
            dθ_drift += -k_f * (acos(1/sqrt(3)) - cos(θ[n]))
        end

        du[n] = dθ_drift
        du[n+nAtoms] = dϕ_drift
    end
    nothing
end

function diffusion!(du, u, p, t)
    Ω, Γ, γ, Δ, V, k_f, ϵ, nAtoms, rng = p
    θ = u[1:nAtoms]
    ϕ = u[nAtoms.+(1:nAtoms)]
    for n = 1:nAtoms
        du[n] = 0.0
        du[nAtoms+n] = sqrt(Γ * (9 / 6 + (4√3 / 6) * cos(θ[n]) + (3 / 6) * cos(2 * (θ[n]))) * csc(θ[n])^2 + 4 * γ)
    end
    nothing
end

function computeTWA(nAtoms, tf, nT, nTraj, dt)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = (2 * nAtoms)
    p = (Ω, Γ, γ , Δ, V, k_f, ϵ, nAtoms, rng) 
    prob = SDEProblem(drift!, diffusion!, u0, tspan, p, noise_rate_prototype=zeros(2 * nAtoms, 2))
    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)
    alg = EM()
    sol = solve(ensemble_prob, alg, EnsembleThreads();
        saveat=tSave, trajectories=nTraj, maxiters=1e+7, dt=dt,
        abstol=1e-5, reltol=1e-5)
    return tSave, sol
end

function compute_spin_Sz(sol, nAtoms)
    θ = sol[1:nAtoms, :, :];   
    Szs = sqrt(3) * cos.(θ) / 2;
    return Szs
end

λ_values = [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4.0]

for λ in λ_values
    global ν = 128
    global Γ = 1
    global γ = 100 * Γ
    global Δ = (ν * γ / 2)
    global V = Δ
    global Ω = √(λ * γ * Γ / 4)
    global ϵ = 0.9
    global k_f = 10 * Γ
    tf = 15
    nT = 200
    nTraj = 100
    dt = 1e-3
    global ϵ = 0.1
    global nAtoms = 16
    global rng = Random.MersenneTwister(1)
    p = (Ω, Γ, γ, Δ, V, k_f, ϵ, nAtoms, rng)
    @time t,sol = computeTWA(nAtoms, tf, nT, nTraj, dt)

    Sz_vals = compute_spin_Sz(sol, nAtoms);
    Sz_vals = mean(Sz_vals, dims=3);
    sz = vec(sum(0.5 .+ Sz_vals, dims=1) ./ nAtoms);

    @save "rydfac4lambda=$(λ).jld2" t sz
end

