using LaTeXStrings
using JLD2
using LinearAlgebra
using Statistics
using DifferentialEquations
using Random
using ArgParse
BLAS.set_num_threads(1)

function sampleSpinZPlus(n)
    θ = fill(acos(1 / sqrt(3)), n)
    ϕ = 2π * rand(n)                  
    return θ, ϕ
end
function sampleSpinZMinus(n)
    θ = fill(π - acos(1 / sqrt(3)), n)   
    ϕ = 2π * rand(n)                  
    return θ, ϕ
end

function prob_func(prob, i, repeat)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    excited_indices_set = Set(excited_indices)
    non_excited_indices = setdiff(1:nAtoms, excited_indices)
    θ_excited, ϕ_excited = sampleSpinZPlus(length(excited_indices))
    u0[excited_indices] = θ_excited
    u0[nAtoms.+excited_indices] = ϕ_excited
    θ_non_excited, ϕ_non_excited = sampleSpinZMinus(length(non_excited_indices))
    u0[non_excited_indices] = θ_non_excited
    u0[nAtoms.+non_excited_indices] = ϕ_non_excited
    return remake(prob, u0=u0)
end

function drift!(du, u, p, t)
    neighbors = get_neighbors_vectorized(nAtoms)
    Ω, Δ, V, Γ, γ = p
    θ = u[1:nAtoms]
    ϕ = u[nAtoms.+(1:nAtoms)]
    sqrt_3 = sqrt(3)
    dϕ_drift_sum = zeros(nAtoms)
    if case == 1
        dϕ_drift_sum[2:end-1] .= 2 .+ sqrt_3 .* (cos.(θ[1:end-2]) .+ cos.(θ[3:end]))
        dϕ_drift_sum[1] = 1 + sqrt_3 * cos(θ[2]) 
        dϕ_drift_sum[end] = 1 + sqrt_3 * cos(θ[end-1])
    end
    if case == 2
        for n in 1:nAtoms
            neighbor_indices = neighbors[n]
            dϕ_drift_sum[n] = sum(1 .+ sqrt_3 * cos.(θ[neighbor_indices]))
        end
    end
    cotθ = cot.(θ)
    cscθ = csc.(θ)
    dθ_drift = -2 .* Ω .* sin.(ϕ) .+ Γ .* (cotθ .+ cscθ ./ sqrt_3)
    dϕ_drift = -2 .* Ω .* cotθ .* cos.(ϕ) .+ (V / 2) .* dϕ_drift_sum .- Δ
    du[1:nAtoms] .= dθ_drift
    du[nAtoms.+(1:nAtoms)] .= dϕ_drift
end
function diffusion!(du, u, p, t)
    Ω, Δ, V, Γ, γ = p
    θ = u[1:nAtoms]
    sqrt_3 = sqrt(3)
    term1 = 9 / 6
    term2 = (4 * sqrt_3 / 6) .* cos.(θ)
    term3 = (3 / 6) .* cos.(2 .* θ)
    cscθ2 = csc.(θ) .^ 2
    diffusion = sqrt.(Γ .* (term1 .+ term2 .+ term3) .* cscθ2 .+ 4 .* γ)
    theta_diffusion = 0.0
    if γ < 10
        theta_diffusion = sqrt.(γ) .* abs.(sin.(θ))
    end
    du[1:nAtoms] .= theta_diffusion
    du[nAtoms.+(1:nAtoms)] .= diffusion
end


function get_neighbors_vectorized(nAtoms)
    matrix_size = sqrt(nAtoms) |> Int
    rows = [(div(i - 1, matrix_size) + 1) for i in 1:nAtoms]
    cols = [(mod(i - 1, matrix_size) + 1) for i in 1:nAtoms]
    neighbor_offsets = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1)    # Right
    ]
    neighbors = Vector{Vector{Int}}(undef, nAtoms)
    for i in 1:nAtoms
        row, col = rows[i], cols[i]
        atom_neighbors = [
            (row + dr - 1) * matrix_size + (col + dc)
            for (dr, dc) in neighbor_offsets
            if 1 <= row + dr <= matrix_size && 1 <= col + dc <= matrix_size
        ]
        neighbors[i] = atom_neighbors
    end
    return neighbors
end


function computeTWA(nAtoms, tf, nT, nTraj, dt, Ω, Δ, V, Γ, γ)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    p = (Ω, Δ, V, Γ, γ, nAtoms)

    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)
    
    sol = solve(ensemble_prob, EM(), EnsembleThreads();
        saveat=tSave, trajectories=nTraj, maxiters=1e+7, dt=dt,
        abstol=1e-5, reltol=1e-5)
    
    return tSave, sol
end

function compute_spin_Sz(sol, nAtoms)
    θ = sol[1:nAtoms, :, :]
    Szs = sum(sqrt(3) * cos.(θ),dims=1)
    return Szs
end

Γ = 1
γ = 0.1 * Γ
Δ = 400 * Γ
V = Δ
nAtoms_list = [400]
tf = 25
nT = 1000
nTraj = 500
dt = 1e-3
percent_excited = 1.0
case = 2

if case == 1
    beta = 0.276
    delta = 0.159
    # Ω_values = range(args["omega-start"], args["omega-end"], step=2.0)
else
    beta = 0.584
    delta = 0.451
    Ω_values = 0:0.5:2
end

script_dir = @__DIR__

@time begin
    for nAtoms1 in nAtoms_list
        global nAtoms = nAtoms1
        global num_excited = Int(round(percent_excited * nAtoms))
        global excited_indices = sort(randperm(nAtoms)[1:num_excited])
       data_folder = joinpath(script_dir, "results_data/atoms=$(nAtoms),Δ=$(Δ),γ=$(γ)")
        if !isdir(data_folder)
            mkdir(data_folder)
        end
        println("Computing for nAtoms = $(nAtoms)...\n")
        index = parse(Int, ARGS[1])
        Ω = Ω_values[index]
        println("Computing for Ω = $(Ω)")
        @time t, sol = computeTWA(nAtoms, tf, nT, nTraj, dt, Ω, Δ, V, Γ, γ)
        @save "$(data_folder)/sz_mean_steady_for_$(case)D,Ω=$(Ω),Δ=$(Δ),γ=$(γ).jld2" t sol
        end
    end
end
