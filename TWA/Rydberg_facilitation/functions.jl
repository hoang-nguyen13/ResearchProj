module MyFunctions
    export computeTWA, compute_spin_Sz
    include("parameters.jl")
    using .MyParams
    using DifferentialEquations
    using Random
    using LinearAlgebra
    using Statistics
    using Folds
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
        neighbors = MyFunctions.get_neighbors_vectorized(MyParams.nAtoms)
        Ω, Δ, V, Γ, γ = p
        θ = u[1:nAtoms]
        ϕ = u[nAtoms.+(1:nAtoms)]
        sqrt_3 = sqrt(3)
        dϕ_drift_sum = zeros(nAtoms)
        if case == 1 # 1D case
            dϕ_drift_sum[2:end-1] .= 2 .+ sqrt_3 .* (cos.(θ[1:end-2]) .+ cos.(θ[3:end]))
            dϕ_drift_sum[1] = 1 + sqrt_3 * cos(θ[2])  # Fixed
            dϕ_drift_sum[end] = 1 + sqrt_3 * cos(θ[end-1])  # Fixed
        end
        if case == 2 # 2D case
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
        du[1:nAtoms] .= 0.0
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
        u0 = (2 * nAtoms)
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
        Szs = sqrt(3) * cos.(θ)
        return Szs
    end
    global num_excited = Int(round(percent_excited * nAtoms))
    global excited_indices = sort(randperm(nAtoms)[1:num_excited])
end