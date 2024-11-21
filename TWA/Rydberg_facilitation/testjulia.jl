using DifferentialEquations, LinearAlgebra
using Statistics, Folds
using Plots, LaTeXStrings
using JLD2
BLAS.set_num_threads(1)

function sampleSpinZPlus()
    θ = acos(1 / sqrt(3))
    ϕ = 2π * rand()
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

function drift!(du, u, p, t)
    Γ, nAtoms = p
    θ = u[1:nAtoms] 
    ϕ = u[nAtoms .+ (1:nAtoms)]

    for n = 1:nAtoms
        dθ_drift = 0.0
        dϕ_drift = 0.0
        for m = 1:nAtoms
            dθ_drift += sqrt(3) * sin(θ[m]) * (Γ / 2) * cos(ϕ[m] - ϕ[n])
            dϕ_drift += sin(θ[m]) * (Γ / 2) * sin(ϕ[m] - ϕ[n])
        end
        du[n] = (Γ / 2) * cot(θ[n]) + dθ_drift
        du[n + nAtoms] = sqrt(3) * cot(θ[n]) * dϕ_drift
    end
    nothing
end

function diffusion!(du, u, p, t)
    Γ, nAtoms = p
    θ = u[1:nAtoms]
    ϕ = u[nAtoms .+ (1:nAtoms)]
    for n = 1:nAtoms
        du[n, 1] = -sqrt(Γ) * cos(ϕ[n])
        du[n, 2] = sqrt(Γ) * sin(ϕ[n])
        du[nAtoms + n, 1] = sqrt(Γ) * cot(θ[n]) * sin(ϕ[n])
        du[nAtoms + n, 2] = sqrt(Γ) * cot(θ[n]) * cos(ϕ[n])
    end
    nothing
end

function computeTWA(nAtoms, tf, nT, nTraj, dt)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = zeros(2 * nAtoms)
    p = (Γ, nAtoms)
    prob = SDEProblem(drift!, diffusion!, u0, tspan, p, noise_rate_prototype=zeros(2 * nAtoms, 2))
    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)
    alg = EM()
    sol = solve(ensemble_prob, alg, EnsembleThreads();
        saveat=tSave, trajectories=nTraj, maxiters=1e+7, dt=dt,
        abstol=1e-5, reltol=1e-5)
    return tSave, sol
end

Ω = 1
Δ = 0
Γ = 1
tf = 5
nT = 10
nTraj = 10
dt = 1e-3
nAtoms = 125
p = (Γ, nAtoms)
@time t,sol = computeTWA(nAtoms, tf, nT, nTraj, dt)

function compute_spin_Sz(sol, nAtoms)
    θ = sol[1:nAtoms, :, :]
    ϕ = sol[nAtoms+1:2*nAtoms, :, :]
    Szs = sqrt(3) * sum(cos.(θ), dims=1)[1, :, :] / 2
    Sz = mean(Szs, dims=2)[:]

    Sxs = sqrt(3) * sum(sin.(θ) .* cos.(ϕ), dims=1)[1, :, :] / 2
    Sys = sqrt(3) * sum(sin.(θ) .* sin.(ϕ), dims=1)[1, :, :] / 2
    Szs = sqrt(3) * sum(cos.(θ), dims=1)[1, :, :] / 2
    Sms = Sxs - im * Sys 

    Sx = mean(Sxs, dims=2)[:]
    Sy = mean(Sys, dims=2)[:]
    Sz = mean(Szs, dims=2)[:]
    S2 = mean(abs2.(Sxs) + abs2.(Sys) + abs2.(Szs), dims=2)[:]
    W = mean(abs2.(Sms), dims=2)[:] + Sz
    return Sz, W
end

Sz_vals, W = compute_spin_Sz(sol, nAtoms)
# @save "DickeTWA_n=$(nAtoms).jld2" Sz_vals W

plot(nAtoms * Γ * t/log(nAtoms), Sz_vals / nAtoms, 
     label="Sz",
     xlabel="Scaled Time (t)", 
     ylabel="Normalized Sz", 
     xlim=(0.0, 5),
     ylim=(-0.5, 0.5))

     