using LaTeXStrings
using JLD2
BLAS.set_num_threads(1)

include("parameters.jl")
include("functions.jl")
using .MyFunctions
using .MyParams

script_dir = @__DIR__
data_folder = joinpath(script_dir, "results_data")

if !isdir(data_folder)
    mkdir(data_folder)
end

@time begin
    for Ω1 in Ω_values
        Ω = Ω1
        @time t, sol = MyFunctions.computeTWA(MyParams.nAtoms, MyParams.tf, MyParams.nT, MyParams.nTraj, MyParams.dt, Ω, MyParams.Δ, MyParams.V, MyParams.Γ, MyParams.γ)
        Sz_vals = MyFunctions.compute_spin_Sz(sol, MyParams.nAtoms)
        sz_mean = mean(Sz_vals, dims=3)[:, :]
        sz_mean_mean = (1 .+ mean(mean(Sz_vals, dims=3)[:, :], dims=1)) / 2
        @save "$(data_folder)/sz_mean_steady_for_$(case)D,Ω=$(Ω).jld2" t sz_mean_mean
    end
end