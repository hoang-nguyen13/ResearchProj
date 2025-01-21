module MyParams
    export Γ, γ, Δ, V, n_atoms, case, tf, nT, nTraj, dt, percent_excited, beta, delta, Ω_values
    Γ = 1
    γ = 1 * Γ
    Δ = 400 * Γ
    V = Δ
    n_atoms = [64,128,256]
    tf = 25
    nT = 1000
    nTraj = 1
    dt = 1e-3
    percent_excited = 1.0
    case = 2
    if case == 1
        beta = 0.276
        delta = 0.159
        Ω_values = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,40]
    else
        beta = 0.584
        delta = 0.451
        Ω_values = [
            0.1, 1, 2
        ]
    end
end