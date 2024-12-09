module MyParams
    export Γ, γ, Δ, V, nAtoms, case, tf, nT, nTraj, dt, percent_excited, beta, delta, Ω_values, Ω_crit
    Γ = 1                 
    γ = 20 * Γ          
    Δ = 430 * Γ         
    V = Δ              
    nAtoms = 9       
    tf = 7                  
    nT = 400                 
    nTraj = 250              
    dt = 1e-4
    percent_excited = 1.0 
    case = 2
    Ω_crit = 5.24
    if case == 1
        beta = 0.276
        delta = 0.159
        Ω_values = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,40]
    else
        beta = 0.584
        delta = 0.451
        Ω_values = [0.1,1,3,5,7,9,11,13,15,17,19,20,23]
    end
end