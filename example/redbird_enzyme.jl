using Redbird
using SparseArrays
using MATLAB
using Debugger
using Enzyme
########################################################
##   prepare simulation input
########################################################
function forward_solver(prop::Array{Float64}, cfg::Redbird.Structs.RBConfig,
    ∇ϕ_i∇ϕ_j::Array{Float64}, detval::Array{Float64})
    ########################################################
    ##   Build LHS
    ########################################################

    Amat, ∇ϕ_i∇ϕ_j = Redbird.Forward.rbfemlhs(cfg, prop, ∇ϕ_i∇ϕ_j, 1)

    (rhs, loc, bary) = Redbird.Forward.rbfemrhs(cfg, prop)[1:3]
    # ########################################################
    # ##   Solve for solutions at all freenodes: Afree*sol=rhs
    # ########################################################
    ϕ = Redbird.Forward.rbfemsolve(Amat, rhs, :qmr)
    # ########################################################
    # ##   Extract detector readings from the solutions
    # ########################################################

    detval = Redbird.Forward.rbfemgetdet(ϕ, cfg, loc, bary)[1]
    return nothing
end




function main()
    mat"addpath('./matlab/iso2mesh')"
    mat"addpath('./matlab/mcx')"

    prop = [0     0 1 1
    0.008 1 0 1.37
    0.016 1 0 1.37]
    prop = reshape(prop, 1, size(prop)...)

    cfg = Redbird.Structs.RBConfig()
    cfg.node, cfg.face, cfg.elem = mxcall(:meshabox, 3, [0 0 0], [60 60 30], 1)
    nn = size(cfg.node, 1)
    cfg.seg = ones(size(cfg.elem, 1), 1)

    (xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

    cfg.srcpos = hcat(xi[:], yi[:], zeros(length(yi), 1))
    cfg.detpos = hcat(xi[:], yi[:], 60 * ones(length(yi), 1))
    cfg.srcdir = [0 0 1]
    cfg.detdir = [0 0 -1]


    # cfg.omega = 2 * pi * 70e6
    cfg.omega = [0]

    cfg.wavelengths = [""]


    cfg = Redbird.Forward.rbmeshprep(cfg, prop)
    ∇s = Redbird.Forward.rb∇̇ϕ_i∇ϕ_j(cfg)
    ∇ϕ_i∇ϕ_j = ∇s.∇ϕ_i∇ϕ_j
    d_∇ϕ_i∇ϕ_j = similar(∇ϕ_i∇ϕ_j)
    # ∇ϕ = ∇s.∇ϕ

    dprop = zeros(Float64, 1, 3, 4)

    detval = zeros(Float64, 25, 25)

    d_detval = ones(Float64, 25, 25)
    # Enzyme.API.runtimeActivity!(true)
    print("Taking the derivative!")
    # forward_solver(prop, cfg, ∇ϕ_i∇ϕ_j, detval)
    Enzyme.autodiff(Reverse, forward_solver, Duplicated(prop, dprop),
    Const(cfg),
    Duplicated(∇ϕ_i∇ϕ_j, d_∇ϕ_i∇ϕ_j),
    Duplicated(detval, d_detval));

    @show dprop
end

main()