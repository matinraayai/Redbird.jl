using Redbird
using SparseArrays
using MATLAB
import Redbird.JlMatBindings as JlMat
using ChainRulesTestUtils
using Zygote
using Combinatorics
using ChainRulesCore
using Krylov
using Statistics
using Flux
using CUDA
using GPUArrays
using ProgressMeter
using FiniteDiff
using Debugger
using LinearAlgebra
using CUDA.CUSPARSE
using ParameterSchedulers


Base.size(c::Combinatorics.Combinations) = binomial(c.n, c.t)

Base.eltype(::Type{Combinatorics.Combinations}) = Int

function get_ground_truth()
    cfg = Redbird.Structs.RBConfig()
    prop=[0 0 1 1;0.4 1 0.7 1.37]

    prop = reshape(prop, 1, size(prop)...)


    cfg.node, cfg.face, cfg.elem = Redbird.JlMatBindings.meshabox([0 0 0], [60 60 30], 1)
    
    @show size(cfg.node), size(prop)
    cfg.seg = ones(Int, size(cfg.elem, 1), 1)
    cfg.srcpos = [30 30 0]
    cfg.detpos = [30 30 30]
    cfg.detdir = [0 0 -1]
    cfg.srcdir = [0 0 1]


    # cfg.seg[1:100] .= 0
    # cfg.seg[101:200] .= 2
    # (xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

    # cfg.srcpos = hcat(xi[:], yi[:], zeros(length(yi), 1))
    # cfg.detpos = hcat(xi[:], yi[:], 60 * ones(length(yi), 1))
    # cfg.srcdir = [0 0 1]
    # cfg.detdir = [0 0 -1]


    cfg.wavelengths = [""]

    cfg.ω = [0]

    cfg = Redbird.Forward.rbmeshprep(cfg, prop)
    cfg = Redbird.Forward.rbprop_prep(cfg, prop)
    ∇ϕ_i∇ϕ_j = Redbird.Forward.rb∇̇ϕ_i∇ϕ_j(cfg)[1]

    print("Problem setup:\n")
    Amat = Redbird.Forward.rbfemlhs(cfg, prop, ∇ϕ_i∇ϕ_j, 1)
    rhs, loc, bary, optode = Redbird.Forward.rbfemrhs(cfg, prop)
    # print("Problem complete\n")
    # Amat = Array(Amat) |> cu
    # rhs = Array(rhs) |> cu
    @show minimum(abs.(diag(Amat)))
    Amat = CuSparseMatrixCSC(Amat)
    rhs = CuMatrix(rhs)
    # @show typeof(Amat), typeof(rhs)
    ϕ = Redbird.Forward.rbfemsolve(Amat, rhs)
    print("Problem solved\n")
    @show sum(ϕ)
    ϕ = ϕ |> cpu
    detval = Redbird.Forward.rbfemgetdet(ϕ, cfg, loc, bary)[1]

    @show detval
    return detval
end


function main()
    detval_gt = get_ground_truth()
    @show sum(detval_gt)

    cfg = Redbird.Structs.RBConfig()

    # prop = [0 0 1 1;0.5 1 0 1.37]
    prop = [0 0 1 1;0.1 0.9 0.5 1.37]

    gt_prop=[0 0 1 1;0.4 1 0.7 1.37]
    # prop = [0.0 0.0 1.0 1.0; -0.006365646944282058 0.8465493280348011 0.5953914276569785 1.37]

    
    prop = reshape(prop, 1, size(prop)...)


    rule = Flux.Optimise.Adam()

    # schedule = SinExp(λ0 = 1e-8, λ1 = 1e-9, period = 20, γ = 0.8)
    schedule = SinDecay2(λ0 = 5e-1, λ1 = 1e-3, period = 100)

    opt_state = Flux.setup(rule, prop)

    @show propertynames(opt_state)


    

    gt_prop = reshape(gt_prop, 1, size(gt_prop)...)


    cfg.node, cfg.face, cfg.elem = Redbird.JlMatBindings.meshabox([0 0 0], [60 60 30], 1)
    
    @show size(cfg.node), size(prop)

    cfg.seg = ones(Int, size(cfg.elem, 1), 1)
    cfg.srcpos = [30 30 0]
    cfg.srcdir = [0 0 1]

    cfg.detpos = [30 30 30]
    cfg.detdir = [0 0 -1]

    # cfg.node, cfg.face, cfg.elem = Redbird.JlMatBindings.meshabox([40 0 0], [160 120 60], 10)


    # @show size(cfg.node), size(cfg.face), size(cfg.elem)
    # # cfg.seg = ones(Int, size(cfg.elem, 1), 1)
    # # cfg.seg[1:100] .= 0
    # # cfg.seg[101:200] .= 2
    # (xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

    # cfg.srcpos = hcat(xi[:], yi[:], zeros(length(yi), 1))
    # cfg.detpos = hcat(xi[:], yi[:], 60 * ones(length(yi), 1))
    # cfg.srcdir = [0 0 1]
    # cfg.detdir = [0 0 -1]

    cfg.wavelengths = [""]

    cfg.ω = [0]


    cfg = Redbird.Forward.rbmeshprep(cfg, prop)
    ∇ϕ_i∇ϕ_j = Redbird.Forward.rb∇̇ϕ_i∇ϕ_j(cfg)[1]

    function forward_solve(prop)
        cfg = Redbird.Forward.rbprop_prep(cfg, prop)
        Amat = Redbird.Forward.rbfemlhs(cfg, prop, ∇ϕ_i∇ϕ_j, 1)
        rhs, loc, bary, optode = Redbird.Forward.rbfemrhs(cfg, prop)
        # Amat = CuArray(Amat)
        # rhs = CuArray(rhs)
        @show size(rhs), size(Amat)
        # Amat = CuSparseMatrixCSC(Amat)
        # rhs = CuMatrix(rhs)
        # Amat = Amat |> cu
        # rhs = Matrix(rhs) |> cu
        ϕ = Redbird.Forward.rbfemsolve(Amat, rhs)
        # @show sum(ϕ)
        # ϕ = ϕ |> cpu
        detval = Redbird.Forward.rbfemgetdet(ϕ, cfg, loc, bary)[1]
        @show detval
        return detval
    end

    α = 1e10
    β = 1
    γ = 1e2


    # p = Progress(50; showspeed=true)
    for (eta, i) ∈ zip(schedule, 1:500)
        print("Iteration: $i\n")
        Flux.adjust!(opt_state, eta)
        t = @timed val, grads = Flux.withgradient(prop) do m
            detval_est = forward_solve(abs.(m))
            @show detval_est, detval_gt
            β * Flux.Losses.mse(detval_est, detval_gt) + α * sum(m[findall(.<(0), m)].^2) - γ * sum(detval_est.^2)
        end
        # grads
        
        # t = @timed f_grads = FiniteDiff.finite_difference_jacobian(x -> Flux.Losses.mse(forward_solve(x), detval_gt), prop)
        Flux.update!(opt_state, prop, grads[1])

        # f_grads = reshape(f_grads, size(prop)...)
        # @show size(f_grads), size(prop)
        # @show size(f_grads), size(prop)
        # Flux.update!(opt_state, prop, reshape(f_grads, size(prop)...))
        
        # ProgressMeter.next!(p; showvalues= [(:itertime, t), (:loss, val), (:grads, grads), (:prop, prop)])
        # @show t.time, val, grads, prop
        # @show sum(grads)
        # @show grads, f_grads
        # @show reshape(prop, 2, 4), Flux.Losses.mse(gt_prop, prop), t.time
        @show reshape(prop, 2, 4), val, Flux.Losses.mae(gt_prop, prop), t.time
        @show reshape(grads[1], 2, 4), opt_state.rule.eta
    end
    print("Final value of prop: $prop")
end

main()