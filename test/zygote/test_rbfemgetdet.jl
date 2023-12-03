using Redbird
using SparseArrays
import Redbird.JlMatBindings as JlMat
using MATLAB
using Debugger
using Zygote


function main()

    prop = [0     0 1 1
            0.008 1 0 1.37
            0.016 1 0 1.37]
    prop = reshape(prop, 1, size(prop)...)

    ∂prop = zeros(Float64, size(prop)...)

    cfg = Redbird.Structs.RBConfig()

    cfg.node, cfg.face, cfg.elem = Redbird.JlMatBindings.meshabox([0 0 0], [60 60 30], 1)
    cfg.seg = ones(Int, size(cfg.elem, 1), 1)

    (xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

    cfg.srcpos = Float64.(hcat(xi[:], yi[:], zeros(length(yi), 1)))
    cfg.detpos = Float64.(hcat(xi[:], yi[:], 60 * ones(length(yi), 1)))
    cfg.srcdir = [0 0 1]
    cfg.detdir = [0 0 -1]

    cfg.wavelengths = [""]


    cfg = Redbird.Forward.rbmeshprep(cfg, prop)

    ∇̇ϕ_i∇ϕ_j = Redbird.Forward.rb∇̇ϕ_i∇ϕ_j(cfg)[1]

    cfg = Redbird.Forward.rbmeshprep(cfg, prop)

    ∇̇ϕ_i∇ϕ_j = Redbird.Forward.rb∇̇ϕ_i∇ϕ_j(cfg)[1]

    Amat = Redbird.Forward.rbfemlhs(cfg, prop, ∇̇ϕ_i∇ϕ_j, 1)

    (rhs, loc, bary, optode) = Redbird.Forward.rbfemrhs(cfg, prop)

    ϕ = Redbird.Forward.rbfemsolve(Amat, rhs, :qmr)

    # (detval, goodix) = Redbird.Forward.rbfemgetdet(ϕ, cfg, loc, bary)
    ∂ϕ = gradient(x -> sum(Redbird.Forward.rbfemgetdet(x, cfg, loc, bary)[1]), ϕ)
    @show typeof(∂ϕ)


    # @show dprop
end

main()