using Redbird
using SparseArrays
import Redbird.JlMatBindings as JlMat
using MATLAB
using Debugger
using Zygote
using Combinatorics


Base.size(c::Combinatorics.Combinations) = binomial(c.n, c.t)

Base.eltype(::Type{Combinatorics.Combinations}) = Int

function main()

    prop = [0     0 1 1
            0.008 1 0 1.37
            0.016 1 0 1.37]
    prop = reshape(prop, 1, size(prop)...)

    cfg = Redbird.Structs.RBConfig()

    cfg.node, cfg.face, cfg.elem = Redbird.JlMatBindings.meshabox([0 0 0], [60 60 30], 1)
    cfg.seg = ones(Int, size(cfg.elem, 1), 1)

    (xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

    cfg.srcpos = hcat(xi[:], yi[:], zeros(length(yi), 1))
    cfg.detpos = hcat(xi[:], yi[:], 60 * ones(length(yi), 1))
    cfg.srcdir = [0 0 1]
    cfg.detdir = [0 0 -1]

    cfg.wavelengths = [""]


    cfg = Redbird.Forward.rbmeshprep(cfg, prop)

    ∇̇ϕ_i∇ϕ_j = Redbird.Forward.rb∇̇ϕ_i∇ϕ_j(cfg)[1]

    wavelength = 1
    (∂prop, ) =
    gradient(prop -> sum(Redbird.Forward.rbfemlhs(cfg, prop, ∇̇ϕ_i∇ϕ_j, wavelength)), prop)

    @show ∂prop

end

main()