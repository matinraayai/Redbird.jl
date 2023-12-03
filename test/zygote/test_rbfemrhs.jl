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

    node = cfg.node
    elem = cfg.elem
    face = cfg.face
    edges = sort(Int.(Redbird.iso2mesh.meshedge(elem)), dims=2)
    area = cfg.area
    reff = cfg.reff
    ω = cfg.ω
    seg = cfg.seg
    wavelengths = cfg.wavelengths
    evol = cfg.evol
    wavelength = 1
    rhs = sparse(zeros(100))
    ∂rhs = sparse(ones(100))
    srcpos = cfg.srcpos
    srcdir = Float64.(cfg.srcdir)
    detpos = cfg.detpos
    detdir = Float64.(cfg.detdir)

    function rbfemrhs_zygote(prop::Array{Float64})         
        res = Redbird.Forward.rbfemrhs(cfg, prop)[1]
        return sum(res)
    end

    # bkprop = Redbird.Forward.rbgetbulk(node, elem, face, seg, wavelengths, prop)
    # @show size(Redbird.Forward.rbfemrhs(cfg, prop)[1])
    (∂prop, ) = gradient(rbfemrhs_zygote, prop)
    @show ∂prop
    # Enzyme.autodiff(Reverse, rbfemrhs_enzyme,
    # Const(node), Const(elem), Const(face), Const(seg), Const(wavelengths), Const(srcpos), Const(srcdir),
    # Const(detpos), Const(detdir),
    # Duplicated(prop, ∂prop),
    # Duplicated(rhs, ∂rhs)
    # )
    # @show rhs, ∂rhs
    # Duplicated(Amat, d_Amat),
    # Duplicated(rhs, d_rhs));


    # @show dprop
end

main()