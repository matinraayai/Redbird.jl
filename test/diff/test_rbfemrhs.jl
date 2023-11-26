using Redbird
using SparseArrays
import Redbird.JlMatBindings as JlMat
using MATLAB
using Debugger
using Enzyme




function rbfemrhs_enzyme(node::Array{Float64}, elem::Array{Int}, 
                         face::Array{Int}, seg::Array{Int}, 
                         wavelengths::Vector{String},
                         srcpos::Array{Float64}, srcdir::Array{Float64},
                         detpos::Array{Float64}, detdir::Array{Float64},
                         prop::Array{Float64},
                         rhs::SparseVector{Float64})
    bkprop = Redbird.Forward.rbgetbulk(node, elem, face, seg, wavelengths, prop)             
    rhs[:] = Redbird.Forward.rbfemrhs(node, elem, bkprop, srcpos, srcdir, detpos, detdir)[1]
    return nothing
end


function main()

    prop = [0     0 1 1
            0.008 1 0 1.37
            0.016 1 0 1.37]
    prop = reshape(prop, 1, size(prop)...)

    ∂prop = zeros(Float64, size(prop)...)

    cfg = Redbird.Structs.RBConfigJL()

    cfg.node, cfg.face, cfg.elem = Redbird.JlMatBindings.meshabox([0 0 0], [60 60 30], 1)
    cfg.seg = ones(Int, size(cfg.elem, 1), 1)

    (xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

    cfg.srcpos = Float64.(hcat(xi[:], yi[:], zeros(length(yi), 1)))
    cfg.detpos = Float64.(hcat(xi[:], yi[:], 60 * ones(length(yi), 1)))
    cfg.srcdir = Float64.([0. 0. 1.])
    cfg.detdir = Float64.([0. 0. -1.])

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

    bkprop = Redbird.Forward.rbgetbulk(node, elem, face, seg, wavelengths, prop)
    @show @code_typed Redbird.Forward.rbfemrhs(node, elem, bkprop, srcpos, srcdir, detpos, detdir)

    Enzyme.autodiff(Reverse, rbfemrhs_enzyme,
    Const(node), Const(elem), Const(face), Const(seg), Const(wavelengths), Const(srcpos), Const(srcdir),
    Const(detpos), Const(detdir),
    Duplicated(prop, ∂prop),
    Duplicated(rhs, ∂rhs)
    )
    @show rhs, ∂rhs
    # Duplicated(Amat, d_Amat),
    # Duplicated(rhs, d_rhs));


    # @show dprop
end

main()