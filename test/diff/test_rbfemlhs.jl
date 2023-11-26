using Redbird
using SparseArrays
import Redbird.JlMatBindings as JlMat
using MATLAB
using Debugger
using Enzyme


function rbfemlhs_enzyme(node::Array{Float64}, elem::Array{Int}, face::Array{Int}, edges::Array{Int}, area::Vector{Float64}, 
                         reff::Array{Float64},
                         ω::Matrix{Float64}, seg::Array{Int}, 
                         wavelengths::Vector{String},
                         evol::Vector{Float64}, ∇̇ϕ_i∇ϕ_j::Array{Float64},
                         prop::Array{Float64}, 
                         wavelength::Int, Amat::SparseMatrixCSC{Float64, Int64})
    Amat[:] = Redbird.Forward.rbfemlhs(node, elem, face, edges, area, reff, ω, seg, wavelengths, evol, ∇̇ϕ_i∇ϕ_j, prop, wavelength)
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

    cfg.srcpos = hcat(xi[:], yi[:], zeros(length(yi), 1))
    cfg.detpos = hcat(xi[:], yi[:], 60 * ones(length(yi), 1))
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
    Amat = sparse(zeros(100, 100))
    ∂Amat = sparse(zeros(100, 100))

    @show size(Redbird.Forward.rbfemlhs(node, elem, face, edges, area, reff,
                             ω, seg, wavelengths,
                             evol, ∇̇ϕ_i∇ϕ_j,
                             prop, 
                             wavelength))

    Enzyme.autodiff(Reverse, rbfemlhs_enzyme,
    Const(node), Const(elem), Const(face), Const(edges), Const(area), Const(reff), Const(ω), Const(seg), Const(wavelengths), Const(evol),
    Const(∇̇ϕ_i∇ϕ_j),
    Duplicated(prop, ∂prop),
    Const(wavelength),
    Duplicated(Amat, ∂Amat)
    )
    @show Amat, ∂Amat
    # Duplicated(Amat, d_Amat),
    # Duplicated(rhs, d_rhs));


    # @show dprop
end

main()