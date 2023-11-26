using Redbird
using SparseArrays
import Redbird.JlMatBindings as JlMat
using MATLAB
using Debugger
using Enzyme



function meshedge_enzyme(elem::Array{Int}, edges::Array{Int})
    edges[:] = sort(elem, dims=2)
    return nothing
end


function main()

    prop = [0     0 1 1
            0.008 1 0 1.37
            0.016 1 0 1.37]
    prop = reshape(prop, 1, size(prop)...)

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

    edges = zeros(Int, 1226766, 2)
    ∂edges = ones(Int, 1226766, 2)

    elem = cfg.elem
    ∂elem = zeros(Int, 204461, 4)

    @show size(Redbird.iso2mesh.meshedge(cfg.elem))
    # print("Taking the derivative!")
    # forward_solver(prop, cfg, ∇ϕ_i∇ϕ_j, detval)
    @show sum(elem), sum(∂elem)
    @show sum(edges), sum(∂edges)

    Enzyme.autodiff(Reverse, meshedge_enzyme,
                    Duplicated(elem, ∂elem),
                    Duplicated(edges, ∂edges)
    )
    # # Redbird.Forward.rbgetbulk(cfg, cfg.wavelengths, prop)
    # @show bkprop, ∂bkprop
    @show sum(elem), sum(∂elem)
    @show sum(edges), sum(∂edges)
    # Duplicated(Amat, d_Amat),
    # Duplicated(rhs, d_rhs));


    # @show dprop
end

main()