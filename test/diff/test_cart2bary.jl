using Redbird
using SparseArrays
import Redbird.JlMatBindings as JlMat
using MATLAB
using Debugger
using Enzyme




function cart2bary_enzyme(node::Array{Float64}, elem::Array{Int}, 
                         optode::Array{Float64},
                         b::Array{Float64})
    mi = size(optode, 1)                     
    ni = collect(1:mi)
    b[:] = Redbird.iso2mesh.cart2bary(node[elem[1, :], :], optode[ni, :])
    return nothing
end


function main()    
    prop = [0     0 1 1
            0.008 1 0 1.37
            0.016 1 0 1.37]
    prop = reshape(prop, 1, size(prop)...)

    cfg = Redbird.Structs.RBConfigJL()

    cfg.node, cfg.face, cfg.elem = JlMat.meshabox([0 0 0], [60 60 30], 1)
    cfg.seg = ones(Int, size(cfg.elem, 1), 1)

    (xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

    cfg.srcpos = Float64.(hcat(xi[:], yi[:], zeros(length(yi), 1)))
    cfg.detpos = Float64.(hcat(xi[:], yi[:], 60 * ones(length(yi), 1)))
    cfg.srcdir = Float64.([0. 0. 1.])
    cfg.detdir = Float64.([0. 0. -1.])

    cfg.wavelengths = [""]


    cfg = Redbird.Forward.rbmeshprep(cfg, prop)
    node = cfg.node
    elem = cfg.elem
    srcpos = cfg.srcpos
    srcdir = Float64.(cfg.srcdir)
    detpos = cfg.detpos
    detdir = Float64.(cfg.detdir)

    face = cfg.face
    wavelengths = cfg.wavelengths
    seg = cfg.seg

    b = zeros(Float64, 50, 4)
    ∂b = zeros(Float64, 50, 4)
    bkprop = Redbird.Forward.rbgetbulk(node, elem, face, seg, wavelengths, prop) 
    (optode, widesrc) = Redbird.Forward.rbgetoptodes(node, bkprop, srcpos, srcdir, detpos, detdir)
    mi = size(optode, 1)                     
    ni = collect(1:mi)
    @show size(node[elem[1, :], :])
    @show size(optode[ni, :])
    @show size(Redbird.iso2mesh.cart2bary(node[elem[1, :], :], optode[ni, :]))

    Enzyme.autodiff(Reverse, cart2bary_enzyme,
    Const(node), Const(elem), Const(optode), Duplicated(b, ∂b))
    @show sum(newloc), sum(∂newloc)

end

main()