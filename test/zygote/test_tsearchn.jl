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

    cfg = Redbird.Structs.RBConfig()

    cfg.node, cfg.face, cfg.elem = Redbird.JlMatBindings.meshabox([0 0 0], [60 60 30], 1)
    cfg.seg = ones(Int, size(cfg.elem, 1), 1)

    (xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

    cfg.srcpos = Float64.(hcat(xi[:], yi[:], zeros(length(yi), 1)))
    cfg.detpos = Float64.(hcat(xi[:], yi[:], 60 * ones(length(yi), 1)))
    cfg.srcdir = Float64.([0. 0. 1.])
    cfg.detdir = Float64.([0. 0. -1.])

    cfg.wavelengths = [""]


    cfg = Redbird.Forward.rbmeshprep(cfg, prop)
    
    function tsearchn_func(prop::Array{Float64})
        (optode, widesrc) = Redbird.Forward.rbgetoptodes(cfg, prop)
        (newloc, newbary) = Redbird.iso2mesh.tsearchn(cfg.node, cfg.elem, optode)
        return sum(newloc)
    end

    (∂prop, ) = gradient(tsearchn_func, prop)
    @show sum(∂prop)
end

main()