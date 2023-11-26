using Redbird
using SparseArrays
import Redbird.JlMatBindings as JlMat
using MATLAB
using Debugger
using Enzyme



function rbgetbulk_enzyme(node::AbstractArray{<:AbstractFloat},
                          elem::AbstractArray{<:Integer},
                          face::AbstractArray{<:Integer},
                          seg::AbstractArray{<:Integer},
                          wavelengths::Array{String},
                          prop::AbstractArray{<:AbstractFloat},
                          bkprop::AbstractArray{<:AbstractFloat})
    bkprop[:] = Redbird.Forward.rbgetbulk(node, elem, face, seg, wavelengths, prop)
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

    bkprop = zeros(Float64, 1, 4)
    ∂bkprop = ones(Float64, 1, 4)

    ∂prop = zeros(Float64, 1, 3, 4)

    Enzyme.API.runtimeActivity!(true)
    @show @code_typed Redbird.Forward.rbgetbulk(cfg, prop)
    # print("Taking the derivative!")
    # forward_solver(prop, cfg, ∇ϕ_i∇ϕ_j, detval)
    Enzyme.autodiff(Reverse, rbgetbulk_enzyme,
                    Const(cfg.node),
                    Const(cfg.elem),
                    Const(cfg.face),
                    Const(cfg.seg),
                    Const(cfg.wavelengths),
                    Duplicated(prop, ∂prop),
                    Duplicated(bkprop, ∂bkprop)
    )
    # Redbird.Forward.rbgetbulk(cfg, cfg.wavelengths, prop)
    @show bkprop, ∂bkprop
    @show prop, ∂prop
    # Duplicated(Amat, d_Amat),
    # Duplicated(rhs, d_rhs));


    # @show dprop
end

main()