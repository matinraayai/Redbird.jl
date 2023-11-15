module Forward

using SparseArrays
using LinearAlgebra
using Krylov
using Match
using MATLAB
using Debugger

using ..iso2mesh: meshedge, tsearchn
using ..Structs: RBConfig



export rbgetbulk, rbrunforward


# https://stackoverflow.com/questions/46289554/drop-julia-array-dimensions-of-length-1
squeeze(M::AbstractArray) = dropdims(M, dims=tuple(findall(size(M) .== 1)...))

"""

 bkprop=rbgetbulk(cfg)

 Return the optical properties of the "bulk" medium, which is considered
 the medium on the outer-most layer and is interfaced directly with air

 author: Qianqian Fang (q.fang <at> neu.edu)

 input:
     `cfg`: the forward simulation data structure

 output:
     `bkprop`: the optical property quadruplet in the order of 
             [μ_a(1/mm), μ_s(1/mm), g, n]
         if single wavelength, and a containers.Map object made of the
         above quadruplet for each wavelength.

 license:
     GPL version 3, see LICENSE_GPLv3.txt files for details 
"""
function rbgetbulk(cfg::RBConfig)
    bkprop = [0 0 0 1.37]
    if !hasproperty(cfg, :bulk)
        if hasproperty(cfg, :prop) && !isempty(cfg.prop)
            nn = size(cfg.node, 1)
            ne = size(cfg.elem, 1)
            prop = Dict()
            bkprop = Dict()
            if !(cfg.prop isa Dict)
                prop[""] = cfg.prop
            else
                prop = cfg.prop
            end
            wavelengths = keys(prop)
            @show wavelengths
            for waveid = wavelengths
                wv = waveid
                pp = prop[wv]
                if size(prop[wv], 1) < min(nn, ne) # label based prop
                    if hasproperty(cfg, :seg)
                        if length(cfg.seg) == nn
                            bkprop[wv] = pp[cfg.seg[cfg.face[1]]+1, :]
                        elseif length(cfg.seg) == ne
                            xyi = findall(!=(0), cfg.elem .== cfg.face[1])
                            bkprop[wv] = pp[Int(cfg.seg[xyi[1][1]])+1, :]
                        else
                            error("cfg.seg must match the length of node or elem")
                        end
                    else
                        error("labeled proper is defined, but cfg.seg is not given")
                    end
                elseif size(pp, 1) == nn # node based prop
                    bkprop[wv] = pp[cfg.face[1], :]
                elseif size(pp, 1) == ne # elem based prop
                    xyi = findall(!=(0), cfg.elem .== cfg.face(1))
                    bkprop[wv] = pp[xyi[1][1], :]
                else
                    error("row number of cfg.prop is invalid")
                end
            end
            if length(bkprop.keys) == 1
                bkprop = bkprop(wavelengths[1])
            end
        end
    else
        if hasproperty(cfg.bulk, :mua)
            bkprop[1] = cfg.bulk.mua
        end
        if hasproperty(cfg.bulk, :dcoeff)
            bkprop[2] = 1 / (3 * cfg.bulk.dcoeff)
            bkprop[3] = 0
        end
        if hasproperty(cfg.bulk, :musp)
            bkprop[2] = cfg.bulk.musp
            bkprop[3] = 0
        end
        if hasproperty(cfg.bulk, :g)
            bkprop[3] = cfg.bulk.g
        end
        if hasproperty(cfg.bulk, :n)
            bkprop[4] = cfg.bulk.n
        end
    end
    return bkprop
end

"""

 ltr = rbgetltr(cfg, wavelength)

 Compute the transport mean free path (l_tr = 1/μ_tr) in mm in a medium 
 where μ_tr = μ_a + μ_sp is the transport coefficient, μ_a is the absorption 
 coeff and μ_p = μ_s * (1 - g) is the reduced scattering coeff, μ_s is the 
 scattering coeff and g is the anisotropy

 author: Qianqian Fang (q.fang <at> neu.edu)

 input:
     cfg: the forward simulation data structure
     wavelength (optional): if cfg.prop is a containers.Map for
          multispectral simulations, wavelength specifies which
          wavelength, it can be a string or an integer.

 output:
     ltr: transport mean free path in mm

 license:
     GPL version 3, see LICENSE_GPLv3.txt files for details 

 -- this function is part of Redbird-m toolbox

"""
function rbgetltr(cfg::RBConfig, wavelength=nothing)
    bkprop = rbgetbulk(cfg)
    if bkprop isa Dict
        if isnothing(wavelength)
            wavelength = collect(keys(bkprop))[1]
        end
        bkprop = bkprop[wavelength]
    end
    ltr = 1 / (bkprop[1] + bkprop[2] * (1 - bkprop[3]))
    return ltr
end

"""

 [pointsrc, widesrc]=rbgetoptodes(cfg)

 Return the combined list of point optodes (sources+dectors) and widefield
 optodes; In a simulation, all sources, or indenpently, all dectors, can
 only be either point sources or widefield sources.

 author: Qianqian Fang (q.fang <at> neu.edu)

 input:
     cfg: the initial simulation data structure

 output:
     pointsrc: combined point source list - a dimension of Np x 3, where
          Np is the total point source+point detector number
     widesrc: combined widefield source list - a dimension of Nw x 3, where
          Nw is the total point source+point detector number

 license:
     GPL version 3, see LICENSE_GPLv3.txt files for details 

 -- this function is part of Redbird-m toolbox

"""
function rbgetoptodes(cfg::RBConfig)

    pointsrc = []
    widesrc = []

    ltr = rbgetltr(cfg)

    if hasproperty(cfg, :srcpos) && !isempty(cfg.srcpos)
        if size(cfg.srcpos, 2) == size(cfg.node, 1)
            widesrc = cfg.srcpos
        else
            pointsrc = cfg.srcpos + repeat(cfg.srcdir * ltr, size(cfg.srcpos, 1), 1)
        end
    end

    if hasproperty(cfg, :detpos) && !isempty(cfg.detpos)
        if size(cfg.detpos, 2) == size(cfg.node, 1)
            widesrc = [widesrc; cfg.detpos]
        else
            pointsrc = [pointsrc; cfg.detpos + repmat(cfg.detdir * ltr, size(cfg.detpos, 1), 1)]
        end
    end

    return (pointsrc, widesrc)
end



"""

 [deldotdel, ∇ϕ] = rb∇̇ϕ_i∇ϕ_j(cfg)
 
 Compute deldotdel = <∇[ϕ_i].∇[ϕ_j]>, where <> means spatial integration
 inside elements, "." means dot-product, grad[] means gradience, ϕ means
 linear basis function in a tetrahedron. For a linear function ϕ
 ∇[ϕ] is a constant across the element.

 author: Qianqian Fang [q.fang <at> neu.edu]

 input:
     cfg: the redbird data structure

 output:
     deldotdel: precomputed deldotdel=<grad[ϕ_i].grad[ϕ_j]>. For each
         element; deldotdel is a 4x4 symmetric matrix - to store this data
         efficiently; we only store the upper triangule of the matrix per
         element - this gives a Ne x 10 matrix [Ne is the number of tets]
         for each row: [d11, d12, d13, d14, d22, d23, d24, d33, d34, d44]
     ∇ϕ: gradient of the basis functions (grad[ϕ_i]) in each
         tetrahedral element; a 3x4 matrix for each element with a total
         dimension is 3 x 4 x Ne; where 
            3 - gradient direction; for x; y & z
            4 - which basis functions - for node 1; 2; 3 & 4
            Ne - number of element

 license:
     GPL version 3; see LICENSE_GPLv3.txt files for details 

 -- this function is part of Redbird.jl toolbox

"""
function rb∇̇ϕ_i∇ϕ_j(cfg::RBConfig)
    no = cfg.node
    el = cfg.elem[:, 1:4]
    # mat"""
    # $no_matlab = reshape($no($el', :)', 3, 4, size($el, 1))
    # """
    #TODO: Ensure correctness
    no = permutedims(no[Int.(el'), :], (3, 1, 2))
    # @show size(el), size(no), size(no_matlab), no_matlab == no

    # no = reshape(no[Int.(el'), :], 3, 4, size(el, 1))
    # no = reshape(no[Int.(el'), :], 3, 4, size(el, 1))


    ∇ϕ = zeros(3, 4, size(el, 1))

    col = [4 2 3 2
        3 1 4 3
        2 4 1 4
        1 3 2 1]

    for coord = 1:3
        idx = Vector(1:3)
        deleteat!(idx, coord)
        for i = 1:4
            ∇ϕ[coord, i, :] = squeeze((
                (no[idx[1], col[i, 1], :] - no[idx[1], col[i, 2], :]) .* (no[idx[2], col[i, 3], :] - no[idx[2], col[i, 4], :]) -
                (no[idx[1], col[i, 3], :] - no[idx[1], col[i, 4], :]) .* (no[idx[2], col[i, 1], :] - no[idx[2], col[i, 2], :]))) ./ (cfg.evol[:] * 6)
        end
    end

    ∇ϕ_i∇ϕ_j = zeros(size(el, 1), 10)
    count = 1

    for i = 1:4
        for j = i:4
            ∇ϕ_i∇ϕ_j[:, count] = sum(squeeze(∇ϕ[:, i, :] .* ∇ϕ[:, j, :]), dims=1)
            count += 1
        end
    end
    ∇ϕ_i∇ϕ_j = ∇ϕ_i∇ϕ_j .* repeat(cfg.evol[:], 1, 10)
    return (∇ϕ_i∇ϕ_j=∇ϕ_i∇ϕ_j, ∇ϕ=∇ϕ)
end


"""
 out [rhs,loc,bary,optode]
 newcfg=rbmeshprep(cfg)

 Create the right-hand-sides for the FEM system equation, here we solve
 forward systems for both source and detector locations in order to use
 the adjoint method to create Jacobians.

 author: Qianqian Fang (q.fang <at> neu.edu)

 input:
     `cfg`: the initial simulation data structure

 output:
     rhs: the right-hand-side of the FEM equation; if multiple sources are
          used, the rhs is a matrix of dimension Nn x (Ns+Nd), where Nn is 
          the total number of nodes, and Ns the number of sources and
          Nd is the total number of detectors
     loc: the indices of the forward mesh element that encloses each
          source or detector; Nan means the source is outside of the mesh
          or a wide-field source/detector
     bary: the barycentric coordinates of the source/detector if it is
          enclosed by a tetrahedral element; Nan if outside of the mesh
     optode: the full source+detector position list returned by
          rbgetoptodes.m

 license:
     GPL version 3, see LICENSE_GPLv3.txt files for details 

 -- this function is part of Redbird-m toolbox

"""
function rbfemrhs(cfg::RBConfig)
    (optode, widesrc) = rbgetoptodes(cfg)

    if size(optode, 1) < 1 && size(widesrc, 1) < 1
        error("you must provide at least one source or detector")
    end

    loc = []
    bary = []

    if !isempty(widesrc) && (size(widesrc, 2) == size(cfg.node, 1))
        rhs = widesrc'
        loc = NaN * ones(1, size(widesrc, 1))
        bary = NaN * ones(size(widesrc, 1), 4)
    end

    if isempty(optode)
        return
    end

    rhs = sparse(zeros(size(cfg.node, 1), size(widesrc, 1) + size(optode, 1)))
    (newloc, newbary) = tsearchn(cfg.node, Int.(cfg.elem), optode)
    (newloc_mat, newbary_mat) = mxcall(:tsearchn, 2, cfg.node, cfg.elem, optode)
    @show size(newbary_mat), size(newbary), newbary_mat, newbary
    @show size(newloc_mat), size(newloc), newloc_mat, newloc
    loc = [loc; newloc]
    bary = [bary; newbary]

    for i = 1:size(optode, 1)
        if !isnan(newloc[i])
            @show size(rhs), Int.(cfg.elem[newloc[i], :]), i+size(widesrc, 1), size(optode, 1), size(newbary)
            rhs[Int.(cfg.elem[newloc[i], :]), i+size(widesrc, 1)] = newbary[i, :]
        end
    end
    return rhs, loc, bary, optode
end


"""
 [detval, goodidx]=rbfemgetdet(ϕ, cfg, optodeloc, optodebary)

 Retrieving measurement data at detectors as a (N_det by N_src) matrix

 author: Qianqian Fang (q.fang <at> neu.edu)

 input:
     ϕ: the forward solution obtained by rbforward or rbfemsolve
     cfg: the redbird simulation data structure
     rhs: the RHS matrix returned by rbfemrhs (only use this when the
          source or detector contains widefield sources)
     optodeloc: the optode enclosing element ID returned by rbgetoptodes
     optodebary: the optode barycentric coordinates returned by rbgetoptodes

 output:
     detval: #det x #src array, denoting the measurement data
     goodidx: the list of "good optodes" - for point optodes, the non-NaN
        optodeloc indices; for widefield sources- all src/det are included

 license:
     GPL version 3, see LICENSE_GPLv3.txt files for details 

 -- this function is part of Redbird-m toolbox

"""
function rbfemgetdet(ϕ, cfg::RBConfig, optodeloc, optodebary)

    if !hasproperty(cfg, :srcpos) || (hasproperty(cfg, :srcpos) && isempty(cfg.srcpos)) ||
       !hasproperty(cfg, :detpos) || (hasproperty(cfg, :detpos) && isempty(cfg.detpos))
        return (detval=[], goodix=[])
    end

    srcnum = size(cfg.srcpos, 1)
    detnum = size(cfg.detpos, 1)

    goodidx = findall(!isnan, optodeloc[srcnum+1:srcnum+detnum])
    detval = zeros(detnum, srcnum)
    gooddetval = zeros(length(goodidx), srcnum)

    if isempty(goodidx) && size(cfg.detpos, 2) == size(cfg.node, 1) # wide-field det
        for i = 1:srcnum
            for j = 1:detnum
                detval[j, i] = sum(ϕ[:, i] .* cfg.detpos[j, :]')
            end
        end
    else
        for i = 1:length(goodidx)
            if !isnan(optodeloc[i])
                gooddetval[i, :] = sum(ϕ[cfg.elem[optodeloc[srcnum+goodidx(i)], :], 1:srcnum] .* repeat(optodebary[srcnum+goodidx(i), :]',
                        1, srcnum), 1)
            end
        end
        detval[goodidx, :] = gooddetval
    end
    return (detval, goodidx)
end

"""
 [detval, goodidx]=rbfemgetdet(ϕ, cfg, rhs)

 Retrieving measurement data at detectors as a (N_det by N_src) matrix

 author: Qianqian Fang (q.fang <at> neu.edu)

 input:
     `ϕ`: the forward solution obtained by rbforward or rbfemsolve
     `cfg`: the redbird simulation data structure
     `rhs`: the RHS matrix returned by rbfemrhs (only use this when the
          source or detector contains widefield sources)

 output:
     `detval`: #det x #src array, denoting the measurement data
     `goodidx`: the list of "good optodes" - for point optodes, the non-NaN
        optodeloc indices; for widefield sources- all src/det are included
"""
function rbfemgetdet(ϕ, cfg::RBConfig, rhs)

    if !hasproperty(cfg, :srcpos) || (hasproperty(cfg, :srcpos) && isempty(cfg.srcpos)) ||
       !hasproperty(cfg, :detpos) || (hasproperty(cfg, :detpos) && isempty(cfg.detpos))
        return (detval=[], goodix=[])
    end

    srcnum = size(cfg.srcpos, 1)
    detnum = size(cfg.detpos, 1)

    goodidx = findall(!isnan, rhs[srcnum+1:srcnum+detnum])
    detval = zeros(detnum, srcnum)

    detval = rhs[:, srcnum+1:srcnum+detnum]' * ϕ[:, 1:srcnum]
    return (goodidx, detaval)
end

"""
[Amat,deldotdel]
 [Amat,deldotdel]=rbfemlhs(cfg)
   or
 [Amat,deldotdel]=rbfemlhs(cfg, wavelength)
 [Amat,deldotdel]=rbfemlhs(cfg, deldotdel, wavelength)

 create the FEM stiffness matrix (left-hand-side) for solving the
 diffusion equation at a given wavelength

 author: Qianqian Fang (q.fang <at> neu.edu)

 input:
     cfg: the initial simulation data structure
     deldotdel (optional): precomputed operator on the mesh (del_phi dot del_phi)
         where del represents the gradient; see help rbdeldotdel
     wavelength (optional): a string or number denoting the wavelength

 output:
     Amat: the left-hand-side matrix of the FEM equation - a sparse matrix
          of dimension Nn x Nn, where Nn is the number of nodes of the
          forward mesh
     deldotdel: if the 2nd input is not given, this function compute
          deldotdel and return as the 2nd output

 license:
     GPL version 3, see LICENSE_GPLv3.txt files for details 

 -- this function is part of Redbird-m toolbox

"""
function rbfemlhs(cfg::RBConfig, ∇̇ϕ_i∇ϕ_j, wavelength=nothing)
    nn = size(cfg.node, 1)
    ne = size(cfg.elem, 1)

    R_C0 = 1.0 / 299792458000.0

    # cfg.prop is updated from cfg.param and contains the updated mua musp.
    # if cfg.param is node/elem based, cfg.prop is updated to have 4 columns
    # with mua/musp being the first two columns
    # if cfg.param is segmentation based, cfg.prop has the same format as mcx's
    # prop, where the first row is label 0, and total length is Nseg+1

    prop = cfg.prop
    cfgreff = cfg.reff
    omega = cfg.omega

    if isnothing(wavelength) && length(∇̇ϕ_i∇ϕ_j) == 1
        wavelength = deldotdel
    end

    if cfg.prop isa Dict # if multiple wavelengths, take current
        if isnothing(wavelength)
            error("you must specify wavelength")
        end
        if !(wavelength isa String)
            wavelength = "$wavelength"
        end
        prop = cfg.prop[wavelength]
        cfgreff = cfg.reff[wavelength]
        if omega isa Dict
            omega = omega[wavelength]
        end
    end

    # if deldotdel is provided, call native code; otherwise, call mex

    # get mua from prop(:,1) if cfg.prop has wavelengths
    if size(prop, 1) == nn || size(prop, 1) == ne
        mua = prop[:, 1]
        if size(prop, 2) < 3
            musp = prop[:, 2]
        else
            musp = prop[:, 2] .* (1 - prop[:, 3])
        end
    elseif size(prop, 1) < min(nn, ne) # use segmentation based prop list
        mua = prop[Int.(cfg.seg) .+ 1, 1]
        if size(prop, 2) < 3
            musp = prop[Int.(cfg.seg) .+ 1, 2] # assume g is 0
        else
            musp = prop[Int.(cfg.seg) .+ 1, 2] .* (1 .- prop[Int.(cfg.seg) .+ 1, 3])
        end
    end
    dcoeff = 1.0 / (3 * (mua + musp))

    if hasproperty(cfg, :bulk) && hasproperty(cfg.bulk, :n)
        nref = cfg.bulk.n
    elseif hasproperty(cfg, :seg) && size(prop, 1) < min(nn, ne)
        nref = rbgetbulk(cfg)
        if nref isa Dict
            @show keys(nref), wavelength
            nref = nref[wavelength]
        end
        nref = nref[4]
    else
        nref = prop[:, 4]
    end
    Reff = cfgreff

    edges = sort(meshedge(cfg.elem), dims=2)

    # what LHS matrix needs is dcoeff and μ_a, must be node or elem long
    if length(mua) == size(cfg.elem, 1)  # element based property
        Aoffd = ∇̇ϕ_i∇ϕ_j[:, vcat(2:4, 6:7, 9)] .* repeat(dcoeff[:], 1, 6) + repeat(0.05 * mua[:] .* cfg.evol[:], 1, 6)
        Adiag = ∇̇ϕ_i∇ϕ_j[:, [1, 5, 8, 10]] .* repeat(dcoeff[:], 1, 4) + repeat(0.10 * mua[:] .* cfg.evol[:], 1, 4)
        if omega > 0
            Aoffd = complex(Aoffd, repeat(0.05 * omega * R_C0 * nref[:] .* cfg.evol[:], 1, 6))
            Adiag = complex(Adiag, repeat(0.10 * omega * R_C0 * nref[:] .* cfg.evol[:], 1, 4))
        end
    else  # node based properties
        w1 = (1 / 120) * [2 2 1 1; 2 1 2 1; 2 1 1 2; 1 2 2 1; 1 2 1 2; 1 1 2 2]'
        w2 = (1 / 60) * (Diagonal([2 2 2 2]) + 1)
        mua_e = reshape(mua[cfg.elem], size(cfg.elem))
        if length(nref) == 1
            nref_e = nref * ones(size(cfg.elem))
        else
            nref_e = reshape(nref[cfg.elem], size(cfg.elem))
        end
        dcoeff_e = mean(reshape(dcoeff[cfg.elem], size(cfg.elem)), 2)
        Aoffd = ∇̇ϕ_i∇ϕ_j[:, [2:4, 6:7, 9]] .* repeat(dcoeff_e, 1, 6) + (mua_e * w1) .* repeat(cfg.evol[:], 1, 6)
        Adiag = ∇̇ϕ_i∇ϕ_j[:, [1, 5, 8, 10]] .* repeat(dcoeff_e, 1, 4) + (mua_e * w2) .* repeat(cfg.evol[:], 1, 4)
        if cfg.omega > 0
            Aoffd = complex(Aoffd, (omega * R_C0) * (nref_e * w1) .* repeat(cfg.evol[:], 1, 6))
            Adiag = complex(Adiag, (omega * R_C0) * (nref_e * w2) .* repeat(cfg.evol[:], 1, 4))
        end
    end
    # add partial current boundary condition
    edgebc = sort(meshedge(cfg.face), dims=2)
    Adiagbc = cfg.area[:] * ((1 - Reff) / (12 * (1 + Reff)))
    Adiagbc = repeat(Adiagbc, 1, 3)
    Aoffdbc = Adiagbc * 0.5

    Amat = sparse([edges[:, 1]; edges[:, 2]; cfg.elem[:]; edgebc[:, 1]; edgebc[:, 2]; cfg.face[:]],
        [edges[:, 2]; edges[:, 1]; cfg.elem[:]; edgebc[:, 2]; edgebc[:, 1]; cfg.face[:]],
        [Aoffd[:]; Aoffd[:]; Adiag[:]; Aoffdbc[:]; Aoffdbc[:]; Adiagbc[:]])
    # else
    # 	if(size(cfg.elem,2)>4)
    # 		cfg.elem(:,5:end)=[];
    # 	end
    # 	cfg.prop=prop; # use property of the current wavelength
    # 	[Adiag, Aoffd, deldotdel]=rbfemmatrix(cfg);
    # 	Amat = sparse([cfg.rows,cfg.cols,(1:nn)],[cfg.cols,cfg.rows,(1:nn)],[Aoffd,Aoffd,Adiag],nn,nn);
    # 	deldotdel=deldotdel';
    # end

    return (Amat, ∇̇ϕ_i∇ϕ_j)
end


"""

 `res = rbfemsolve(Amat, rhs, method; options...)`

 Solving a linear system defined by Amat*res=rhs using various methods

 author: Qianqian Fang (q.fang <at> neu.edu)

 input:
     Amat: the left-hand-size matrix, can be sparse
     rhs:  the right-hand-size vector or matrix (multiple rhs)
     method: (optional) a string specifying the solving method
            'mldivide': use the left-divide method: res=Amat\rhs
            'blqmr': use the block-QMR iterative method for entire RHS

          other supported methods include
            qmr, tfqmr, cgs, gmres, pcg, cgs, minres, symmlq, bicgstab;

          if method is a positive integer, it calls blqmr.m (part of the
          blit toolbox) with block size defined by "method".

          for all non-block solvers, adding "par" prefix calls parfor to
          solve each RHS in parallel, one must call matlabpool or parpool
          (matlab 2016 or newer) to create the pool first
     options: (optional) additional parameters are passed to the solver
          function to specify solver options, please help qmr, cgs, etc to
          see the additionally accepted parameters
 output:


 license:
     GPL version 3, see LICENSE_GPLv3.txt files for details 

 -- this function is part of Redbird-m toolbox

"""
function rbfemsolve(Amat, rhs, method::Symbol=:qmr; options...)

    # block solvers can handle multiple RHSs
    # switch method
    #     case 'blqmr'
    #         [varargout{1:nargout}]=blqmr(Amat, Array(rhs), varargin{:});
    #         return;
    #     case 'mldivide'
    #         varargout{1}=full(Amat\rhs);
    #         for i=2:nargout
    #             varargout{i}=[];
    #         end
    #         return;
    # end


    # non-block solvers have to solve one RHS at a time

    sol = Any[]

    # do not use parfor
    solver_method = @match method begin
        :\ => \
        :qmr => qmr
        :cgs => cgs
        :lsqr => lsqr
        :gmres => gmres
        :minres => minres
        :symmlq => symmlq
        :bicgstab => bicgstab
        _ => error("method $method is not supported")
    end

    for i = 1:size(rhs, 2)
        @show size(rhs), size(Amat)
        result = solver_method(Amat, SparseVector(rhs))[1]
        @show size(result)
        append!(sol, result)
    end
    res = stack(sol, dims=1)
    res_mat = mxcall(:rbfemsolve, 1, Amat, SparseVector(rhs))
    @show size(res), size(res_mat)
    return stack(sol, dims=1)
end


"""
 returns [detval, ϕ, Amat, rhs, sflag]
 [detval, ϕ] = rbrunforward(cfg)
    or
 [detval, ϕ, Amat, rhs] = rbrunforward(cfg, param1=value1,...)

 Perform forward simulations at all sources and all wavelengths based on the input structure

 ### Arguments
     `cfg::RBConfig`: the redbird data structure

 ### output:
     `detval`: the values at the detector locations
     `ϕ`: the full volumetric forward solution computed at all wavelengths
     `Amat`: the left-hand-side matrices (a containers.Map object) at specified wavelengths 
     `rhs`: the right-hand-side vectors for all sources (independent of wavelengths)
     param/value pairs: (optional) additional parameters
          'solverflag': a cell array to be used as the optional parameters
               for rbfemsolve (starting from parameter 'method'), for
               example  rbrunforward(...,'solverflag',{'pcg',1e-10,200})
               calls rbfemsolve(A,rhs,'pcg',1e-10,200) to solve forward
               solutions
 

 license:
     GPL version 3, see LICENSE_GPLv3.txt files for details 

 -- this function is part of Redbird-m toolbox

"""
function rbrunforward(cfg::RBConfig; kwargs...)

    if !hasproperty(cfg, :∇ϕ_i∇ϕ_j)
        cfg.∇ϕ_i∇ϕ_j = rb∇̇ϕ_i∇ϕ_j(cfg)
        deldotdel = mxcall(:rbdeldotdel, 1, cfg.cfg)
        @show cfg.∇ϕ_i∇ϕ_j == deldotdel
    end


    wavelengths = [""]

    if cfg.prop isa Dict
        wavelengths = keys(cfg.prop)
    end

    Amat = Dict()
    ϕ = Dict()
    detval = Dict()

    if :solverflag in keys(kwargs)
        solverflag = kwargs[:solverflag]
    else
        solverflag = :qmr
    end


    ########################################################
    ##   Build RHS
    ########################################################

    (rhs, loc, bary, optode) = rbfemrhs(cfg)
    # mat"
    # [cfg_mat.node, cfg_mat.face, cfg_mat.elem] = meshabox([0 0 0], [60 60 30], 1);

    # nn = size(cfg_mat.node, 1);
    # cfg_mat.seg = ones(size(cfg_mat.elem, 1), 1);
    # cfg_mat.srcpos = [30 30 0];
    # cfg_mat.srcdir = [0 0 1];

    # cfg_mat.prop = [0 0 1 1;0.005 1 0 1.37];
    # cfg_mat.omega = 0;

    # cfg_mat = rbmeshprep(cfg_mat);
    # [$rhs_mat, $loc_mat, $bary_mat, $optode_mat] = rbfemrhs(cfg_mat);
    # "
    # @show rhs ≈ rhs_mat, rhs, rhs_mat
    # @show loc, loc_mat
    # @show bary ≈ bary_mat, bary, bary_mat
    # @show optode == optode_mat
    @show wavelengths cfg.prop
    for waveid = wavelengths
        wv = waveid
        @show wv

        ########################################################
        ##   Build LHS
        ########################################################
        (Amat[wv], ∇ϕ_i∇ϕ_j) = rbfemlhs(cfg, cfg.∇ϕ_i∇ϕ_j, wv) # use native matlab code, 1 sec for 50k nodes

        mat"
        [cfg_mat.node, cfg_mat.face, cfg_mat.elem] = meshabox([0 0 0], [60 60 30], 1);
    
        nn = size(cfg_mat.node, 1);
        cfg_mat.seg = ones(size(cfg_mat.elem, 1), 1);
        cfg_mat.srcpos = [30 30 0];
        cfg_mat.srcdir = [0 0 1];
    
        cfg_mat.prop = [0 0 1 1;0.005 1 0 1.37];
        cfg_mat.omega = 0;
    
        cfg_mat = rbmeshprep(cfg_mat);
        [$rhs_mat, $loc_mat, $bary_mat, $optode_mat] = rbfemrhs(cfg_mat);
        $Amat_mat = rbfemlhs(cfg_mat, cfg_mat.deldotdel, '');
        "
        # wavelengths={''};
        # if(isa(cfg_mat.prop,'containers.Map'))
        #     wavelengths=cfg_mat.prop.keys;
        # end
        @show Amat[wv] ≈ Amat_mat, sum(Amat[wv] - Amat_mat)
        ########################################################
        ##   Solve for solutions at all nodes: Amat*res=rhs
        ########################################################

        #solverflag={'pcg',1e-12,200}; # if iterative pcg method is used
        ϕ[wv] = rbfemsolve(Amat[wv], rhs, solverflag)

        ########################################################
        ##   Extract detector readings from the solutions
        ########################################################

        detval[wv] = rbfemgetdet(ϕ[wv], cfg, loc, bary) # or detval=rbfemgetdet(ϕ(wv), cfg, rhs); 
    end

    # if only a single wavelength is required, return regular arrays instead of a map
    if length(wavelengths) == 1
        Amat = Amat[wavelengths[1]]
        ϕ = ϕ[wavelengths[1]]
        detval = detval[wavelengths[1]]
    end
    return (detval, ϕ, Amat, rhs)
end

end