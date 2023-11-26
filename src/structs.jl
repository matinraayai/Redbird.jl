module Structs

export Float3, Float4, Integer3, Integer4, Medium, Config, TetMesh, RbForward, RBConfig

AbstractVectorType{N, T} = NTuple{N, VecElement{T}} where{N, T <: Number}

AbstractInt3{T} = AbstractVectorType{3, T} where {T <: Integer}
AbstractInt4{T} = AbstractVectorType{4, T} where {T <: Integer}
AbstractFloat3{T} = AbstractVectorType{3, T} where {T <: AbstractFloat}
AbstractFloat4{T} = AbstractVectorType{4, T} where {T <: AbstractFloat}


Short3 = AbstractInt3{Int16}
Int3 = AbstractInt3{Int32}
Long3 = AbstractInt3{Int64}
LongLong3 = AbstractInt3{Int128}

Short4 = AbstractInt4{Int16}
Int4 = AbstractInt4{Int32}
Long4 = AbstractInt4{Int64}
LongLong4 = AbstractInt4{Int128}


Half3 = AbstractFloat3{Float16}
Float3 = AbstractFloat3{Float32}
Double3 = AbstractFloat3{Float64}

Half4 = AbstractFloat4{Float16}
Float4 = AbstractFloat4{Float32}
Double4 = AbstractFloat4{Float64}


struct Medium{T <: AbstractFloat}
    mua::T
    mus::T
    g::T
    n::T
end

struct RBConfig
    cfg::Dict{Symbol, Any}

    function RBConfig(cfg::Dict{Symbol, Any})
        c = new(cfg)
        return c
    end
end

function RBConfig()
    return RBConfig(Dict{Symbol, Any}())
end


function RBConfig(cfg::Dict{String, Any})
    c = RBConfig()
    for (k, v) ∈ cfg
        c.cfg[Symbol(k)] = v
    end
    return c
end


function Base.getproperty(value::RBConfig, name::Symbol)
    if name == :cfg
        return getfield(value, name)
    else
        return getfield(value, :cfg)[name]
    end
end

function Base.setproperty!(value::RBConfig, name::Symbol, x)
    if name == :cfg
        return setfield!(value, name, x)
    else
        return getfield(value, :cfg)[name] = x
    end
end

function Base.hasproperty(x::RBConfig, s::Symbol)
    if s == :cfg
        return true
    else
        return s ∈ keys(x.cfg)
    end
end

Base.delete!(x::RBConfig, s::Symbol) = return Base.delete!(x.cfg, s)

Optional{T} = Union{T, Nothing}

# Base.convert(::Type{T}, x::Optional{T}) where {T} = T(x)

mutable struct Bulk{FT <: AbstractFloat}
    mua::Optional{FT}
    dcoeff::Optional{FT}
    musp::Optional{FT}
    g::Optional{FT} 
    n::Optional{FT}
end


# Make all the types of the struct below Optional

@kwdef mutable struct RBConfigJL
    node::Optional{Array{Float64}} = nothing
    elem::Optional{Array{Int}} = nothing
    face::Optional{Array{Int}} = nothing
    wavelengths::Optional{Vector{String}} = nothing
    seg::Optional{Array{Int}} = nothing
    area::Optional{Vector{Float64}} = nothing
    isreoriented::Bool = false
    idxcount::Optional{Matrix{Float64}} = nothing
    evol::Optional{Vector{Float64}} = nothing
    detdir::Optional{Matrix{Int}} = nothing
    idxsum::Optional{Matrix{Float64}} = nothing
    detpos::Optional{Matrix{Float64}} = nothing
    detpos0::Optional{Matrix{Float64}} = nothing
    dettype::Optional{Symbol} = nothing
    cols::Optional{Matrix{Float64}} = nothing
    srcdir::Optional{Matrix{Int}} = nothing
    ω::Optional{Matrix{Float64}} = nothing
    srcpos::Optional{Matrix{Float64}} = nothing
    srcpos0::Optional{Matrix{Float64}} = nothing
    srctype::Optional{Symbol} = nothing
    rows::Optional{Matrix{Float64}} = nothing
    nvol::Optional{Array{Float64}} = nothing
    ∇ϕ_i∇ϕ_j::Optional{Matrix{Float64}} = nothing
    ∇ϕ::Optional{Array{Float64}} = nothing
    reff::Optional{Array{Float64}} = nothing
    μ_sp0::Optional{Array{Float64}} = nothing
    bulk::Optional{Bulk{Float64}} = nothing
end

# function RBConfigJL() 
#     return RBConfigJL{Float64, Int64}(
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         false,
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         Symbol(),
#         [],
#         [],
#         [],
#         [],
#         [],
#         Symbol(),
#         [],
#         [],
#         [],
#         [],
#         [],
#         [],
#         Bulk{Float64}(nothing, nothing, nothing, nothing, nothing),
#     )
# end

struct Config{T <: AbstractFloat}
    omega::T
    reff::T
    srcpos::T
    srcdir::T
end

struct TetMesh
    nn::Int     # < number of nodes
    ne::Int     # < number of elements
    nf::Int     # < number of surface triangles
    prop::Int   # < number of media
    ntype::Int  # < number of property indices, ie. type length
    ntot::Int
    e0::Int
    isreoriented::Int
    node::Array{Float3, 1}      # < node coordinates
    face::Array{Int3, 1}    # < boundary triangle node indices
    elem::Array{Int4, 1}    # < tetrahedron node indices
    type::Array{Int, 1}         # < element or node-based media index
    med::Array{Medium, 1}       # < optical property of different media
    evol::Array{Float64, 1}     # < volume of an element
    area::Array{Float64, 1}     # < area of the triangular face
    rows::Array{Int, 1}
    cols::Array{Int, 1}
    idxcount::Array{Int, 1}
    idxsum::Array{Int, 1}
end

struct RbForward
    mesh::Array{TetMesh, 1}
    Ar::Array{Float64, 1}
    Ai::Array{Float64, 1}
    Dr::Array{Float64, 1}
    Di::Array{Float64, 1}
    deldotdel::Array{Float64, 1}
end

struct Jacobian
    isnodal::Int
    nsd::Int
    nsdcod::Int
    nn::Int
    ne::Int
    Phir::Array{Float64, 1}
    Phii::Array{Float64, 1}
    Jmuar::Array{Float64, 1}
    Jdr::Array{Float64, 1}
    Jdi::Array{Float64, 1}
    sd::Array{Float64, 1}
    deldotdel::Array{Float64, 1}
    elemid::Array{Float64, 1}
    elembary::Array{Float64, 1}
    relem::Array{Float64, 1}
end

# function rbmeshprep(cfg::RBConfig)
#     mat"addpath($(redbird_m_path))"
#     cfg = RBConfig(mxcall(:rbmeshprep, 1, cfg.cfg))
#     for (k, v) ∈ cfg.cfg
#         @show k, typeof(v)
#     end
#     if hasproperty(cfg, :deldotdel)
#         cfg.∇ϕ_i∇ϕ_j = cfg.deldotdel
#         # delete!(cfg, :deldotdel)
#     end
#     return cfg
# end 

# """

#  `prop=rbupdateprop(cfg)`

#  Update the direct material properties (cfg.prop - optical | EM 
#  properties used by the forward solver) using multispectral properties 
#  (cfg.param - physiological parameters) for all wavelengths

#  author: Qianqian Fang [q.fang <at> neu.edu]

#  input:
#      `cfg`: the redbird data structure 

#  output:
#      `prop`: the updated material property data to replace cfg.prop

#  license:
#      BSD | GPL version 3; see LICENSE_GPLv3.txt files for details 

#  -- this function is part of Redbird-m toolbox


# """
# function rbupdateprop(cfg::RBConfig, wv=nothing)

    
#     # for single-wavelength
#     if !hasfield(cfg, :param) && (!( cfg.prop isa Dict) || length(cfg.prop) ==1)
#         prop = cfg.prop
#         return
#     end
    
#     # fieldnames(cfg.param) provides chromorphore species, 2nd input wv | 
#     # keys[cfg.prop] provides the wavelength list; cfg.prop is a Dict
#     # object & cfg.param is a struct.
    
#     if(!hasfield(cfg, :prop) || !(cfg.prop isa Dict))
#         error("input cfg must be a struct and must have subfield names $prop & $param")
#     end
    
#     if isnothing(wv)
#         wv = keys(cfg.prop)
#     end
    
#     prop = Dict()
    
#     for wavelen ∈ wv
#         types = intersect(propertynames(cfg.param), (:hbo, :hbr, :water, :lipids, :aa3))
#         if isempty(types)
#             error("specified parameters are not supported")
#         end
#         extin = rbextinction(parse(ComplexF64, wavelen), types)
#         mua = zeros(size(getproperty(cfg.param, types[1])))
#         for (j, type) ∈ types
#             mua = mua + extin[j] * getproperty(cfg.param, type)
#         end
#         if hasfield(cfg.param, :scatamp) && hasfield(cfg.param, :scatpow)
#             musp = getproperty(cfg.param, :scatamp) .* ((parse(ComplexF64, wavelen) / 500) .^ (- getproperty(cfg.param, :scatpow)))
#         end
#         segprop = cfg.prop[wavelen]
#         if length(mua) < min(size(cfg.node,1), size(cfg.elem,1)) # label-based properties
#             segprop[length(mua) + 2: end, :] = []
#             segprop[2: end, 1] = mua[:]
#             if exist("musp","var")
#                 segprop[2:end,2]=musp[:]
#                 segprop[2:end,3]=0
#             end
#             prop[wavelen]=segprop
#         else # mua & musp are defined on the node | elements
#             if exist("musp", "var")
#                 prop[wavelen] = [mua[:] musp[:] zeros(size(musp[:])) segprop[2,4] * ones(size(musp[:]))]
#             else
#                 segprop = repeat(segprop[2, 2:end], length(mua), 1)
#                 prop[wavelen] = [mua[:] segprop]
#             end
#         end
#     return prop
# end


# """

#  Reff = rbgetreff(n_in,n_out)

#  given refractive index of the diffuse medium, calculate the effective
#  refractive index, defined as in Haskell 1994 paper.

#  author: David Boas <dboas at bu.edu>

#  input:
#      n_in: the refractive index n of the interior of the domain
#      n_out: the refractive index n of the outside space

#  output:
#      Reff: effective reflection coefficient, see 

#  license:
#      GPL version 3, see LICENSE_GPLv3.txt files for details 

#     original file name calcExtBnd
#     this file was modified from the PMI toolbox
#  -- this function is part of Redbird-m toolbox
# """
# function rbgetreff(n_in, n_out=1)
#     oc = asin(1/n_in)
#     ostep = pi / 2000
    
#     o = 0:ostep:oc
    
#     cosop = (1-n_in^2 * sin(o).^2).^0.5
#     coso = cos(o)
#     r_fres = 0.5 * ( (n_in*cosop-n_out*coso)./(n_in*cosop+n_out*coso) ).^2
#     r_fres = r_fres + 0.5 * ( (n_in*coso-n_out*cosop)./(n_in*coso+n_out*cosop) ).^2
    
#     r_fres[ceil(oc/ostep):1000] = 1
    
#     o = 0:ostep:ostep * (length(r_fres)-1)
#     coso = cos(o)
    
#     r_phi_int = 2 * sin(o) .* coso .* r_fres
#     r_phi = sum(r_phi_int) / 1000 * pi/2
    
#     r_j_int = 3 * sin(o) .* coso.^2 .* r_fres
#     r_j = sum(r_j_int) / 1000 * pi/2
    
#     Reff = (r_phi + r_j) / (2 - r_phi + r_j)
#     return Reff
# end    


# function rbsrc2bc(cfg::RBConfig, isdet::Bool = false)
# 	# return srcbc
# 	#
# 	# srcbc=rbsrc2bc(cfg)
# 	#
# 	# Converting wide-field source forms into a boundary condition by defining
# 	# in-ward flux on the mesh surface
# 	#
# 	# author: Qianqian Fang (q.fang <at> neu.edu)
# 	#
# 	# input:
# 	#     cfg: the simulation data structure, with srctype, srcpos, srcdir, 
# 	#          srcparam1, srcparam2, srcpattern fields
# 	#     isdet: default is 0; if set to 1, rbsrc2bc process widefield
# 	#          detectors, the relevant fields are dettype, detpos, detdir,
# 	#          detparam1, detparam2, detpattern
# 	#
# 	# output:
# 	#     srcbc: an array of Ns x Nt, where Nt is size(cfg.face,1) and Ns is
# 	#            the number of sources (if isdet=1, the detector counts)
# 	#
# 	# license:
# 	#     GPL version 3, see LICENSE_GPLv3.txt files for details 
# 	#
# 	# -- this function is part of Redbird-m toolbox
# 	#

# 	srcbc = []


# 	if !isdet
# 		if !hasproperty(cfg, :srctype) || cfg.srctype == :pencil || cfg.srctype == :isotropic
# 			return;
# 		end
# 		srctype = cfg.srctype
# 		srcpos = cfg.srcpos
# 		srcdir = cfg.srcdir
# 		srcparam1 = cfg.srcparam1
# 		srcparam2 = cfg.srcparam2
# 		if srctype == :pattern
# 			srcpattern = cfg.srcpattern
# 		end
# 		if hasproperty(cfg, :srcweight)
# 			srcweight = cfg.srcweight
# 		end
# 	else
# 		if !hasproperty(cfg, :dettype) || cfg.dettype == :pencil || cfg.dettype == :isotropic
# 			return
# 		end
# 		srctype = cfg.dettype
# 		srcpos = cfg.detpos
# 		srcdir = cfg.detdir
# 		srcparam1 = cfg.detparam1
# 		srcparam2 = cfg.detparam2
# 		if srctype == :pattern
# 			srcpattern = cfg.detpattern
# 		end
# 		if hasfield(cfg, :detweight)
# 			srcweight = cfg.detweight
# 		end
# 	end

# 	# already converted
# 	if size(srcpos, 2) == size(cfg.face, 1)
# 		srcbc = srcpos
# 		return srcbc
# 	end

# 	z0 = 1 / (cfg.prop[2, 1] + cfg.prop[2, 2] * (1 - cfg.prop[2,3]))
# 	if srctype == :planar || srctype == :pattern || srctype == :fourier
# 			ps = [srcpos; srcpos+srcparam1(1:3); srcpos + srcparam1(1:3) + srcparam2(1:3); srcpos+srcparam2(1:3); srcpos]
			
# 			pnode = cfg.node
# 			pface = cfg.face
			
# 			# if src is colimated (default), sink it by 1/mus'
# 			if !hasproperty(cfg, :iscolimated) || cfg.iscolimated
# 				sinkplane = srcdir  # the plane where sinked planar source is located as [A,B,C,D] where A*x+B*y+C*z+D=0
# 				sinkplane[4] = -sum(srcdir.*(srcpos + srcdir * z0))
# 				[cutpos, cutvalue, facedata, elemid, nodeid] = qmeshcut(cfg.elem, cfg.node, zeros(size(cfg.node,1), 1), sinkplane)
# 				pnode = cutpos
# 				idx = find(facedata[:, 3] !=facedata[:,4])
# 				pface = facedata(facedata[:, 3] == facedata[:,4], 1:3)
# 				pface = [pface;facedata[idx, [1 2 3]];facedata[idx,[1 3 4]]]
# 			end
# 			c0 = meshcentroid(pnode,pface)
# 			newnode = rotatevec3d([c0; ps], srcdir[1:3])
# 			srcpoly = newnode[end - 4: end, 1: 2];
# 			(isin, ison) = inpolygon(newnode[1:end-5,1], newnode[1:end-5,2],srcpoly[:,1], srcpoly[:,2])
# 			isin = isin | ison;
# 			idx = find(isin);
# 			if(~isempty(idx)) # the below test only works for convex shapes
# 				AB = pnode(pface(idx,2),1:3)-pnode(pface(idx,1),1:3);
# 				AC = pnode(pface(idx,3),1:3)-pnode(pface(idx,1),1:3);
# 				N = cross(AB',AC')';
# 				dir = sum(N.*repmat(srcdir(:)',size(N,1),1),2);
# 				if(exist('sinkplane','var'))
# 					dir(dir>0)=-dir(dir>0);
# 				end
# 				if(all(dir>=0))
# 						error('please reorient the surface triangles');
# 				end
# 				srcbc=zeros(1,size(pface,1));
# 				srcbc(idx(dir<0))=1;
# 				pbc=newnode(idx(dir<0),1:2);
				
# 				dp=pbc-repmat(srcpoly(1,:),size(pbc,1),1);
# 				dx=srcpoly(2,:)-srcpoly(1,:);
# 				dy=srcpoly(4,:)-srcpoly(1,:);
# 				nx=dx/norm(dx);
# 				ny=dy/norm(dy);
				
# 				bary=[sum(dp.*repmat(nx/norm(dx),size(dp,1),1),2), sum(dp.*repmat(ny/norm(dy),size(dp,1),1),2)];
# 				bary(bary<0)=0;
# 				bary(bary>=1)=1-1e-6;
				
# 				if(exist('srcpattern','var'))
# 						srcpattern=permute(srcpattern,[3 1 2]);
# 						pdim=size(srcpattern);
# 						patsize=pdim(1);
# 						srcbc=repmat(srcbc,patsize,1);
# 						for i=1:patsize
# 								srcbc(i,idx(dir<0))=srcpattern(sub2ind(pdim, i*ones(size(bary,1),1), floor(bary(:,1)*pdim(2))+1, floor(bary(:,2)*pdim(3))+1));
# 						end
# 				elseif(strcmp(srctype,'fourier'))
# 						kx=floor(srcparam1(4));
# 						ky=floor(srcparam2(4));
# 						phi0=(srcparam1(4)-kx)*2*pi;
# 						M=1-(srcparam2(4)-ky);
# 						srcbc=repmat(srcbc,kx*ky,1);
# 						for i=1:kx
# 								for j=1:ky
# 									srcbc((i-1)*ky+j,idx(dir<0))=0.5*(1+M*cos((i*bary(:,1)+j*bary(:,2))*2*pi+phi0))';
# 								end
# 						end
# 				end
# 			else
# 					error('source direction does not intersect with the domain');
# 			end
# 		else
# 			error("this source type is not supported");
# 	end

# 	# at this point, srcbc stores the J- at each surface triangle (or sinked triangles)

# 	Reff=cfg.reff;
# 	maxbcnode=max(pface(:));
# 	if(exist('nodeid','var'))
# 		nodeweight=nodeid(:,3);
# 		nodeid=nodeid(:,1:2);
# 		maxbcnode=max(max(nodeid(pface,:)));
# 		parea=elemvolume(pnode,pface);
# 	else
# 		parea=cfg.area;
# 	end

# 	# 1/18 = 1/2*1/9, where 2 comes from the 1/2 in ls=(1+Reff)/(1-Reff)/2*D,
# 	# and 1/9 = (1/6+1/12+1/12)/3, where A/6 is <phi_i,phi_j> when i=j, and
# 	# A/12 is i!=j
# 	Adiagbc=parea(:)*((1-Reff)/(18*(1+Reff)));
# 	Adiagbc=repmat(Adiagbc,1,size(srcbc,1)).*(srcbc');
# 	rhs=sparse(size(cfg.node,1),size(srcbc,1));

# 	for i=1:size(srcbc,1)
# 		if(exist('nodeid','var'))
# 			allnodes=nodeid(pface,:);
# 			rhs(1:maxbcnode,i)=sparse(allnodes(:), 1, [repmat(Adiagbc(:,i),3,1).*nodeweight(pface(:));repmat(Adiagbc(:,i),3,1).*(1-nodeweight(pface(:)))]);
# 		else
# 			rhs(1:maxbcnode,i)=sparse(cfg.face(:), 1, repmat(Adiagbc(:,i),1,3));
# 		end
# 		wsrc=1;
# 		if(exist('srcweight','var') && numel(srcweight)==size(srcbc,1))
# 			wsrc=srcweight(i);
# 		end
# 		rhs(:,i)=rhs(:,i)*(wsrc/sum(rhs(:,i)));
# 	end
# 	srcbc=rhs.';



end