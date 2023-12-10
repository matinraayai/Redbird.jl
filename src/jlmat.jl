"""
Contains Julia bindings to Redbird-M, and iso2mesh. They cannot be used directly 
for differentiation with Enzyme or ChainRules.jl. They can, however, be used with
FiniteDiff.jl. Functions requiring differentiation are already ported to Julia. 
They are also used for testing the Julia implementations.
"""
#TODO Can this work with Octave.jl?
module JlMatBindings

using MATLAB

using ..Structs: RBConfig

is_redbird_m_initialized = false

is_iso2mesh_initialized = false


function init_redbird_m()
    if !is_redbird_m_initialized
        redbird_m_path = Base.Filesystem.joinpath(@__DIR__(), "redbird-m/matlab/")
        @show redbird_m_path
        mat"addpath($redbird_m_path)"
        global is_redbird_m_initialized = true
    end
end

function init_iso2mesh()

    if !is_iso2mesh_initialized
        iso2mesh_path = Base.Filesystem.joinpath(@__DIR__(), "iso2mesh/")
        @show iso2mesh_path
        mat"addpath($iso2mesh_path)"
        global is_iso2mesh_initialized = true
    end
end

function jlcfg2mx(cfg::RBConfig, prop::Union{Array{<:AbstractFloat}, Nothing} = nothing)
    out = Dict{Symbol, Any}()
    for p ∈ keys(cfg)
        # Change the keys to be MATLAB compatible
        k = begin
            if p == :∇ϕ_i∇ϕ_j
                :deldotdel
            elseif p == :ω
                :omega
            elseif p == :∇ϕ
                :detphi
            elseif p == :μ_sp0
                :musp0
            else
                p
            end
        end
        v = getproperty(cfg, p)
        if !isnothing(v)
            out[k] = v
        end
    end

    if !isnothing(prop)
        out[:prop] = prop
    end

    return out
end

function meshreorient(node::AbstractArray{<:AbstractFloat}, elem::AbstractArray{<:Integer})
    init_iso2mesh()
    (newelem, evol, idx) = mxcall(:meshreorient, 3, node, elem)
    newelem = Int.(newelem)
    return (newelem, evol, idx)
end

function volface(t::AbstractArray{<:Number})
    init_iso2mesh()
    openface, elemid = mxcall(:volface, 2, t)
    return (openface, elemid)
end

function elemvolume(node::AbstractArray{<:Number}, elem::AbstractArray{<:Number}, option::Symbol = :unsigned)
    init_iso2mesh()
    return mxcall(:elemvolume, 1, node, elem, String(option))
end


function nodevolume(node, elem, evol)
    init_iso2mesh()
    return mxcall(:nodevolume, 1, node, elem, evol)
end

function meshabox(p0, p1, opt, nodesize=1.)
    init_iso2mesh()
    (node, face, elem) = mxcall(:meshabox, 3, p0, p1, opt, nodesize)
    face = Int.(face)
    elem = Int.(elem)
    return (node, face, elem)
end

function rbgetreff(n_in, n_out)
    init_redbird_m()
    return mxcall(:rbgetreff, 1, n_in, n_out)
end


function rbsrc2bc(cfg::RBConfig, prop::Union{Array{<:AbstractFloat}, Nothing}=nothing, isdet=0)
   init_redbird_m()
   return mxcall(:rbsrc2bc, 1, jlcfg2mx(cfg, prop), isdet) 
end

function rbfemnz(elem, nn)
    init_redbird_m()
    return mxcall(:rbfemnz, 3, elem, nn)
end

function rbfemlhs(cfg, prop, ∇ϕ_i∇ϕ_j, wavelength)
    init_redbird_m()
    mat_cfg = jlcfg2mx(cfg, prop)
    Amat_mat, ∇ϕ_i∇ϕ_j_mat = mxcall(:rbfemlhs, 2, mat_cfg, ∇ϕ_i∇ϕ_j, wavelength)
    return Amat_mat, ∇ϕ_i∇ϕ_j_mat
end

function rbmeshprep(cfg::RBConfig, prop=nothing)
    init_redbird_m()
    mat_cfg = jlcfg2mx(cfg, prop)
    return RBConfig(mxcall(:rbmeshprep, 1, mat_cfg))
end

function rb∇̇ϕ_i∇ϕ_j(cfg::RBConfig)
    init_redbird_m()
    mat_cfg = jlcfg2mx(cfg, nothing)
    return mxcall(:rbdeldotdel, 1, mat_cfg)
end

function rbfemrhs(cfg, prop=nothing)
    init_redbird_m()
    mat_cfg = jlcfg2mx(cfg, prop)
    return mxcall(:rbfemrhs, 4, mat_cfg)
end

function rbfemsolve(Amat, rhs, method; kwargs...)
    init_redbird_m()
    method_str = String(method)
    return mxcall(:rbfemsolve, 1, Amat, rhs, method_str)
end

function rbfemgetdet(ϕ, cfg, optodeloc, optodebary)
    init_redbird_m()
    mat_cfg = jlcfg2mx(cfg)
    return mxcall(:rbfemgetdet, 2, ϕ, mat_cfg, optodeloc, optodebary)
end

end