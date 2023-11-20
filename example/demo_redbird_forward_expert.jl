########################################################
# Redbird - A Diffusion Solver for Diffuse Optical Tomography, 
#      Copyright Qianqina Fang, 2018
#
# This example shows explicitly the detailed steps of running a forward
# simulation. One can call rbrun or rbrunforward as one-liner alternatives
#
# This file is part of Redbird URL:http://mcx.sf.net/mmc
########################################################
using Redbird
using MATLAB
using Debugger
using SparseArrays

mat"clear cfg"
mat"addpath('./matlab/iso2mesh')"
mat"addpath('./matlab/mcx')"
mat"addpath('/scratch/raayaiardakani.m/Redbird.jl/src/redbird-m/matlab')"

########################################################
##   prepare simulation input
########################################################

cfg = Redbird.Structs.RBConfig()
cfg.node, cfg.face, cfg.elem = mxcall(:meshabox, 3, [0 0 0], [60 60 30], 1)


#[cfg.node, cfg.face, cfg.elem]=meshabox([0 0 0],[60 60 30],3);
nn = size(cfg.node, 1)
cfg.seg = ones(size(cfg.elem, 1), 1)

(xi, yi) = mxcall(:meshgrid, 2, 60:20:140,20:20:100)

cfg.srcpos = hcat(xi[:], yi[:], zeros(length(yi), 1))
cfg.detpos = hcat(xi[:], yi[:], 60 * ones(length(yi), 1))
cfg.srcdir = [0 0 1]
cfg.detdir = [0 0 -1]

cfg.prop=[0     0 1 1
          0.008 1 0 1.37
          0.016 1 0 1.37]


cfg.omega = 2 * pi * 70e6
cfg.omega = 0

t = @timed cfg = Redbird.Structs.rbmeshprep(cfg)


print("preparing mesh took $(t.time) seconds.")

# save config.mat

########################################################
##   Build LHS
########################################################
wavelengths = [""]

if cfg.prop isa Dict
    wavelengths = keys(cfg.prop)
end

t = @timed begin
    ∇s = Redbird.Forward.rb∇̇ϕ_i∇ϕ_j(cfg)
    ∇ϕ_i∇ϕ_j = ∇s.∇ϕ_i∇ϕ_j
    ∇ϕ = ∇s.∇ϕ
    Amat, ∇ϕ_i∇ϕ_j = Redbird.Forward.rbfemlhs(cfg, ∇ϕ_i∇ϕ_j, wavelengths[1])
end
print("LHS using native code: $(t.time) seconds\n")

∇ϕ_i∇ϕ_j_mat, ∇ϕ_mat = mxcall(:rbdeldotdel, 2, cfg.cfg)
Amat_mat, ∇ϕ_i∇ϕ_j_mat = mxcall(:rbfemlhs, 2, cfg.cfg, ∇ϕ_i∇ϕ_j_mat, wavelengths[1])

@show ∇ϕ_i∇ϕ_j_mat ≈ ∇ϕ_i∇ϕ_j
@show ∇ϕ ≈ ∇ϕ_mat
@show typeof(Amat_mat), typeof(Amat), size(Amat_mat), size(Amat), sum(Amat_mat), sum(Amat)
@show nonzeros(Amat) ≈ nonzeros(Amat_mat)
# @show Amat_mat[1:20, 1:20]

# ########################################################
# ##   Build RHS
# ########################################################

(rhs, loc, bary, optode) = Redbird.Forward.rbfemrhs(cfg)
cfg.srcdir = Float64.(cfg.srcdir)
cfg.detpos = Float64.(cfg.detpos)
cfg.detdir = Float64.(cfg.detdir)
(rhs_matlab, loc_matlab, bary_matlab, optode_matlab) = mxcall(:rbfemrhs, 4, cfg.cfg)
@show rhs_matlab ≈ rhs
# @show loc_matlab ≈ loc
@show loc_matlab[(!isnan).(loc_matlab)] ≈ loc[(!isnan).(loc)]
@show bary_matlab[(!isnan).(bary_matlab)] ≈ bary[(!isnan).(bary)]
# @show bary_matlab ≈ bary
@show optode_matlab[(!isnan).(optode_matlab)] ≈ optode[(!isnan).(optode)]
# @show optode_matlab ≈ optode
# ########################################################
# ##   Solve for solutions at all freenodes: Afree*sol=rhs
# ########################################################

# print("solving for the solution ...\n")
ϕ = Redbird.Forward.rbfemsolve(Amat, rhs, :qmr)
t = @timed begin
    #[phi,sflag]=rbfemsolve(Amat,rhs,'pcg',1e-8,200);
    ϕ = Redbird.Forward.rbfemsolve(Amat, rhs, :qmr)
end
print("solving forward solutions ... \t$(t.time) seconds")
t = @timed (phi,sflag) = mxcall(:rbfemsolve, 2, Amat, rhs, "qmr")
print("solving forward solutions ... \t$(t.time) seconds")
@show sum(abs.(ϕ - phi))
# ########################################################
# ##   Extract detector readings from the solutions
# ########################################################

(detval, goodix) = Redbird.Forward.rbfemgetdet(ϕ, cfg, loc, bary)
(detval_matlab, goodidx_matlab) = mxcall(:rbfemgetdet, 2, ϕ, loc_matlab, bary_matlab)

@show size(detval), detval_matlab
@show goodix, goodidx_matlab
# or detval=rbfemgetdet(phi, cfg, rhs); 

# # ########################################################
# # ##   Analytical solution
# # ########################################################

# # sid = 13;

# # srcloc=cfg.srcpos(sid,1:3);
# # detloc=cfg.node;

# # phicw=cwdiffusion(cfg.prop(2,1), cfg.prop(2,2)*(1-cfg.prop(2,3)), cfg.reff, srcloc, detloc);

# # ########################################################
# # ##   Visualization
# # ########################################################

# # figure;
# # subplot(221);plotmesh([cfg.node,log10(abs(phi(1:size(cfg.node,1),sid)))],cfg.elem,'y=30','facecolor','interp','linestyle','none')
# # cl=get(gca,'clim');
# # set(gca, 'xlim', [60, 140]);
# # set(gca, 'zlim', [0 60]);
# # view([0 1 0]);
# # colorbar;

# # subplot(222);plotmesh([cfg.node,log10(abs(phicw(1:size(cfg.node,1),1)))],cfg.elem,'y=30','facecolor','interp','linestyle','none')
# # view([0 1 0]);
# # set(gca, 'xlim', [60, 140]);
# # set(gca, 'zlim', [0 60]);
# # set(gca, 'clim', cl);
# # colorbar;

# # dd=log10(abs(phi(1:size(cfg.node,1),sid))) - log10(abs(phicw(1:size(cfg.node,1),1)));
# # subplot(223);plotmesh([cfg.node,dd],cfg.elem,'y=30','facecolor','interp','linestyle','none')
# # view([0 1 0]);
# # set(gca, 'xlim', [60, 140]);
# # set(gca, 'zlim', [0 60]);
# # colorbar;

# # subplot(224);plotmesh([cfg.node,dd],cfg.elem,'y=30','facecolor','interp','linestyle','none')
# # hist(dd(:),100);

# # ## test add-noise function

# # dist=rbgetdistance(cfg.srcpos, cfg.detpos);
# # plot(dist(:),log10(abs(detval(:))),'.');
# # newdata=rbaddnoise(detval, 110,40);
# # hold on;
# # plot(dist(:),log10(abs(newdata(:))),'r.');
# # hold off;
