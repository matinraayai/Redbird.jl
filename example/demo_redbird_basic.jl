"""
########################################################
# Redbird - A Diffusion Solver for Diffuse Optical Tomography; 
#      Copyright Qianqina Fang; 2018
#
# In this example; we show the most basic usage of Redbird.
#
# This file is part of Redbird URL:http://mcx.sf.net/mmc
########################################################
"""

using Redbird
using MATLAB

"""
########################################################
##   prepare simulation input()
########################################################
"""

mat"clear cfg"
mat"addpath('./matlab/iso2mesh')"
mat"addpath('./matlab/mcx')"
mat"addpath('/scratch/raayaiardakani.m/Redbird.jl/src/redbird-m/matlab')"

cfg = Redbird.Structs.RBConfig()
cfg.node, cfg.face, cfg.elem = mxcall(:meshabox, 3, [0 0 0], [60 60 30], 1)

nn = size(cfg.node, 1)
cfg.seg = ones(size(cfg.elem, 1), 1)
cfg.srcpos = [30 30 0]
cfg.srcdir = [0 0 1]

cfg.prop = [0 0 1 1;0.005 1 0 1.37]
cfg.omega = 0

cfg = Redbird.Structs.rbmeshprep(cfg)

#save config.mat

########################################################
##   Build LHS
########################################################
(detval, ϕ, Amat, rhs) = Redbird.Forward.rbrunforward(cfg)
# @show size(detval[""]), size(ϕ[""]), size(Amat[""]), size(rhs[""])
@show typeof(detval), typeof(ϕ), typeof(Amat), typeof(rhs)

mat"
    [cfg_mat.node, cfg_mat.face, cfg_mat.elem] = meshabox([0 0 0], [60 60 30], 1);

    nn = size(cfg_mat.node, 1);
    cfg_mat.seg = ones(size(cfg_mat.elem, 1), 1);
    cfg_mat.srcpos = [30 30 0];
    cfg_mat.srcdir = [0 0 1];

    cfg_mat.prop = [0 0 1 1;0.005 1 0 1.37];
    cfg_mat.omega = 0;

    cfg_mat = rbmeshprep(cfg_mat);
    [$detval_mat, $phi_mat, $Amat_mat, $rhs_mat] = rbrunforward(cfg_mat, 'solverflag', {'qmr',});
"
@show typeof(detval_mat), typeof(phi_mat), typeof(Amat_mat), typeof(rhs_mat)
@show sum(Amat_mat), sum(Amat)
@show sum(rhs_mat), sum(rhs)
@show sum(phi_mat), sum(ϕ), sum(phi_mat - ϕ), sum(phi_mat ./ ϕ)
@show detval, detval_mat
#print("forward solution ... \t$(forward_stats.time) seconds\n")

########################################################
##   Analytical solution
########################################################

# if(exist("cwdiffusion','file"))
#     srcpos = [30 30 0]
#     phicw = cwdiffusion[cfg.prop[2,1], cfg.prop[2,2] * (1 - cfg.prop[2, 3]), cfg.reff, srcpos, cfg.node]
# else()
#     warning("please download MCX from http://mcx.space to use cwdiffuse.m in mcx/utils")
# end
# ########################################################
# ##   Visualization
# ########################################################

# clines = 0:-0.5:-5
# (xi, yi) = mxcall(:meshgrid, 2, 0.5:59.5,0.5:29.5)
# [cutpos,cutvalue] = qmeshcut[cfg.elem,cfg.node,phi[:,1],"x=29.5"]
# vphi = griddata(cutpos[:,2],cutpos[:,3],cutvalue,xi+0.5,yi)

# figure,[c,h] = contour(xi,yi,log10(vphi),clines,"r-','LineWidth",2)

# if(exist("cwdiffusion','file"))
#     [cutpos,cutvalue] = qmeshcut[cfg.elem,cfg.node,phicw[:],"x=29.5"]
#     vphidiffu = griddata(cutpos[:,2],cutpos[:,3],cutvalue,xi+0.5,yi)

#     hold on,contour(xi,yi,log10(vphidiffu),clines,"b-','LineWidth",2)

#     legend("redbird solution [slab]', 'diffusion [semi-infinite]")
# end