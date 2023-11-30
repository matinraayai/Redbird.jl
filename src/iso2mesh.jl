module iso2mesh

using Combinatorics
using MATLAB
using Debugger
using Zygote

export meshedge, tsearchn
"""
 `edges = meshedge(elem, opt)`

 return all edges in a surface or volumetric mesh

 author: Qianqian Fang, <q.fang at neu.edu>
 date: 2011/02/26

 input:
    elem:  element table of a mesh (support N-d space element)
    opt: optional input, giving the additional options. If opt
         is a struct, it can have the following field:
       opt.nodeorder: if 1, assuming the elem node indices is in CCW 
                      orientation; 0 use nchoosek() output to order edges
         you can replace opt by a series of ('param', value) pairs.

 output:
    edge:  edge list; each row is an edge, specified by the starting and
           ending node indices, the total edge number is
           size(elem,1) x nchoosek(size(elem,2),2). All edges are ordered
           by looping through each element first. 
"""

function meshedge(elem)
    dim = size(elem)
    edgeid = collect(combinations(1:dim[2], 2))
    len = size(edgeid, 1)
    edges = Zygote.Buffer(zeros(dim[1] * len, 2))
    for i = 0:len-1
        edges[(i*dim[1]+1):((i+1)*dim[1]), :] = [elem[:, edgeid[i+1][1]] elem[:, edgeid[i+1][2]]]
    end
    return copy(edges)
end


"""
## Conversion of Cartesian to Barycentric coordinates.
## Given a reference simplex in N dimensions represented by an
## N+1-by-N matrix, an arbitrary point P in Cartesian coordinates,
## represented by an N-by-1 column vector can be written as
##
## P = Beta * T
##
## Where Beta is an N+1 vector of the barycentric coordinates.  A criteria
## on Beta is that
##
## sum (Beta) == 1
##
## and therefore we can write the above as
##
## P - T(end, :) = Beta(1:end-1) * (T(1:end-1,:) - ones (N,1) * T(end,:))
##
## and then we can solve for Beta as
##
## Beta(1:end-1) = (P - T(end,:)) / (T(1:end-1,:) - ones (N,1) * T(end,:))
## Beta(end) = sum (Beta)
##
## Note code below is generalized for multiple values of P, one per row.
"""
function cart2bary(T, P)
    (M, N) = size(P)
    T_end = T[end, :]
    T_end = reshape(T_end, (1, size(T_end)...))
    Beta = (P - ones(M, 1) * T_end) / (T[1:end-1, :] - ones(N, 1) * T_end)
    Beta = hcat(Beta, 1 .- sum(Beta, dims=2))
    return Beta
end

"""
########################################################################
##
## Copyright (C) 2007-2022 The Octave Project Developers
##
## See the file COPYRIGHT.md in the top-level directory of this
## distribution or <https://octave.org/copyright/>.
##
## This file is part of Octave.
##
## Octave is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## Octave is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not, see
## <https://www.gnu.org/licenses/>.
##
########################################################################

## -*- texinfo -*-
## @deftypefn  {} {@var{idx} =} tsearchn (@var{x}, @var{t}, @var{xi})
## @deftypefnx {} {[@var{idx}, @var{p}] =} tsearchn (@var{x}, @var{t}, @var{xi})
## Search for the enclosing Delaunay convex hull.
##
## For @code{@var{t} = delaunayn (@var{x})}, finds the index in @var{t}
## containing the points @var{xi}.  For points outside the convex hull,
## @var{idx} is NaN.
##
## If requested @code{tsearchn} also returns the Barycentric coordinates
## @var{p} of the enclosing triangles.
## @seealso{delaunay, delaunayn}
## @end deftypefn
"""
function tsearchn(x, t, xi)

    nt = size(t, 1)
    (m, n) = size(x)
    mi = size(xi, 1)
    idx = NaN * ones(mi, 1)
    p = NaN * ones(mi, n + 1)

    ni = collect(1:mi)
    for i = 1:nt
        ## Only calculate the Barycentric coordinates for points that have not
        ## already been found in a triangle.
        b = cart2bary(x[t[i, :], :], xi[ni, :])

        ## Our points xi are in the current triangle if
        ## (all (b >= 0) && all (b <= 1)).  However as we impose that
        ## sum (b,2) == 1 we only need to test all(b>=0).  Note need to add
        ## a small margin for rounding errors
        # intri = all(>=(-1e-12), b, dims=2)
        # idx(ni(intri>0)) = i;
        # p(ni(intri),:) = b(intri, :);
        # ni(intri) = [];
        # idx[ni] .= i
        # @show size(b), size(p[ni, :])
        # p[:] .= b[:]
        intri = all(>=(-1e-12), b, dims=2)[:]
        idx[ni[intri[:]]] .= i
        p[ni[intri[:]], :] = b[intri[:], :]
        # p(ni(intri),:) = b(intri, :);
        deleteat!(ni, intri)
        # ni[]
        # ni(intri) = [];
        # ni[intri] = []
    end
    return (idx, p)
end




end