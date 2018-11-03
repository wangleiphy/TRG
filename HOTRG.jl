using TensorOperations
using LinearAlgebra
using JLD2
function IsingMatrix(K::Float64)
    M = [[sqrt(cosh(K)) sqrt(sinh(K))];
         [sqrt(cosh(K)) -sqrt(sinh(K))];
         ]
end
function Ising(K::Float64)
    D = 2;
    inds = 1:D;
    T = zeros(Float64, D, D, D, D)
    M = IsingMatrix( K )
    for i in inds, j in inds, k in inds, l in inds
        for a in inds
            T[i, j, k, l] += M[a, i] * M[a, j] * M[a, k] * M[a, l]
        end
    end
    T;
end
function Gauge( T::Array{Float64} , Dcut::Int, s::Char)
    # T is a D*D*D*D tensor u l d r
    if s == 'l' || s == 'L'
        @tensor M_l[a,A,c,C] :=  (T[x,a,z,b] * T[x,c,w,b])* (T[w,C,y,B] * T[z,A,y,B]);
        #@show M_l
        D = size(M_l,1);
        M_l = reshape(M_l, (D*D,D*D));
        M_l = ( M_l + M_l') / 2;

        #@show M_l

    #    println("M_l nsym:", norm(M_l - M_l'))
        vl, Ul = eigen( M_l);
        D_new = min(D^2, Dcut)
        inds_new = collect(1:D_new)
        p = sortperm(vl,rev=true)
        TrunErrLeft = 1 - sum( vl[p[inds_new]]) / sum( vl)
        vl = vl[p[inds_new]]
        Ul = Ul[:,p[inds_new]]
        Ul = reshape( Ul, (D,D,D_new))
        return Ul, TrunErrLeft
    elseif s == 'r' || s == 'R'
        @tensor M_r[a,A,c,C] :=  (T[x,b,z,a] * T[x,b,w,c])* (T[w,B,y,C] * T[z,B,y,A]);
        #@show M_l
        D = size(M_r,1);
        M_r = reshape(M_r, (D*D,D*D));
        M_r = ( M_r + M_r')/2;
        vr, Ur = eigen( M_r);
        D_new = min(D^2, Dcut)
        inds_new = collect(1:D_new)
        p = sortperm(vr,rev=true)
        TrunErrRight = 1 - sum( vr[p[inds_new]]) / sum( vr)
        vr = vr[p[inds_new]]
        Ur = Ur[:,p[inds_new]]
        Ur = reshape( Ur, (D,D,D_new))
        return Ur, TrunErrRight
    end
end

function HOTRG(T::Array{Float64} , Dcut::Int, no_iter::Int)
    lnZ = 0.0
    for k = 1:no_iter
        Ul, TrunErrLeft = Gauge(T, Dcut,'l')
        Ur, TrunErrRight = Gauge(T,Dcut,'r')
        U = TrunErrLeft < TrunErrRight ? Ul : Ur
        @tensoropt T[w,x,z,y] := T[x,a,o,b] * U[a,A,z] * T[o,A,y,B] * U[b,B,w]
        f = norm(T)
        lnZ += log(f) / 2^k
        T = T / f
    #    println(lnZ)
    end
    sum = 0.0;
    D1 = size(T,1)
    D2 = size(T,2)
    for x = 1:D1, y = 1:D2
        sum += T[x,y,x,y]
    end
    @show sum
    lnZ += log(sum)/2^no_iter
    return lnZ
end
function OneIsingTensor( Ku::Float64, Kl::Float64, Kd::Float64,Kr::Float64)
    T = zeros(Float64, 2, 2, 2, 2)
    Mu = IsingMatrix( Ku )
    Md = IsingMatrix( Kd )
    Ml = IsingMatrix( Kl )
    Mr = IsingMatrix( Kr )
    inds = 1:2;
    for i in inds, j in inds, k in inds, l in inds
        for a in inds
            T[i, j, k, l] += Mu[a, i] * Ml[a, j] * Md[a, k] * Mr[a, l]
        end
    end
    return T
end
function SpinGlassTensor( Ks::Array{Tuple{Float64,Float64},2}, ix::Int,iy::Int)
    # Construct the Tensor at site (i,j)
    # Ks is a Nx * Ny matrix of all the coefficients (Jx,Jy)
    # Square lattice of size (Lx,Ly) PBC
    #   1,1 -- 1,2 -- ... 1,Ly ---
    #    |      |           |
    #   2,1 -- 2,2 -- ... 2,Ly ---
    #    |      |           |
    #    .      .     ...   .
    #  Lx,1 -- Lx,2 .... Lx,Ly ---
    #    \       |          |
    Lx, Ly = size( Ks )
    Kd = Ks[ix,iy][2]
    Kr = Ks[ix,iy][1]
    Ku = Ks[ix,iy ==1 ? Ly : iy-1][2]
    Kl = Ks[ ix ==1 ? Lx : ix-1, iy][1]
    T = OneIsingTensor( Ku, Kl, Kd, Kr )
end
function test()
Dcut = 30
n = 30
@show "=====HOTRG======"
ts = 0.1:0.1:3;
β = inv.(ts);
#lnZ = zeros(size(β))
lnZ = []
for K in β
    T = Ising( K )
    y = HOTRG(T, Dcut, n)
    #@show lnZ
    println(1/K, " ", y)
    push!(lnZ,y)
end
F = - ts.* lnZ
end
