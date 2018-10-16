using LinearAlgebra:svd, Diagonal
using TensorOperations

function TRG(K::Float64, Dcut::Int, no_iter::Int)
    D = 2
    inds = collect(1:D) 

    T = zeros(Float64, D, D, D, D)
    M = [[sqrt(cosh(K)) sqrt(sinh(K))]; 
         [sqrt(cosh(K)) -sqrt(sinh(K))];
         ]
    for i in inds, j in inds, k in inds, l in inds
        for a in inds
            T[i, j, k, l] += M[a, i] * M[a, j] * M[a, k] * M[a, l]
        end
    end

    lnZ = 0.0 
    for n in collect(1:no_iter)
        
        #println(n, " ", maximum(T), " ", minimum(T))
        maxval = maximum(T)
        T = T/maxval 
        lnZ += 2^(no_iter-n+1)*log(maxval)

        D_new = min(D^2, Dcut)

        Ma = reshape(permutedims(T, (3, 2, 1, 4)),  (D^2, D^2))
        Mb = reshape(permutedims(T, (4, 3, 2, 1)),  (D^2, D^2))

        F = svd(Ma)
        S1 = reshape(F.U[:,1:D_new]*Diagonal(sqrt.(F.S[1:D_new])), (D, D, D_new))
        S3 = reshape(Diagonal(sqrt.(F.S[1:D_new]))*F.Vt[1:D_new, :], (D_new, D, D))
        F = svd(Mb)
        S2 = reshape(F.U[:,1:D_new]*Diagonal(sqrt.(F.S[1:D_new])), (D, D, D_new))
        S4 = reshape(Diagonal(sqrt.(F.S[1:D_new]))*F.Vt[1:D_new, :], (D_new, D, D))

        @tensor T_new[r, u, l, d] := S1[w, a, r] * S2[a, b, u] * S3[l, b, g] * S4[d, g, w]

        D = D_new
        T = T_new
    end
    trace = 0.0
    for i in inds
        trace += T[i, i, i, i]
    end
    lnZ += log(trace)
end

Dcut = 24
n = 20

for K in collect(0.0:0.1:2.0)
    lnZ = TRG(K, Dcut, n)
    println(K, " ", lnZ/2^n)
end 
