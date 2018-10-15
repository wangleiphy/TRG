using LinearAlgebra:svd
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
        inds_new = collect(1:D_new)

        #Ma = zeros(Float64, D^2, D^2)
        #Mb = zeros(Float64, D^2, D^2)
        #for r in inds, u in inds, l in inds, d in inds
        #    Ma[l + D*(u-1), r + D*(d-1)] = T[r, u, l, d]
        #    Mb[l + D*(d-1), r + D*(u-1)] = T[r, u, l, d]
        #end
        Ma = reshape(permutedims(T, (3, 2, 1, 4)),  (D^2, D^2))
        Mb = reshape(permutedims(T, (3, 4, 1, 2)),  (D^2, D^2))

        S1 = zeros(Float64, D, D, D_new)
        S2 = zeros(Float64, D, D, D_new)
        S3 = zeros(Float64, D, D, D_new)
        S4 = zeros(Float64, D, D, D_new)

        F = svd(Ma)
        for x in inds, y in inds, m in inds_new
            S1[x, y, m] = sqrt(F.S[m]) * F.U[x+D*(y-1), m]
            S3[x, y, m] = sqrt(F.S[m]) * F.Vt[m, x+D*(y-1)]
        end 
        F = svd(Mb)
        for x in inds, y in inds, m in inds_new
            S2[x, y, m] = sqrt(F.S[m]) * F.U[x+D*(y-1), m]
            S4[x, y, m] = sqrt(F.S[m]) * F.Vt[m, x+D*(y-1)]
        end 

        #T_new = zeros(Float64, D_new, D_new, D_new, D_new)
        #for r in inds_new, u in inds_new, l in inds_new, d in inds_new
        #    for a in inds, b in inds, g in inds, w in inds
        #        T_new[r, u, l, d] += S1[w, a, r] * S2[a, b, u] * S3[b, g, l] * S4[g, w, d]
        #    end
        #end
        @tensor T_new[r, u, l, d] := S1[w, a, r] * S2[a, b, u] * S3[b, g, l] * S4[g, w, d]

        D = D_new
        inds = inds_new 
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
