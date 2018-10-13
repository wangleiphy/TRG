using LinearAlgebra:svd
using TensorOperations

function TRG(K, Dcut, no_iter)
    D = 2
    inds = collect(1:D) 

    T = zeros(Float64, D, D, D, D)
    for r in inds, u in inds, l in inds, d in inds
        T[r, u, l, d] = 0.5*(1 + (2*r-3)*(2*u-3)*(2*l-3)*(2*d-3))*exp(2*K*(r+u+l+d-6))
    end
    
    lnZ = 0.0 
    for n in collect(1:no_iter)
        D_new = min(D^2, Dcut)
        inds_new = collect(1:D_new)

        Ma = zeros(Float64, D^2, D^2)
        Mb = zeros(Float64, D^2, D^2)
        for r in inds, u in inds, l in inds, d in inds
            Ma[l + D*(u-1), r + D*(d-1)] = T[r, u, l, d]
            Mb[l + D*(d-1), r + D*(u-1)] = T[r, u, l, d]
        end

        S1 = zeros(Float64, D, D, D_new)
        S2 = zeros(Float64, D, D, D_new)
        S3 = zeros(Float64, D, D, D_new)
        S4 = zeros(Float64, D, D, D_new)

        U, L, V = svd(Ma)
        for x in inds, y in inds, m in inds_new
            S1[x, y, m] = sqrt(L[m]) * U[x+D*(y-1), m]
            S3[x, y, m] = sqrt(L[m]) * V[m, x+D*(y-1)]
        end 
        U, L, V = svd(Mb)
        for x in inds, y in inds, m in inds_new
            S2[x, y, m] = sqrt(L[m]) * U[x+D*(y-1), m]
            S4[x, y, m] = sqrt(L[m]) * V[m, x+D*(y-1)]
        end 

        T_new = zeros(Float64, D_new, D_new, D_new, D_new)
        for r in inds_new, u in inds_new, l in inds_new, d in inds_new
            for a in inds, b in inds, g in inds, w in inds
                T_new[r, u, l, d] += S1[w, a, r] * S2[a, b, u] * S3[b, g, l] * S4[g, w, d]
            end
        end
        #@tensor T_new[r, u, l, d] = S1[w, a, r] * S2[a, b, u] * S3[b, g, l] * S4[g, w, d]

        D = D_new
        inds = inds_new 
        #maxval = maximum(T_new)
        #lnZ += 2*(no_iter-n)*log(maxval)
        #T = T_new/maxval
        T = T_new
    end
    println(sum(T))
    lnZ += log(sum(T)) 
end

lnZ = TRG(0.44, 10, 1)
println(lnZ)
