function TRG(K, Dcut, no_iter)
    D = 2
    inds = collect(1:D) 

    T = zeros(Float64, D, D, D, D)
    for r, u, l, d in collect(Iterators.product(inds, inds, inds, inds))
        T[r, u, l, d] = 0.5*(1 + (2*r-1)*(2*u-1)*(2*l-1)*(2*d-1))*exp(2*K*(r+u+l+d-2))
    end

    for n in collect(no_iter)
        D_new = min(D^2, Dcut)
        inds_new = collect(1:D_new)

        Ma = zeros(Float64, D^2, D^2)
        Mb = zeros(Float64, D^2, D^2)
        for r, u, l, d in collect(Iterators.product(inds, inds, inds, inds))
            Ma[l + D*u, r + D*d] = T[r, u, l, d]
            Mb[l + D*d, r + D*u] = T[r, u, l, d]
        end

        S1 = zeros(Float64, D, D, D_new)
        S2 = zeros(Float64, D, D, D_new)
        S3 = zeros(Float64, D, D, D_new)
        S4 = zeros(Float64, D, D, D_new)

        U, L, V = svd(Ma)
        L = sort(L)[::-1][1:D_new]
        for x, y, m in product(inds, inds, inds_new)
            S1[x, y, m] = sqrt(L[m]) * U[x+D*y, m]
            S3[x, y, m] = sqrt(L[m]) * V[m, x+D*y]
        end 
        U, L, V = svd(Mb)
        L = sort(L)[::-1][1:D_new]
        for x, y, m in product(inds, inds, inds_new)
            S2[x, y, m] = sqrt(L[m]) * U[x+D*y, m]
            S4[x, y, m] = sqrt(L[m]) * V[m, x+D*y]
        end 

        T_new = zeros(Float64, D_new, D_new, D_new)
        for r, u, l, d in product(inds, inds, inds, inds) 
            for a, b, g, w in product(inds, inds, inds, inds) 
                T_new[r, u, l, d] += S1[w, a, r] * S2[a, b, u] * S3[b, g, l] * S4[g, w, d]
            end
        end

        D = D_new
        inds = inds_new 
        T = T_new 
 
    end

    Z = 0.0
    for r, u, l, d in product(inds, inds, inds, inds) 
        Z += T[r, u, l, d]
    end

    Z
end

TRG(1.0, 6, 3)
