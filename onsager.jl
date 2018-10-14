using HCubature

function integrand(x, K) 
    log(cosh(2*K)^2 - sinh(2*K) *cos(x[1]) - sinh(2*K)*cos(x[2]))
end

for K in collect(0.1:0.1:2.0)
    println(K, " ", log(2) + hcubature(x->integrand(x, K), (0.0, 0.0), (2*pi, 2*pi), rtol=1e-6)[1]/(8*pi^2) )
end
