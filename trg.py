import torch

def TRG(K, Dcut, no_iter):
    D = 2
    inds = range(D)

    c = torch.sqrt(torch.cosh(K))
    s = torch.sqrt(torch.sinh(K))
    M = torch.stack([torch.cat([c, s]), torch.cat([c, -s])])

    T = torch.einsum('ai,aj,ak,al->ijkl', (M, M, M, M))
    
    lnZ = torch.zeros(1)
    for n in range(no_iter):
        
        #print(n, " ", T.max(), " ", T.min())
        maxval = T.max()
        T = T/maxval 
        lnZ += 2**(no_iter-n)*torch.log(maxval)

        D_new = min(D**2, Dcut)
        inds_new = range(D_new)

        Ma = T.permute(2, 1, 0, 3).contiguous().view(D**2, D**2)
        Mb = T.permute(3, 2, 1, 0).contiguous().view(D**2, D**2)

        S1 = torch.zeros(D, D, D_new)
        S2 = torch.zeros(D, D, D_new)
        S3 = torch.zeros(D, D, D_new)
        S4 = torch.zeros(D, D, D_new)

        U, S, V = torch.svd(Ma)
        for x in inds:
            for y in inds:
                for m in inds_new:
                    S1[x, y, m] = torch.sqrt(S[m]) * U[x+D*y, m]
                    S3[x, y, m] = torch.sqrt(S[m]) * V.t()[m, x+D*y]

        U, S, V = torch.svd(Mb)
        for x in inds: 
            for y in inds:
                for m in inds_new:
                    S2[x, y, m] = torch.sqrt(S[m]) * U[x+D*y, m]
                    S4[x, y, m] = torch.sqrt(S[m]) * V.t()[m, x+D*y]

        T_new = torch.einsum('war,abu,bgl,gwd->ruld', (S1, S2, S3, S4))

        D = D_new
        inds = inds_new 
        T = T_new

    trace = 0.0
    for i in inds:
        trace += T[i, i, i, i]
    lnZ += torch.log(trace)

    return lnZ 

if __name__=="__main__":
    import numpy as np 

    Dcut = 10
    n = 10

    for K in np.linspace(0, 2.0, 21):
        beta = torch.tensor([K]).requires_grad_()
        lnZ = TRG(beta, Dcut, n)
        print(torch.autograd.grad(lnZ, beta))
        print (K, lnZ.item()/2**n)
