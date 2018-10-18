import torch

def TRG(K, Dcut, no_iter, device='cpu', epsilon=1E-6):
    D = 2

    c = torch.sqrt(torch.cosh(K))
    s = torch.sqrt(torch.sinh(K))
    M = torch.stack([torch.cat([c, s]), torch.cat([c, -s])])

    T = torch.einsum('ai,aj,ak,al->ijkl', (M, M, M, M))
    
    lnZ = 0.0
    for n in range(no_iter):
        
        #print(n, " ", T.max(), " ", T.min())
        maxval = T.max()
        T = T/maxval 
        lnZ += 2**(no_iter-n)*torch.log(maxval)

        Ma = T.permute(2, 1, 0, 3).contiguous().view(D**2, D**2)
        Mb = T.permute(3, 2, 1, 0).contiguous().view(D**2, D**2)

        Ua, Sa, Va = torch.svd(Ma)
        Ub, Sb, Vb = torch.svd(Mb)

        D_new = min(min(D**2, Dcut), min((Sa>epsilon).sum().item(), (Sb>epsilon).sum().item()))

        S1 = (Ua[:, :D_new]* torch.sqrt(Sa[:D_new])).view(D, D, D_new)
        S3 = (Va[:, :D_new]* torch.sqrt(Sa[:D_new])).view(D, D, D_new)
        S2 = (Ub[:, :D_new]* torch.sqrt(Sb[:D_new])).view(D, D, D_new)
        S4 = (Vb[:, :D_new]* torch.sqrt(Sb[:D_new])).view(D, D, D_new)

        T_new = torch.einsum('war,abu,bgl,gwd->ruld', (S1, S2, S3, S4))

        D = D_new
        T = T_new

    trace = 0.0
    for i in range(D):
        trace += T[i, i, i, i]
    lnZ += torch.log(trace)

    return lnZ 

if __name__=="__main__":
    import numpy as np 
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-cuda", type=int, default=-1, help="use GPU")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

    Dcut = 20
    n = 20

    for K in np.linspace(0, 2.0, 21):
        beta = torch.tensor([K], device=device).requires_grad_()
        lnZ = TRG(beta, Dcut, n, device=device)
        lnZ.backward()
        print (K, lnZ.item()/2**n, beta.grad.item())
