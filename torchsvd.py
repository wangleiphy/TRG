import torch 

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        U, S, V = torch.svd(A)
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV, epsilon=1E-20):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)
        S2 = S**2
        Sinv = S/(S2 + epsilon)
        Finv = S2 - S2[:,None]
        F = Finv/(Finv**2 + epsilon)

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = F * (UdU-UdU.t()) * S
        Sv = S[:,None] * (F*(VdV-VdV.t()))

        dA1 = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        dA2 = (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*Sinv) @ Vt 
        dA3 = (U*Sinv) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        #print (dU.norm().item(), dS.norm().item(), dV.norm().item())
        #print (Su.norm().item(), Sv.norm().item(), dS.norm().item())
        #print (dA1.norm().item(), dA2.norm().item(), dA3.norm().item())
        return dA1 + dA2 + dA3

def test_svd():
    M, N = 50, 20
    torch.manual_seed(2)
    A = torch.randn(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVD.apply, A, eps=1e-6, atol=1e-4))
    print("Test Pass!")

if __name__=='__main__':
    test_svd()
