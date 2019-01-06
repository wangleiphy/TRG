import numpy as np
import torch, pdb

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        U, S, V = torch.svd(A)
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)
        S2 = S**2
        Sinv = S/(S**2 + 1e-12)
        Finv = S2 - S2[:,None]
        F = Finv/(Finv**2 + 1e-12)

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = F * (UdU-UdU.t()) * S
        Sv = S[:,None] * (F*(VdV-VdV.t()))

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt +\
                (torch.eye(M, dtype=dU.dtype) - U@Ut) @ (dU*Sinv) @ Vt +\
                (U*Sinv) @ dV.t() @ (torch.eye(N, dtype=dU.dtype) - V@Vt)
        return dA

def test_svd():
    M, N = 100, 20
    torch.manual_seed(2)
    A = torch.randn(M, N, dtype=torch.float64)
    A.requires_grad=True
    assert(torch.autograd.gradcheck(SVD.apply, A, eps=1e-4, atol=1e-2))
    print("Test Pass!")

if __name__=='__main__':
    test_svd()
