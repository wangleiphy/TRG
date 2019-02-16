import numpy as np
import scipy.linalg 
import torch, pdb

def safe_inverse(x, epsilon=1E-20):
    return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        U, S, V = torch.svd(A)
        #numpy_input = A.detach().numpy()
        #U, S, Vt = scipy.linalg.svd(numpy_input, full_matrices=False, lapack_driver='gesvd')
        #U = torch.as_tensor(U, dtype=A.dtype, device=A.device)
        #S = torch.as_tensor(S, dtype=A.dtype, device=A.device)
        #V = torch.as_tensor(np.transpose(Vt), dtype=A.dtype, device=A.device)
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

        Sinv = 1/S
        
        Sminus = (S - S[:, None])
        Sminus = safe_inverse(Sminus)

        Splus = (S + S[:, None])
        Splus.diagonal().fill_(np.inf)
        Splus = 1/Splus

        UdU = Ut @ dU
        VdV = Vt @ dV

        A = (UdU-UdU.t())
        B = (VdV-VdV.t())

        Su = Sminus*(A+B)/2 
        Sv = Splus*(A-B)/2 

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*Sinv) @ Vt 
        if (N>NS):
            dA = dA + (U*Sinv) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        #print (dU.norm().item(), dS.norm().item(), dV.norm().item())
        #print (Su.norm().item(), Sv.norm().item(), dS.norm().item())
        #print (dA1.norm().item(), dA2.norm().item(), dA3.norm().item())
        return dA

def test_svd():
    M, N = 50, 50
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVD.apply, input, eps=1e-6, atol=1e-4))
    print("Test Pass!")

if __name__=='__main__':
    test_svd()
