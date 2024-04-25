import torch
from torch import Tensor
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.expressions.variable import Variable
import numpy as np

from geotorch.product import ProductManifold
from geotorch.psd import PSD
from geotorch.skew import Skew
from geotorch.exceptions import (
    VectorError,
    NonSquareError,
    InManifoldError,
)
from geotorch.utils import _extra_repr


class Hurwitz(ProductManifold):
    def __init__(self, size, alpha: float = 1e-4, triv="expm"):
        r"""
        Manifold of Hurwitz matrices i.e. that have eigevalues lower than
        alpha > 0. These are stable matrices for continuous time dynamical systems.

        Args:
            size (torch.size): Size of the tensor to be parametrized
            alpha (float): bound on the greatest eigenvalue of the tensor.
                It has to be strictly positive
            triv (str or callable): Optional.
                A map that maps skew-symmetric matrices onto the orthogonal matrices
                surjectively. This is used to optimize the :math:`Q` in the eigenvalue
                decomposition. It can be one of ``["expm", "cayley"]`` or a custom
                callable. Default: ``"expm"``
        """
        n, tensorial_size = Hurwitz.parse_size(size)
        super().__init__(Hurwitz.manifolds(n, tensorial_size, triv))
        self.n = n
        self.tensorial_size = tensorial_size
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive found : {alpha}")
        self.alpha = alpha
        self.In = torch.eye(n)

    @classmethod
    def parse_size(cls, size):
        if len(size) < 2:
            raise VectorError(cls.__name__, size)
        n, k = size[-2:]
        tensorial_size = size[:-2]
        if n != k:
            raise NonSquareError(cls.__name__, size)
        return n, tensorial_size

    @staticmethod
    def manifolds(n, tensorial_size, triv):
        size_p = tensorial_size + (n, n)
        size_q = tensorial_size + (n, n)
        return PSD(size_q, triv=triv), PSD(size_p, triv=triv), Skew()

    def submersion(self, Q, P, S):
        return P @ (-0.5 * Q + S) - self.alpha * self.In

    def forward(self, X1, X2, X3):
        Q, P, S = super().forward([X1, X2, X3])
        return self.submersion(Q, P, S)

    def submersion_inv(self, A, check_in_manifold=True, epsilon=1e-8, solver="MOSEK"):
        if check_in_manifold and not self.in_manifold_eigen(A, epsilon):
            raise InManifoldError(A, self)
        with torch.no_grad():
            A = A.detach().numpy()
            nx = A.shape[0]
            P = Variable((nx, nx), "P", PSD=True)
            Q = (
                A.T @ P + P @ A + 2 * self.alpha * P  # type: ignore
            )  # solve the negative definite version
            constraints = [Q << -epsilon * np.eye(nx), P - (epsilon) * np.eye(nx) >> 0]  # type: ignore
            objective = Minimize(0)  # Feasibility problem

            prob = Problem(objective, constraints=constraints)
            prob.solve(solver)
            if prob.status not in ["infeasible", "unbounded"]:
                # Otherwise, problem.value is inf or -inf, respectively.
                print(f" P eigenvalues : {np.linalg.eig(P.value)[0]}\n")
            else:
                raise ValueError("SDP problem is infeasible or unbounded")

            # Now initialize
            P_inv = torch.inverse(Tensor(P.value))
            Q = Tensor(-Q.value)

            S = Tensor(P.value) @ Tensor(A) + 0.5 * Q + self.alpha * Tensor(P.value)
        return Q, P_inv, S

    def right_inverse(self, A, check_in_manifold=True):
        Q, P, S = self.submersion_inv(A, check_in_manifold)
        X1, X2, X3 = super().right_inverse([Q, P, S])
        return X1, X2, X3

    def in_manifold_eigen(self, A, eps=1e-6):
        r"""
        Check that all eigenvalues are lower than -alpha
        """
        if A.size()[:-2] != self.tensorial_size:  # check dimensions
            return False
        else:
            eig = torch.linalg.eigvals(A)
            reig = torch.real(eig)
            return (reig <= -self.alpha + eps).all().item()

    def sample(self, init_=torch.nn.init.xavier_normal_):
        with torch.no_grad():
            X_p = torch.empty(*(self.tensorial_size + (self.n, self.n)))
            init_(X_p)
            X_q = torch.empty_like(X_p)
            init_(X_q)
            X_s = torch.empty_like(X_p)
            init_(X_s)
            P = X_p @ X_p.transpose(-2, -1)
            Q = X_q @ X_q.transpose(-2, -1)
            S = X_s - X_s.transpose(-2, -1)

            return P @ (-0.5 * Q + S) - self.alpha * self.In

    def extra_repr(self) -> str:
        return _extra_repr(
            n=self.n,
            alpha=self.alpha,
            tensorial_size=self.tensorial_size,
        )
