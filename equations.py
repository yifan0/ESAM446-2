
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:

    def __init__(self, u):
        dtype = u.dtype
        self.u = u
        x_basis = u.bases[0]
        # print(x_basis)
        # print(x_basis.wavenumbers(dtype))
        self.dudx = spectral.Field([x_basis],dtype=dtype)
        self.RHS = spectral.Field([x_basis],dtype=dtype)
        self.problem = spectral.InitialValueProblem([self.u],[self.RHS])
        p = self.problem.subproblems[0]

        self.N = x_basis.N
        self.kx = x_basis.wavenumbers(dtype)
        p.M = sparse.eye(self.N)
        if dtype == np.complex128:
            p.L = sparse.diags(-1j*self.kx**3)
        else:
            upper_diag = np.zeros(self.N-1)
            upper_diag[::2] = x_basis.wavenumbers(dtype)[::2]**3
            lower_diag = - upper_diag
            p.L = sparse.diags([upper_diag, lower_diag], offsets=[1, -1])
            # print(p.L.A[:6,:6])
        pass

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        x_basis = u.bases[0]
        dtype = u.dtype
        dudx = self.dudx
        RHS = self.RHS
        kx = self.kx
        # print(x_basis.wavenumbers(dtype))
        for i in range(num_steps):
            u.require_coeff_space()
            dudx.require_coeff_space()
            if dtype == np.complex128:
                dudx.data = 1j*kx*u.data
            else:
                upper_diag = np.zeros(self.N-1)
                upper_diag[::2] = -x_basis.wavenumbers(dtype)[::2]
                lower_diag = - upper_diag
                D = sparse.diags([upper_diag, lower_diag], offsets=[1, -1])
                # print(D.A[:6,:6])
                # break
                dudx.data = D@u.data
            u.require_grid_space(scales=3/2)
            dudx.require_grid_space(scales=3/2)
            RHS.require_grid_space(scales=3/2)
            RHS.data = 6 * u.data * dudx.data

            ts.step(dt)
        pass

class SHEquation:

    def __init__(self, u):
        dtype = u.dtype
        self.u = u
        x_basis = u.bases[0]
        r = -0.3

        self.dudx = spectral.Field([x_basis],dtype=dtype)
        self.RHS = spectral.Field([x_basis],dtype=dtype)
        self.problem = spectral.InitialValueProblem([self.u],[self.RHS])
        p = self.problem.subproblems[0]

        self.N = x_basis.N
        self.kx = x_basis.wavenumbers(dtype)
        p.M = sparse.eye(self.N)
        if dtype == np.complex128:
            diag = (1-self.kx**2)**2 - r
            p.L = sparse.diags(diag)
        else:
            p.L = (sparse.eye(self.N)-sparse.diags(self.kx**2))@(sparse.eye(self.N)-sparse.diags(self.kx**2)) - sparse.diags(np.repeat(r,self.N))
        pass

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        x_basis = u.bases[0]
        dtype = u.dtype
        dudx = self.dudx
        RHS = self.RHS
        kx = self.kx
        # print(x_basis.wavenumbers(dtype))
        for i in range(num_steps):
            u.require_coeff_space()
            u.require_grid_space(scales=2)
            RHS.require_grid_space(scales=2)
            RHS.data = u.data * u.data * (1.8 - u.data)

            ts.step(dt)
        pass


