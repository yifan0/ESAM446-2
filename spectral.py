
import numpy as np
import scipy.fft

class Basis:

    def __init__(self, N, interval):
        self.N = N
        self.interval = interval


class Fourier(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        return np.linspace(self.interval[0], self.interval[1], num=N_grid, endpoint=False)

    def transform_to_grid(self, data, axis, dtype, scale=1):
        if dtype == np.complex128:
            return self._transform_to_grid_complex(data, axis, scale)
        elif dtype == np.float64:
            return self._transform_to_grid_real(data, axis, scale)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def transform_to_coeff(self, data, axis, dtype):
        if dtype == np.complex128:
            return self._transform_to_coeff_complex(data, axis)
        elif dtype == np.float64:
            return self._transform_to_coeff_real(data, axis)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def _transform_to_grid_complex(self, data, axis, scale):
        # input is coefficient with hw index and hw norm
        coeff_hw_hw = data
        # scaling: put in zeros in place of extended coefficients
        coeff_hw_hw = np.concatenate((coeff_hw_hw[0:int(self.N/2)], np.zeros(int(self.N*(scale-1))),coeff_hw_hw[int(self.N/2):self.N]))
        # change to scipy index, hw norm
        # coeff_sp_hw = np.append(coeff_hw_hw[int(self.N/2):self.N],coeff_hw_hw[0:int(self.N/2)])
        coeff_sp_hw = coeff_hw_hw
        # scipy index, scipy norm
        coeff_sp_sp = coeff_sp_hw*(int(self.N*scale))
        # ifft
        grid = scipy.fft.ifft(coeff_sp_sp)
        return grid

    def _transform_to_coeff_complex(self, data, axis):
        # fft transform grid to scipy index, scipy norm
        coeff_sp_sp = scipy.fft.fft(data)
        N_scaled = len(coeff_sp_sp)
        # get rid of extended coefficients
        coeff_sp_sp = np.concatenate((coeff_sp_sp[0:int(self.N/2)], coeff_sp_sp[-int(self.N/2):]))
        # change to scipy index, hw norm
        coeff_sp_hw = coeff_sp_sp/N_scaled
        # change to hw index, hw norm
        # coeff_hw_hw = np.append(coeff_sp_hw[int(self.N/2):self.N],coeff_sp_hw[0:int(self.N/2)])
        coeff_hw_hw = coeff_sp_hw
        # Nyquist node = 0
        # coeff_hw_hw[int(self.N/2)] = 0
        return coeff_hw_hw

    def _transform_to_grid_real(self, data, axis, scale):
        # convert coefficients of real mode to that of complex mode
        coeff_real = data
        coeff_comp = np.zeros(self.N, dtype=np.complex128)
        coeff_comp[0] = coeff_real[0]
        for i in range(1,int(self.N/2)):
            coeff_comp[i] = coeff_real[2*i]/2 - coeff_real[2*i+1]/2j 
            coeff_comp[-i] = coeff_real[2*i]/2 + coeff_real[2*i+1]/2j
        # transform to grid as in complex Fourier mode case
        grid = self._transform_to_grid_complex(coeff_comp,axis,scale)
        return grid

    def _transform_to_coeff_real(self, data, axis):
        coeff_comp =  self._transform_to_coeff_complex(data, axis)
        # convert coefficients of complex mode to that of real one
        coeff_real = np.zeros(self.N, dtype=np.complex128)
        coeff_real[0] = coeff_comp[0]
        for i in range(2,self.N):
            if (i%2 == 0):
                index_c = int(i/2)
                coeff_real[i] = coeff_comp[index_c] + coeff_comp[-index_c]
            else:
                index_c = int(i/2)
                coeff_real[i] = (-coeff_comp[index_c] + coeff_comp[-index_c])*1j
        return coeff_real


class Field:

    def __init__(self, bases, dtype=np.float64):
        self.bases = bases
        self.dim = len(bases)
        self.dtype = dtype
        self.data = np.zeros([basis.N for basis in self.bases], dtype=dtype)
        self.coeff = np.array([True]*self.dim)

    def _remedy_scales(self, scales):
        if scales is None:
            scales = 1
        if not hasattr(scales, "__len__"):
            scales = [scales] * self.dim
        return scales

    def towards_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        axis = np.where(self.coeff == False)[0][0]
        self.data = self.bases[axis].transform_to_coeff(self.data, axis, self.dtype)
        self.coeff[axis] = True

    def require_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        else:
            self.towards_coeff_space()
            self.require_coeff_space()

    def towards_grid_space(self, scales=None):
        if not self.coeff.any():
            # already in full grid space
            return
        axis = np.where(self.coeff == True)[0][-1]
        scales = self._remedy_scales(scales)
        self.data = self.bases[axis].transform_to_grid(self.data, axis, self.dtype, scale=scales[axis])
        self.coeff[axis] = False

    def require_grid_space(self, scales=None):
        if not self.coeff.any(): 
            # already in full grid space
            return
        else:
            self.towards_grid_space(scales)
            self.require_grid_space(scales)
