# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:38:01 2022

Author: Li, Jiangnan
        Kunming University of Science and Technology (KUST)

Email : li-jn12@tsinghua.org.cn

-------------------

MAG2305 : An FDM-FFT micromagnetic simulator
          Originated from MAG Group led by
          Prof. Dan Wei in Tsinghua University

Library version : numpy   1.25.0
                  pytorch 2.0.1

-------------------

"""


__version__ = 'UnetHd_Public_2024.10.17'
print('MAG2305 version: {:s}\n'.format(__version__))


import torch
import numpy as np
import sys


# =============================================================================
# Load Unet model
# =============================================================================
def load_model(m):
    global _ckpt_model
    _ckpt_model = m

def MFNN(spin):
    _ckpt_model.eval()
    with torch.no_grad():
        spin = spin.permute(2,3,0,1)
        spin = spin.view(1, -1, spin.size(2), spin.size(3))
        return _ckpt_model(spin).permute(2,3,0,1).view(spin.size(2), spin.size(3), -1, 3)


# =============================================================================
# Define constants here
# =============================================================================


class Const():
    " Define constants in this class "

    def __init__(self, value, unit):

        self.__value = value
        self.__unit  = unit

    @property
    def value(self):
        return self.__value

    @property
    def unit(self):
        return self.__unit

"""
gamma0: Gyromagnetic ratio of spin
        [Lande g factor] * [electron charge] / [electron mass] / [light speed]
"""
gamma0 = Const(1.75882e7, '[Oe s]^-1')


# =============================================================================
# Define general functions here
# =============================================================================


def get_randspin_2D(size=(1,1,1), split=1, rand_seed=0):
    """
    To get a random spin distribution in 2D view
    """
    size  = tuple(size)
    split = int(split)
    np.random.seed(rand_seed)         # initialize random seed

    spin_cases = []
    spin_cases.append([    1.0,    0.0, 0.0])   # +x
    spin_cases.append([   -1.0,    0.0, 0.0])   # -x
    spin_cases.append([    0.0,    1.0, 0.0])   # +y
    spin_cases.append([    0.0,   -1.0, 0.0])   # -y
    spin_cases.append([ 0.7071, 0.7071, 0.0])   # +x+y
    spin_cases.append([ 0.7071,-0.7071, 0.0])   # +x-y
    spin_cases.append([-0.7071, 0.7071, 0.0])   # -x+y
    spin_cases.append([-0.7071,-0.7071, 0.0])   # -x-y

    xsplit = size[0] / split          # x length of each split area
    ysplit = size[1] / split          # y length of each split area
    spin = np.empty( size + (3,) )

    for nx in range(split):
        for ny in range(split):

            xlow_bound  = int(nx * xsplit)
            xhigh_bound = int((nx+1) * xsplit) if nx + 1 < split \
                          else size[0]

            ylow_bound  = int(ny * ysplit)
            yhigh_bound = int((ny+1) * ysplit) if ny + 1 < split \
                          else size[1]

            spin[xlow_bound:xhigh_bound, ylow_bound:yhigh_bound, :] \
                        = spin_cases[np.random.randint(len(spin_cases))]

    return spin


def DemagCell(D, rv):
    """
    To get the demag matrix of a cell at a certain distance

    Arguments
    ---------
    D  : Float(3)
         Cell size : DX,DY,DZ
    rv : Float(3)
         Distance vector from cell center : RX,RY,RZ

    Returns
    -------
    DM : Float(3,3)
         Demag Matrix
    """
    D  = np.array(D)
    rv = np.array(rv)
    DM = np.zeros((3,3))

    pqw_range = [ [i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1] ]

    for pqw in pqw_range:
        pqw = np.array(pqw)
        R = 0.5*D + pqw*rv
        RR = np.linalg.norm(R)

        for i in range(3):
            j = (i+1)%3
            k = (i+2)%3
            DM[i,i] += np.arctan(R[j]*R[k]/R[i]/RR)
            DM[i,j] += 0.5*pqw[i]*pqw[j]*np.log((RR-R[k])/(RR+R[k]))
            DM[j,i] = DM[i,j]

    return DM/np.pi/4.


def numpy_roll(arr, shift, axis, pbc):
    """
    Re-defined numpy.roll(), including pbc judgement

    Arguments
    ---------
    arr      : Numpy Float(...)
               Array to be rolled
    shift    : Int
               Roll with how many steps
    axis     : Int
               Roll along which axis
    pbc      : Int or Bool
               Periodic condition for rolling; 1: pbc, 0: non-pbc

    Returns
    -------
    arr_roll : Numpy Float(...)
               arr after rolling
    """
    arr_roll = np.roll(arr, shift=shift, axis=axis)

    if not pbc:
        if axis == 0:
            if shift == 1:
                arr_roll[ 0,...] = arr[ 0,...]
            elif shift == -1:
                arr_roll[-1,...] = arr[-1,...]

        elif axis == 1:
            if shift == 1:
                arr_roll[:, 0,...] = arr[:, 0,...]
            elif shift == -1:
                arr_roll[:,-1,...] = arr[:,-1,...]

        elif axis == 2:
            if shift == 1:
                arr_roll[:,:, 0,...] = arr[:,:, 0,...]
            elif shift == -1:
                arr_roll[:,:,-1,...] = arr[:,:,-1,...]

    return arr_roll


def torch_roll(arr, shift, axis, pbc):
    """
    Re-defined torch.roll(), including pbc judgement

    Arguments
    ---------
    arr      : Torch Float(...)
               Array to be rolled
    shift    : Int
               Roll with how many steps
    axis     : Int
               Roll along which axis
    pbc      : Int
               Periodic condition for rolling; 1: pbc, 0: non-pbc

    Returns
    -------
    arr_roll : Torch Float(...)
               arr after rolling
    """
    arr_roll = torch.roll(arr, shifts=shift, dims=axis)

    if not pbc:
        if axis == 0:
            if shift == 1:
                arr_roll[ 0,...] = arr[ 0,...]
            elif shift == -1:
                arr_roll[-1,...] = arr[-1,...]

        elif axis == 1:
            if shift == 1:
                arr_roll[:, 0,...] = arr[:, 0,...]
            elif shift == -1:
                arr_roll[:,-1,...] = arr[:,-1,...]

        elif axis == 2:
            if shift == 1:
                arr_roll[:,:, 0,...] = arr[:,:, 0,...]
            elif shift == -1:
                arr_roll[:,:,-1,...] = arr[:,:,-1,...]

    return arr_roll


# =============================================================================
# Define mmModel here
# =============================================================================


class mmModel():
    " Define a micromagnetic model "


    # =============================================================================
    # PART I - Initialize mmModel
    # =============================================================================


    def __init__(self, types="3DPBC", cell=(1,1,1),
                       size=(1,1,1),  model=None, 
                       Ms=(1,),       Ax=(1.0e-6,), 
                       Ku=(0.0e0,),   Kvec=((0,0,1),), 
                       matters=None,  device='cuda' ):
        """
        Arguments
        ---------
        types : String
                "3DPBC" : Periodic along all X,Y,Z directions
                "film"  : Periodic along X,Y directions, Z in-plane
                "track" : Periodic along X direction, Y,Z finite, Z in-plane
                "bulk"  : Finite along all X,Y,Z directions
                # Default = "3DPBC"
        cell  : Float(3)
                Cell size: DX,DY,DZ [unit nm]
                # Default = (1,1,1)
                # Recorded as self.cell
        size  : Int(3)
                Model cell count along each direction: RNX,RNY,RNZ
                # Default = (1,1,1)
                # Recorded as self.size
        model : Int(size)
                Input data of model, defining matter id for each cell
                # Default = None
                # If [model] not None, input [size] will be ignored
                # Recorded as self.model
        Ms    : Float(Nmats)
                Saturation magnetization [unit emu/cc] for each matter
                # Default = 1
                # Recorded as self.Ms
        Ax    : Float(Nmats)
                Heisenberg exchange stiffness constant [unit erg/cm] for each matter
                # Default = 1.0e-6
                # Recorded as self.Ax
        Ku    : Float(Nmats)
                1st order uniaxial anisotropy energy density [unit erg/cc] for each matter
                # Default = 1
                # Recorded as self.Ku
        Kvec  : Float(Nmats,3)
                Easy axis for each matter
                # Default = (0,0,1)
                # Normalization will be performed on the input vectors
                # Recorded as self.Kvec
        matters : Float(Nmats,6)
                  Magnetic properties of matters
                  # Format : Ms[1], Ax[1], Ku[1], kx[1], ky[1], kz[1]
                             Ms[2], Ax[2], Ku[2], kx[2], ky[2], kz[2]
                             Ms[3], Ax[3], Ku[3], kx[3], ky[3], kz[3]
                             ...
                   [Units] : emu/cc, erg/cm, erg/cc, 1 , 1 , 1
                  # Default = None
                  # If [matters] not None, inputs [Ms], [Ax], [Ku] and [Kvec] will be ignored
        device  : "cuda" or "cuda:x" or "cpu"
                  Calculation device for updating spin state (when using torch)
                  # Recorded as self.device

        Parameters
        ----------
        self.Nmats: Int
                    Number of matters
        self.fftsize : Int(3)
                       fft model size FNX,FNY,FNZ
        self.Spin : Torch Float(size,3)
                    Spin direction of each cell
        self.He   : Torch Float(size,3)
                    Exchange field distribution
        self.Ha   : Torch Float(size,3)
                    Anisotropy field distribution
        self.Hd   : Torch Float(size,3)
                    Demag field distribution
        self.Heff : Torch Float(size,3)
                    Effective field distribution
        self.FDMW : Torch Complex(3,3,FNX,FNY,FNZ//2+1)
                    DFT of the demagnetization matrix of the whole model
        """
        print("\nInitialize an mmModel:\n")

        # Basic inputs
        self.types = types
        self.cell  = np.array(cell, dtype=float)

        if "cuda" in device:
            if torch.cuda.is_available():
                if device == "cuda":
                    print("  Cuda device available. Spin evolution using cuda.\n")
                    device = "cuda"
                else:
                    _, dev_index = device.split(":")
                    if dev_index == "":
                        print("  Cuda device available. Spin evolution using cuda.\n")
                        device = "cuda"
                    elif -1 < int(dev_index) < torch.cuda.device_count():
                        print("  Cuda:{0} available. Spin evolution using cuda:{0}.\n"
                              .format(int(dev_index)))
                        device = "cuda:" + dev_index
                    else:
                        print("  Cuda:{0} unavailable. Spin evolution using cpu instead.\n"
                              .format(int(dev_index)))
                        device = "cpu"
            else:
                print("  Cuda device unavailable. Spin evolution using cpu instead.\n")
                device = "cpu"
        else:
            print("  Spin evolution using cpu.\n")
            device = "cpu"
        self.device = torch.device(device)


        # model
        if model is None:
            self.size = np.array(size, dtype=int)
            self.model = np.ones(tuple(self.size), dtype=int)
            self.Nmats = 1
        else:
            self.model = np.array(model, dtype=int)
            self.size = np.array(model.shape)
            self.Nmats = self.model.max()
        self.Ncells = np.prod(self.model.shape)
        print("  # Cells  : {:d}".format(self.Ncells))
        print("  # Matters: {:d}".format(self.Nmats))


        # model size && fft size
        self.fftsize = np.array(self.size, dtype=int)
        if self.types == 'film':
            self.fftsize[-1] = 2*self.fftsize[-1]
        if self.types == 'track':
            self.fftsize[1]  = 2*self.fftsize[1]
            self.fftsize[-1] = 2*self.fftsize[-1]
        if self.types == 'bulk':
            self.fftsize = 2*self.fftsize
        self.pbc = (self.size==self.fftsize)


        # matters
        if matters is None:
            if type(Ms) == int or type(Ms) == float:
                self.Ms = np.full(self.Nmats, Ms, dtype=float)
            elif len(Ms) == self.Nmats:
                self.Ms = np.array(Ms, dtype=float)
            else:
                print("  [Input Error] Length of Ms does not match with Matters Number !")
                sys.exit(0)

            if type(Ax) == int or type(Ax) == float:
                self.Ax = np.full(self.Nmats, Ax, dtype=float)
            elif len(Ax) == self.Nmats:
                self.Ax = np.array(Ax, dtype=float)
            else:
                print("  [Input Error] Length of Ax does not match with Matters Number !")
                sys.exit(0)

            if type(Ku) == int or type(Ku) == float:
                self.Ku = np.full(self.Nmats, Ku, dtype=float)
            elif len(Ku) == self.Nmats:
                self.Ku = np.array(Ku, dtype=float)
            else:
                print("  [Input Error] Length of Ku does not match with Matters Number !")
                sys.exit(0)

            if len(Kvec) == 3 and (type(Kvec[0]) == int or type(Kvec[0]) == float):
                self.Kvec = np.full((self.Nmats,3), Kvec, dtype=float)
            elif len(Kvec) == self.Nmats and len(Kvec[0]) == 3:
                self.Kvec = np.array(Kvec, dtype=float)
            else:
                print("  [Input Error] Length of Kvec does not match with Matters Number !")
                sys.exit(0)

        else:
            if len(matters) >= self.Nmats:
                self.Ms = np.zeros(self.Nmats)
                self.Ax = np.zeros(self.Nmats)
                self.Ku = np.zeros(self.Nmats)
                self.Kvec = np.zeros((self.Nmats,3))
                matters = np.array(matters, dtype=float)
                for i in range(self.Nmats):
                    self.Ms[i], self.Ax[i], self.Ku[i], \
                    self.Kvec[i,0], self.Kvec[i,1], self.Kvec[i,2] = matters[i]

            else:
                print("  [Input Error] Lines of matters less than model matters !")
                sys.exit(0)

        for i in range(self.Nmats):
            Knorm = np.linalg.norm(self.Kvec[i])
            if Knorm == 0 and self.Ku[i] != 0:
                print("  [Input Error] Easy axis direction not assigned ! Matter {:d}".format(i+1))
                sys.exit(0)
            elif Knorm == 0 and self.Ku[i] == 0:
                pass
            else:
                self.Kvec[i] /= Knorm

        print("  Ms check : 1st {:8.3f}, last {:8.3f}   [emu/cc]"
              .format(self.Ms[0], self.Ms[-1]))
        print("  Ax check : 1st {:.2e}, last {:.2e}   [erg/cm]"
              .format(self.Ax[0], self.Ax[-1]))
        print("  Ku check : 1st {:.2e}, last {:.2e}   [erg/cc]\n"
              .format(self.Ku[0], self.Ku[-1]))


        # Magnetization, anisotropy, and exchange matrix
        self.MakeConstantMatrix()


        # Torch tensors; spin, fields, and demag matrix
        self.Spin = torch.zeros( tuple(self.size) + (3,), device=self.device)
        self.He   = torch.zeros( tuple(self.size) + (3,), device=self.device)
        self.Ha   = torch.zeros( tuple(self.size) + (3,), device=self.device)
        self.Hd   = torch.zeros( tuple(self.size) + (3,), device=self.device)
        self.Heff = torch.zeros( tuple(self.size) + (3,), device=self.device)

        self.FDMW = torch.zeros( (3,3) + ( self.fftsize[0], self.fftsize[1],
                                           self.fftsize[2]//2+1 ), 
                                 dtype=torch.complex64, device=self.device)


        # Field functions
        self.Demag      = self.DemagField_FFT
        self.Anisotropy = self.AnisotropyField_uniaxial
        self.Exchange   = self.ExchangeField_Heisenberg

        return None


    def MakeConstantMatrix(self):
        """
        To create constant matrix for further calculations
        { Called in self.__init__() }

        Parameters
        ----------
        self.Hk0  : Torch Float(self.size,3)
                    1st order uniaxial anisotropy field constant [unit Oe^1/2]
        self.Hx0  : Torch Float(6,self.size,3)
                    Heisenberg exchange field constant [unit Oe]
        self.Msmx : Torch Float(self.size)
                    Ms for each cell [unit emu/cc]
        """
        # Magnetization, anisotropy, and exchange matrix
        Msmx = np.zeros(tuple(self.size))
        Hk0 = np.zeros( tuple(self.size) + (3,) )
        for i in range(self.Nmats):
            Msmx[ self.model == i+1 ] = self.Ms[i]
            self.Energy_base = (self.model==i+1).sum() * self.Ku[i]
            Hk0[ self.model == i+1 ] = np.sqrt(2.0 * self.Ku[i] / self.Ms[i]) * self.Kvec[i] \
                                       if self.Ms[i] != 0.0 else 0.0

        Hx0 = np.zeros( (6,3) + tuple(self.size) )
        Ax = np.zeros_like(Msmx)
        for i in range(self.Nmats):
            Ax[ self.model == i+1 ] = self.Ax[i] if self.Ms[i] !=0.0 else 0.0

        Ax_nb = numpy_roll(Ax, shift= 1, axis=0, pbc=self.pbc[0])
        for l in range(3):
            np.divide( 4.0 * 1.0e14 * Ax * Ax_nb,
                       Msmx * (Ax + Ax_nb) * self.cell[0]**2, 
                       where= (Msmx!=0), out=Hx0[0,l] )

        Ax_nb = numpy_roll(Ax, shift=-1, axis=0, pbc=self.pbc[0])
        for l in range(3):
            np.divide( 4.0 * 1.0e14 * Ax * Ax_nb,
                       Msmx * (Ax + Ax_nb) * self.cell[0]**2, 
                       where= (Msmx!=0), out=Hx0[1,l] )

        Ax_nb = numpy_roll(Ax, shift= 1, axis=1, pbc=self.pbc[1])
        for l in range(3):
            np.divide( 4.0 * 1.0e14 * Ax * Ax_nb,
                       Msmx * (Ax + Ax_nb) * self.cell[1]**2, 
                       where= (Msmx!=0), out=Hx0[2,l] )

        Ax_nb = numpy_roll(Ax, shift=-1, axis=1, pbc=self.pbc[1])
        for l in range(3):
            np.divide( 4.0 * 1.0e14 * Ax * Ax_nb,
                       Msmx * (Ax + Ax_nb) * self.cell[1]**2, 
                       where= (Msmx!=0), out=Hx0[3,l] )

        Ax_nb = numpy_roll(Ax, shift= 1, axis=2, pbc=self.pbc[2])
        for l in range(3):
            np.divide( 4.0 * 1.0e14 * Ax * Ax_nb,
                       Msmx * (Ax + Ax_nb) * self.cell[2]**2, 
                       where= (Msmx!=0), out=Hx0[4,l] )

        Ax_nb = numpy_roll(Ax, shift=-1, axis=2, pbc=self.pbc[2])
        for l in range(3):
            np.divide( 4.0 * 1.0e14 * Ax * Ax_nb,
                       Msmx * (Ax + Ax_nb) * self.cell[2]**2, 
                       where= (Msmx!=0), out=Hx0[5,l] )

        # Simple statistic
        self.Msavg = Msmx.sum() / self.Ncells
        self.Hmax  = 2.0 * np.divide(self.Ku, self.Ms, where= self.Ms!=0 ).max() \
                   + 6.0 * Hx0.max() + 4*np.pi* self.Ms.max()

        np_warning = np.seterr()
        np.seterr(divide='ignore', invalid='ignore') # Ignore divided by 0
        np.seterr(**np_warning)  # Reset to default

        print("  Average magnetization  : {:9.2f}  [emu/cc]".format(self.Msavg))
        print("  Maximal anisotropy Hk  : {:9.3e}  [Oe]    ".format(Hk0.max()**2))
        print("  Maximal Heisenberg Hx  : {:9.3e}  [Oe]    ".format(Hx0.max()))
        print("  Maximal effective Heff : {:9.3e}  [Oe]\n  ".format(self.Hmax))

        # Torch tensors
        self.Msmx = torch.Tensor(Msmx).to(self.device)
        self.Hk0  = torch.Tensor(Hk0).to(self.device)
        self.Hx0  = torch.Tensor(Hx0.transpose(0,2,3,4,1)).to(self.device)

        return None


    def NormSpin(self):
        """
        Normalize self.Spin

        Parameters
        ----------
        self.Spin : Torch Float(self.size,3)
                    Spin direction of each cell
        """
        norm = torch.sqrt( torch.einsum( 'ijkl,ijkl -> ijk', 
                                         self.Spin, self.Spin ) )
        for l in range(3):
            self.Spin[...,l] /= norm

        self.Spin[~self.Spin.isfinite()] = 0.0

        return None


    def SpinInit(self, Spin_in):
        """
        Initialize Spin state from input

        Arguments
        ---------
        Spin_in   : Float(self.size,3)
                    Input Spin state

        Returns
        -------
        self.Spin.clone.cpu

        Parameters
        ----------
        self.Spin : Torch Float(self.size,3)
                    Spin direction of each cell
        """
        Spin_in = np.array(Spin_in, dtype=float)

        if Spin_in.shape != tuple(self.size) + (3,):
            print('[Input error] Spin size mismatched! Should be {}\n'
                  .format( tuple(self.size) + (3,) ) )
            sys.exit(0)

        else:
            self.Spin = torch.Tensor(Spin_in).to(self.device)
            print('Spin state initialized according to input.\n')

            if (self.model <= 0).sum() > 0:
                self.Spin[self.model<=0] = torch.zeros(size=(3,),
                                                       device=self.device)
            self.NormSpin()

            return self.Spin.clone().cpu()


    def DemagInit(self):
        """
        To get the demag matrix of the whole model
        Periodic boundary conditions applied !

        Parameters
        ----------
        FN  : Int(3)
              FN = self.fftsize, model fft size FNX,FNY,FNZ
        RN  : Int(3)
              RN = self.size, model size RNX,RNY,RNZ
        D   : Float(3)
              D = self.cell, cell size DX,DY,DZ
        DMW : Float(fftsize,3,3)
              Demag matrix of the whole model
              # DMW(0,0,0) , self-demag matrix
        self.FDMW : Torch Complex(3,3,FNX,FNY,FNZ//2+1)
                    DFT of DMW
        """
        FN = self.fftsize
        RN = self.size
        D  = self.cell
        DFN = D*FN
        DMW = np.zeros( tuple(FN) + (3,3) )

        # General demagmatrix for each cell
        rvm = np.empty(tuple(FN) + (3,))
        for ijk in np.ndindex(tuple(FN)):
            for l in range(3):
                rvm[ijk][l] = -1.*ijk[l]*D[l]
                rvm[ijk][l] += DFN[l] if ijk[l] > FN[l]//2 else 0.0

        pqw_range = [ [i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1] ]

        for pqw in pqw_range:
            pqw = np.array(pqw)
            R = 0.5*D + pqw*rvm
            RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )

            for i in range(3):
                j = (i+1)%3
                k = (i+2)%3
                DMW[...,i,i] += np.arctan(R[...,j]*R[...,k]/R[...,i]/RR)
                DMW[...,i,j] += 0.5*pqw[i]*pqw[j]*np.log((RR-R[...,k])/(RR+R[...,k]))
                DMW[...,j,i]  = DMW[...,i,j]

        # Demagmatrix for cells on the facets
        # surfaces x
        D1 = 1.0*D
        D1[0] = 0.5*D[0]
        pqw_bd = np.zeros(tuple(FN))
        for ijk in np.ndindex(tuple(FN)):
            pqw_bd[ijk] = 0
            if ijk[0] == FN[0]//2 and ijk[1] != FN[1]//2 and ijk[2] != FN[2]//2:
                pqw_bd[ijk] = 1

        for p in [+1, -1]:
            if p > 0:
                rvm1 = 1.0 * rvm
                rvm1[...,0] -= 0.5 * pqw_bd * D1[0]
            if p < 0:
                rvm1[...,0] += pqw_bd * DFN[0]

            for pqw in pqw_range:
                pqw = np.array(pqw)
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                for i in range(3):
                    j = (i+1)%3
                    k = (i+2)%3
                    DMW[...,i,i] -= p * np.arctan(R[...,j]*R[...,k]/R[...,i]/RR)
                    DMW[...,i,j] -= p * 0.5*pqw[i]*pqw[j]*np.log((RR-R[...,k])/(RR+R[...,k]))
                    DMW[...,j,i] -= p * 0.5*pqw[j]*pqw[i]*np.log((RR-R[...,k])/(RR+R[...,k]))

            for q,w in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
                pqw = np.array([abs(p),q,w])
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                DMW[...,0,0] += p * np.arctan(R[...,1]*R[...,2]/R[...,0]/RR)
                DMW[...,1,0] += p * 0.5*q*np.log((RR-R[...,2])/(RR+R[...,2]))
                DMW[...,2,0] += p * 0.5*w*np.log((RR-R[...,1])/(RR+R[...,1]))

        # surfaces y
        D1 = 1.0*D
        D1[1] = 0.5*D[1]
        pqw_bd = np.zeros(tuple(FN))
        for ijk in np.ndindex(tuple(FN)):
            pqw_bd[ijk] = 0
            if ijk[0] != FN[0]//2 and ijk[1] == FN[1]//2 and ijk[2] != FN[2]//2:
                pqw_bd[ijk] = 1

        for q in [+1, -1]:
            if q > 0:
                rvm1 = 1.0 * rvm
                rvm1[...,1] -= 0.5 * pqw_bd * D1[1]
            if q < 0:
                rvm1[...,1] += pqw_bd * DFN[1]

            for pqw in pqw_range:
                pqw = np.array(pqw)
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                for i in range(3):
                    j = (i+1)%3
                    k = (i+2)%3
                    DMW[...,i,i] -= q * np.arctan(R[...,j]*R[...,k]/R[...,i]/RR)
                    DMW[...,i,j] -= q * 0.5*pqw[i]*pqw[j]*np.log((RR-R[...,k])/(RR+R[...,k]))
                    DMW[...,j,i] -= q * 0.5*pqw[j]*pqw[i]*np.log((RR-R[...,k])/(RR+R[...,k]))

            for p,w in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
                pqw = np.array([p,abs(q),w])
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                DMW[...,1,1] += q * np.arctan(R[...,2]*R[...,0]/R[...,1]/RR)
                DMW[...,2,1] += q * 0.5*w*np.log((RR-R[...,0])/(RR+R[...,0]))
                DMW[...,0,1] += q * 0.5*p*np.log((RR-R[...,2])/(RR+R[...,2]))

        # surfaces z
        D1 = 1.0*D
        D1[2] = 0.5*D[2]
        pqw_bd = np.zeros(tuple(FN))
        for ijk in np.ndindex(tuple(FN)):
            pqw_bd[ijk] = 0
            if ijk[0] != FN[0]//2 and ijk[1] != FN[1]//2 and ijk[2] == FN[2]//2:
                pqw_bd[ijk] = 1

        for w in [+1, -1]:
            if w > 0:
                rvm1 = 1.0 * rvm
                rvm1[...,2] -= 0.5 * pqw_bd * D1[2]
            if w < 0:
                rvm1[...,2] += pqw_bd * DFN[2]

            for pqw in pqw_range:
                pqw = np.array(pqw)
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                for i in range(3):
                    j = (i+1)%3
                    k = (i+2)%3
                    DMW[...,i,i] -= w * np.arctan(R[...,j]*R[...,k]/R[...,i]/RR)
                    DMW[...,i,j] -= w * 0.5*pqw[i]*pqw[j]*np.log((RR-R[...,k])/(RR+R[...,k]))
                    DMW[...,j,i] -= w * 0.5*pqw[j]*pqw[i]*np.log((RR-R[...,k])/(RR+R[...,k]))

            for p,q in [ [1,1], [1,-1], [-1,1], [-1,-1] ]:
                pqw = np.array([p,q,abs(w)])
                R = 0.5*D1 + pqw*rvm1
                RR = np.sqrt( np.einsum( 'ijkl,ijkl -> ijk', R, R ) )
                DMW[...,2,2] += w * np.arctan(R[...,0]*R[...,1]/R[...,2]/RR)
                DMW[...,0,2] += w * 0.5*p*np.log((RR-R[...,1])/(RR+R[...,1]))
                DMW[...,1,2] += w * 0.5*q*np.log((RR-R[...,0])/(RR+R[...,0]))

        DMW /= 4*np.pi


        # Demag Matrix SumTest
        sum = np.zeros((3,3))
        for ijk in np.ndindex(tuple(RN)):
            sum += DMW[ijk]

        print("Demag Matrix SumTest:")
        for i in range(3):
            print("  [{:12.5e} {:12.5e} {:12.5e}]"
                  .format(sum[i,0], sum[i,1], sum[i,2]))
        print("  trace = {:.5f}".format(sum[0,0]+sum[1,1]+sum[2,2]))
        print("")

        # FFT of Demag Matrix
        for m in range(3):
            for n in range(3):
                self.FDMW[m,n] = torch.fft.rfftn( torch.Tensor(DMW[...,m,n]). 
                                                               to(self.device) )

        return None


    # =============================================================================
    # PART II - Calculate effective fields
    # =============================================================================


    def DemagField_FFT(self):
        """
        To get demagfield distribution of the whole model

        Returns
        -------
        None

        Parameters
        ----------
        FN  : Int(3)
              FN = self.fftsize, model fft size FNX,FNY,FNZ
        RN  : Int(3)
              RN = self.size, model size RNX,RNY,RNZ
        H   : Torch Float(3,FN)
              Demag field for whole fft model
        FH  : Torch Complex(3,FNX,FNY,FNZ//2+1)
              DFT of H
        FM  : Torch Complex(3,FNX,FNY,FNZ//2+1)
              DFT of spin for whole fft model
        self.Msmx : Torch Float(RN)
                    Ms for each cell [unit emu/cc]
        self.FDMW : Torch Complex(3,3,FNX,FNY,FNZ//2+1)
                    DFT of DMW
        self.Hd   : Torch Float(RN,3)
                    Demag field distribution
        """
        FN = self.fftsize
        RN = self.size

        shape_out = (FN[0], FN[1], FN[2]//2+1)

        M_tmp = torch.zeros( size= (3,) + tuple(FN), 
                             dtype=torch.float32, device=self.device )
        M_tmp[:, :RN[0], :RN[1], :RN[2]] = self.Spin.permute(3,0,1,2) * self.Msmx
        FM = torch.fft.rfftn( M_tmp, dim=(1,2,3) )

        FH = torch.zeros( size= (3,) + shape_out, 
                          dtype=torch.complex64, device=self.device )
        for m in range(3):
            for n in range(3):
                FH[m] -= self.FDMW[m,n] * FM[n]
        H = torch.fft.irfftn( FH, dim=(1,2,3) )

        self.Hd = H[:, :RN[0], :RN[1], :RN[2]].permute(1,2,3,0) * torch.pi*4.0

        return None


    def ExchangeField_Heisenberg(self):
        """
        To get Heisenberg exchange field distribution of the whole model

        Returns
        -------
        None

        Parameters
        ----------
        self.Hx0 : Torch Float(6,self.size,3)
                   Heisenberg exchange field constant [unit Oe]
        self.He  : Torch Float(self.size,3)
                   Exchange field distribution
        """
        self.He  = self.Hx0[0] * ( torch_roll( self.Spin, shift= 1, 
                                               axis=0, pbc=self.pbc[0] )
                                 - self.Spin )

        self.He += self.Hx0[1] * ( torch_roll( self.Spin, shift=-1, 
                                               axis=0, pbc=self.pbc[0] )
                                 - self.Spin )

        self.He += self.Hx0[2] * ( torch_roll( self.Spin, shift= 1, 
                                               axis=1, pbc=self.pbc[1] )
                                 - self.Spin )

        self.He += self.Hx0[3] * ( torch_roll( self.Spin, shift=-1, 
                                               axis=1, pbc=self.pbc[1] )
                                 - self.Spin )

        self.He += self.Hx0[4] * ( torch_roll( self.Spin, shift= 1, 
                                               axis=2, pbc=self.pbc[2] )
                                 - self.Spin )

        self.He += self.Hx0[5] * ( torch_roll( self.Spin, shift=-1, 
                                               axis=2, pbc=self.pbc[2] )
                                 - self.Spin )

        return None


    def AnisotropyField_uniaxial(self):
        """
        To get uniaxial anisotropy field distribution of the whole model

        Returns
        -------
        None

        Parameters
        ----------
        self.Hk0 : Torch Float(self.size,3)
                   1st order uniaxial anisotropy field constant [unit Oe^1/2]
        self.Ha  : Torch Float(self.size,3)
                   Anisotropy field distribution
        """
        SKdot = torch.einsum('ijkl,ijkl -> ijk', self.Spin, self.Hk0)
        for l in range(3):
            self.Ha[...,l] = self.Hk0[...,l] * SKdot

        return None


    def GetHeff_intrinsic(self):
        """
        To get effective field distribution of the whole model
        { Without external field, Hext }

        Returns
        -------
        None

        Parameters
        ----------
        self.Hd   : Torch Float(self.size,3)
                    Demag field distribution
        self.He   : Torch Float(self.size,3)
                    Exchange field distribution
        self.Ha   : Torch Float(self.size,3)
                    Anisotropy field distribution
        self.Heff : Torch Float(self.size,3)
                    Effective field distribution
        """
        self.Demag()

        self.Exchange()

        self.Anisotropy()

        self.Heff = self.Hd + self.He + self.Ha

        return None


    def GetHeff_woHd(self):
        """
        To get effective field distribution of the whole model
        { Without external field, Hext && demag field, Hd }

        Returns
        -------
        None

        Parameters
        ----------
        self.He   : Torch Float(self.size,3)
                    Exchange field distribution
        self.Ha   : Torch Float(self.size,3)
                    Anisotropy field distribution
        self.Heff : Torch Float(self.size,3)
                    Effective field distribution
        """
        self.Exchange()

        self.Anisotropy()

        self.Heff = self.He + self.Ha

        return None


    def GetHeff_unetHd(self):
        """
        To get effective field distribution; Hd from unet model

        Returns
        -------
        None

        Parameters
        ----------
        self.Heff : Torch Float(self.size,3)
                    Effective field distribution
        """
        self.Hd = MFNN( self.Spin ) * self.Ms[0] / 1000

        self.Exchange()

        self.Anisotropy()

        self.Heff = self.Hd + self.He + self.Ha

        return None


    def GetEnergy_fromHeff(self, Hext=(0.0, 0.0, 0.0)):
        """
        To calculate free energy from existing effective field (self.Heff)

        Returns
        -------
        float(self.Energy)

        Parameters
        ----------
        self.Energy : Torch Float
                      Magnetic free energy of MAG2305 model
                      # Energy = E0 - 1/2 * Ms * (Heff + 2*Hext)
        """
        Hext = torch.Tensor(Hext).to(self.device)

        self.Energy = -0.5 * (self.Msmx * torch.einsum('ijkl,ijkl -> ijk', 
                                                        self.Spin, 
                                                        self.Heff + 2*Hext)).sum()
        self.Energy += self.Energy_base

        return float(self.Energy)


    def GetEnergy_detailed(self, Hext=(0.0, 0.0, 0.0)):
        """
        To calculate free energy from each contribution

        Returns
        -------
        float(self.Energy)

        Parameters
        ----------
        self.Energy : Torch Float
                      Magnetic free energy of MAG2305 model
                      # Energy = Demag + Excha + Aniso + Exter
        """
        Hext = torch.Tensor(Hext).to(self.device)

        self.Energy_demag = -0.5 * (self.Msmx * torch.einsum('ijkl,ijkl -> ijk', 
                                                              self.Spin, self.Hd)).sum()

        self.Energy_excha = -0.5 * (self.Msmx * torch.einsum('ijkl,ijkl -> ijk', 
                                                              self.Spin, self.He)).sum()

        self.Energy_aniso = -0.5 * (self.Msmx * torch.einsum('ijkl,ijkl -> ijk', 
                                                              self.Spin, self.Ha)).sum() \
                            + self.Energy_base

        self.Energy_exter = -1.0 * (self.Msmx * torch.einsum('ijkl,l -> ijk', 
                                                              self.Spin, Hext)).sum()

        self.Energy = self.Energy_demag + self.Energy_excha \
                    + self.Energy_aniso + self.Energy_exter

        return float(self.Energy)


    # =============================================================================
    # PART III - Update spin state
    # =============================================================================


    def GetDSpin_LLG(self, Hext=(0.0, 0.0, 0.0), tau=1.0e-4, damping=0.05):
        """
        To get direct Spin change based on LLG equation

        Arguments
        ---------
        Hext   : Float(3)
                 External field
        tau    : Float
                 Pseudo time step for Spin update : dtime * gamma
        damping: Float
                 Damping constant

        Returns
        -------
        DSpin  : Torch Float(self.size,3)
                 Delta Spin; DSpin = tau * ( -ASpin + damping * GSpin )

        Parameters
        ----------
        ASpin  : Torch Float(self.size,3)
                 Spin rotation; ASpin = Spin X Heff
        GSpin  : Torch Float(self.size,3)
                 Spin damping; GSpin = ASpin X Spin
        """
        self.GetHeff()

        self.Heff += torch.Tensor(Hext).to(self.device)

        # Spin X Heff
        Aspin = torch.linalg.cross(self.Spin, self.Heff)

        # ASpin X Spin
        GSpin = torch.linalg.cross(Aspin, self.Spin)

        DSpin = tau * ( damping * GSpin - Aspin )

        return DSpin


    def GetDSpin_LLG_unet(self, Hext=(0.0, 0.0, 0.0), tau=1.0e-4, damping=0.05):
        """
        To get direct Spin change based on LLG equation

        Arguments
        ---------
        Hext   : Float(3)
                 External field
        tau    : Float
                 Pseudo time step for Spin update : dtime * gamma
        damping: Float
                 Damping constant

        Returns
        -------
        DSpin  : Torch Float(self.size,3)
                 Delta Spin; DSpin = tau * ( -ASpin + damping * GSpin )

        Parameters
        ----------
        ASpin  : Torch Float(self.size,3)
                 Spin rotation; ASpin = Spin X Heff
        GSpin  : Torch Float(self.size,3)
                 Spin damping; GSpin = ASpin X Spin
        """
        self.GetHeff_unet()

        self.Heff += torch.Tensor(Hext).to(self.device)

        # Spin X Heff
        Aspin = torch.linalg.cross(self.Spin, self.Heff)

        # ASpin X Spin
        GSpin = torch.linalg.cross(Aspin, self.Spin)

        DSpin = tau * ( damping * GSpin - Aspin)

        return DSpin


    def SpinLLG_RK4(self, Hext=(0.0, 0.0, 0.0), dtime=1.0e-13, damping=0.05, 
                          woHd=False):
        """
        To update Spin state based on LLG equation and RK4 method

        Arguments
        ---------
        Hext    : Float(3)
                  External field
        dtime   : Float
                  Time step for Spin update, unit [s]
        damping : Float
                  Damping constant

        Returns
        -------
        error   : Float
                  Maximal Spin change among all cells ( |DSpin|.max )

        Parameters
        ----------
        gamma   : Float
                  LLG gyromagnetic ratio; gamma = gamma0 / (1 + damping**2)
        tau     : Float
                  Pseudo time step for Spin update : dtime * gamma
        RSpin   : Torch Float(self.size,3)
                  Modified delta spin in RK4 algorithm: K1 + 2*K2 + 2*K3 + K4
        DSpin   : Torch Float(self.size,3)
                  Delta Spin; Final DSpin = RSpin / 6
        self.Spin : Torch Float(self.size,3)
                    Spin direction of each cell
        """
        gamma = gamma0.value / (1.0 + damping**2)
        tau = dtime * gamma

        # Include Hd or not (the latter for test only)
        if woHd:
            self.GetHeff = self.GetHeff_woHd
        else:
            self.GetHeff = self.GetHeff_intrinsic

        Spin0 = self.Spin

        # Get K1
        DSpin = self.GetDSpin_LLG(Hext=Hext, tau=tau, damping=damping)
        RSpin = DSpin

        # Get K2
        self.Spin = Spin0 + 0.5 * DSpin
        DSpin = self.GetDSpin_LLG(Hext=Hext, tau=tau, damping=damping)
        RSpin += 2.0 * DSpin

        # Get K3
        self.Spin = Spin0 + 0.5 * DSpin
        DSpin = self.GetDSpin_LLG(Hext=Hext, tau=tau, damping=damping)
        RSpin += 2.0 * DSpin

        # Get K4
        self.Spin = Spin0 + DSpin
        DSpin = self.GetDSpin_LLG(Hext=Hext, tau=tau, damping=damping)
        RSpin += DSpin

        # Get DSpin
        DSpin = RSpin / 6.0
        self.Spin = Spin0 + DSpin

        self.NormSpin()

        error = torch.sqrt( torch.einsum('ijkl,ijkl -> ijk', DSpin, DSpin).max() )

        return float(error)


    def SpinLLG_RK4_unetHd(self, Hext=(0.0, 0.0, 0.0), dtime=1.0e-13, damping=0.05):
        """
        To update Spin state based on LLG equation and RK4 method

        Arguments
        ---------
        Hext    : Float(3)
                  External field
        dtime   : Float
                  Time step for Spin update, unit [s]
        damping : Float
                  Damping constant

        Returns
        -------
        error   : Float
                  Maximal Spin change among all cells ( |DSpin|.max )
        """
        gamma = gamma0.value / (1.0 + damping**2)
        tau = dtime * gamma

        # Unet field type
        self.GetHeff_unet = self.GetHeff_unetHd

        Spin0 = self.Spin

        # Get K1
        DSpin = self.GetDSpin_LLG_unet(Hext=Hext, tau=tau, damping=damping)
        RSpin = DSpin

        # Get K2
        self.Spin = Spin0 + 0.5 * DSpin
        DSpin = self.GetDSpin_LLG_unet(Hext=Hext, tau=tau, damping=damping)
        RSpin += 2.0 * DSpin

        # Get K3
        self.Spin = Spin0 + 0.5 * DSpin
        DSpin = self.GetDSpin_LLG_unet(Hext=Hext, tau=tau, damping=damping)
        RSpin += 2.0 * DSpin

        # Get K4
        self.Spin = Spin0 + DSpin
        DSpin = self.GetDSpin_LLG_unet(Hext=Hext, tau=tau, damping=damping)
        RSpin += DSpin

        # Get DSpin
        DSpin = RSpin / 6.0
        self.Spin = Spin0 + DSpin

        self.NormSpin()

        error = torch.sqrt( torch.einsum('ijkl,ijkl -> ijk', DSpin, DSpin).max() )

        return float(error)


    def SpinDescent(self, Hext=(0.0, 0.0, 0.0), dtime=1.0e-6, woHd=False):
        """
        To update Spin state based on energy descent direction (Heff)

        Arguments
        ---------
        Hext  : Float(3)
                External field
        dtime : Float
                Pseudo time step for Spin update
                # default = 1.0e-6

        Returns
        -------
        error : Float
                Maximal Spin change among all cells ( |DSpin|.max )

        Parameters
        ----------
        self.Msmx  : Torch Float(RN)
                      Ms for each cell [unit emu/cc]
        GSpin      : Torch Float(self.size,3)
                     GSpin = Heff - Spin * (Spin .dot. Heff)
        DSpin      : Torch Float(self.size,3)
                     Delta Spin; DSpin = dtime * GSpin
        self.Spin  : Torch Float(self.size,3)
                     Spin direction of each cell
        """
        if woHd:
            self.GetHeff_woHd()
        else:
            self.GetHeff_intrinsic()

        self.Heff += torch.Tensor(Hext).to(self.device)

        GSpin = torch.empty( tuple(self.size) + (3,) ).to(self.device)
        SHdot = torch.einsum('ijkl,ijkl -> ijk', self.Spin, self.Heff)
        for l in range(3):
            GSpin[...,l] = self.Heff[...,l] * (self.Msmx != 0) \
                - self.Spin[...,l] * SHdot

        DSpin = dtime * GSpin

        error = torch.sqrt( torch.einsum('ijkl,ijkl -> ijk', DSpin, DSpin).max() )

        self.Spin += DSpin

        self.NormSpin()

        return float(error)


    def SpinDescent_unetHd(self, Hext=(0.0, 0.0, 0.0), dtime=1.0e-6):
        """
        To update Spin state based on energy descent direction (Heff)

        Arguments
        ---------
        Hext  : Float(3)
                External field
        dtime : Float
                Pseudo time step for Spin update
                # default = 1.0e-6

        Returns
        -------
        error : Float
                Maximal Spin change among all cells ( |DSpin|.max )

        Parameters
        ----------
        self.Msmx  : Torch Float(RN)
                     Ms for each cell [unit emu/cc]
        GSpin      : Torch Float(self.size,3)
                     GSpin = Heff - Spin * (Spin .dot. Heff)
        DSpin      : Torch Float(self.size,3)
                     Delta Spin; DSpin = dtime * GSpin
        self.Spin  : Torch Float(self.size,3)
                     Spin direction of each cell
        """
        # Unet field type
        self.GetHeff_unet = self.GetHeff_unetHd

        self.GetHeff_unet()

        self.Heff += torch.Tensor(Hext).to(self.device)

        GSpin = torch.empty( tuple(self.size) + (3,) ).to(self.device)
        SHdot = torch.einsum('ijkl,ijkl -> ijk', self.Spin, self.Heff)
        for l in range(3):
            GSpin[...,l] = self.Heff[...,l] * (self.Msmx != 0) \
                - self.Spin[...,l] * SHdot

        DSpin = dtime * GSpin

        error = torch.sqrt( torch.einsum('ijkl,ijkl -> ijk', DSpin, DSpin).max() )

        self.Spin += DSpin

        self.NormSpin()

        return float(error)


    # =============================================================================
    # PART IV - Integrated tasks
    # =============================================================================


    def GetStableState(self, Hext=(0.0, 0.0, 0.0), 
                             method="Descent", error_limit=1.0e-5, 
                             iters_max=100000, iters_check=1000, 
                             damping=0.1, dtime=None):
        """
        To get the stable state at a given external field

        Arguments
        ---------
        Hext       : Float(3)
                     External field
        method     : String
                     "Descent"       : Update Spin state based on energy descent
                     "Descent_woHd"  : Energy descent, but w/o Hd (for test only)
                     "Descent_unetHd": Update Spin state based on energy descent && unet-Hd
                     "LLG_RK4"       : Update Spin state based on LLG dynamics
                     "LLG_RK4_woHd"  : LLG RK4, but w/o Hd (for test only)
                     "LLG_RK4_unetHd": Update Spin state based on LLG dynamics && unet-Hd
                     # default = "Descent"
        error_limit: Float
                     Error value for determining convergence
                     # default = 1.0e-5
        iters_max  : Int
                     Maximal iteration steps
                     # default = 100000
        iters_check: Int
                     Steps for printing average magnetization
                     # default = 1000
        damping    : Float
                     Damping constant [For "LLG" method only]
                     # default = 0.1

        Returns
        -------
        self.Spin.clone.cpu

        Spin_sum   : Float(3)
                     Average magnetization
        error      : Float
                     Maximal Spin change when terminated
        steps      : Int
                     Iteration steps when terminated

        Parameters
        ----------
        Hax        : Float
                     Upper bound of magnetic field
                     # Hmax = Heff_upbound + Hext.max
        dtime      : Float
                     Pseudo time step for Spin update
                     # Auto-estimated; ~ Hmax^-1
        self.Msmx  : Torch Float(RN)
                     Ms for each cell [unit emu/cc]
        self.Spin  : Torch Float(self.size,3)
                     Spin direction of each cell
        """
        Hext = np.array(Hext, dtype=float)
        iters_max = int(iters_max)
        iters_check = int(iters_check)

        if method == "Descent":
            if dtime is None:
                Hmax = self.Hmax + abs(Hext).max()
                exp = int(np.log10(1.0/Hmax)) - 2
                dtime = int((1.0/Hmax) / 10**exp) * 10**exp
            Update = self.SpinDescent
            Args = Hext, dtime, False

            print('  Calculation method : Energy Descent')
            print('  Psuedo time step   : {:.3e}\n'.format(dtime))

        elif method == "Descent_woHd":
            if dtime is None:
                Hmax = self.Hmax + abs(Hext).max()
                exp = int(np.log10(1.0/Hmax)) - 2
                dtime = int((1.0/Hmax) / 10**exp) * 10**exp
            Update = self.SpinDescent
            Args = Hext, dtime, True

            print('  Calculation method : Energy Descent - w/o Hd')
            print('  Psuedo time step   : {:.3e}\n'.format(dtime))

        elif method == "Descent_unetHd":
            if dtime is None:
                Hmax = self.Hmax + abs(Hext).max()
                exp = int(np.log10(1.0/Hmax)) - 2
                dtime = int((1.0/Hmax) / 10**exp) * 10**exp
            Update = self.SpinDescent_unetHd
            Args = Hext, dtime

            print('  Calculation method : Energy Descent - Unet Hd')
            print('  Psuedo time step   : {:.3e}\n'.format(dtime))

        elif method == "LLG_RK4":
            if dtime is None:
                Hmax = self.Hmax + abs(Hext).max()
                exp = int(np.log10(1.0/Hmax)) - 2
                dtime = int((1.0/Hmax) / 10**exp) * 10**(exp-7)
            Update = self.SpinLLG_RK4
            Args = Hext, dtime, damping, False

            print('  Calculation method : LLG - RK4')
            print('  Time step  : {:.3e}  '.format(dtime))
            print('  Damping    : {:.3f}\n'.format(damping))

        elif method == "LLG_RK4_woHd":
            if dtime is None:
                Hmax = self.Hmax + abs(Hext).max()
                exp = int(np.log10(1.0/Hmax)) - 2
                dtime = int((1.0/Hmax) / 10**exp) * 10**(exp-7)
            Update = self.SpinLLG_RK4
            Args = Hext, dtime, damping, True

            print('  Calculation method : LLG - RK4 - w/o Hd')
            print('  Time step  : {:.3e}  '.format(dtime))
            print('  Damping    : {:.3f}\n'.format(damping))

        elif method == "LLG_RK4_unetHd":
            if dtime is None:
                Hmax = self.Hmax + abs(Hext).max()
                exp = int(np.log10(1.0/Hmax)) - 2
                dtime = int((1.0/Hmax) / 10**exp) * 10**(exp-7)
            Update = self.SpinLLG_RK4_unetHd
            Args = Hext, dtime, damping

            print('  Calculation method : LLG - RK4 - Unet Hd')
            print('  Time step  : {:.3e}  '.format(dtime))
            print('  Damping    : {:.3f}\n'.format(damping))

        else:
            print('  [Input Error] Unknown method!')
            sys.exit(0)

        for iters in range(iters_max):
            error = Update(*Args)

            if iters % iters_check == 0 or \
                iters == iters_max-1 or error <= error_limit:

                Spin_sum = np.empty(3)
                for l in range(3):
                    Spin_sum[l] = (self.Spin[...,l] * self.Msmx).sum().cpu()
                Spin_sum = Spin_sum / self.Ncells / self.Msavg
                print('  Hext={},  steps={:d}'.format(list(Hext), iters+1))
                print('  M_avg={},  error={:.8f}\n'.format(Spin_sum, error))

                if error <= error_limit:
                    break

        return self.Spin.clone().cpu(), Spin_sum, error, iters+1


    def GetMHLoop(self, Hext_range=(0.0,), Hext_vec=(0.0,0.0,1.0), 
                        method="Descent",  error_limit=1.0e-5, 
                        iters_max=100000,  iters_check=1000, 
                        save_MH=False,     save_spin=None):
        """
        To get the M-H curve under given external field range

        Arguments
        ---------
        Hext_range : Float(:)
                     External field values
                     # default = (0, )
        Hext_vec   : Float(3)
                     External field direction
                     # default = (0, 0, 1)
        method     : String
                     "Descent"       : Update Spin state based on energy descent
                     "Descent_woHd"  : Energy descent, but w/o Hd (for test only)
                     "Descent_unetHd": Update Spin state based on energy descent && unet-Hd
                     "LLG_RK4"       : Update Spin state based on LLG dynamics
                     "LLG_RK4_woHd"  : LLG RK4, but w/o Hd (for test only)
                     "LLG_RK4_unetHd": Update Spin state based on LLG dynamics && unet-Hd
                     # default = "Descent"
        error_limit: Float
                     Error value for determining convergence
                     # default = 1.0e-5
        iters_max  : Int
                     Maximal iteration steps at each external field
                     # default = 100000
        iters_check: Int
                     Steps for printing average magnetization
                     # default = 1000
        save_MH    : Bool
                     Save the M-H data or not; file name: "outMHloop.txt"
                     # default = False
        save_spin  : String or Bool
                     Save the spin state [=String or True] or not [None or False]
                     "txt" : txt format,         file name: "spinxyz_xxxxx.txt"
                     "vtk" : vtk format,         file name: "spinxyz_xxxxx.vtk"
                     True or "vtkb": vtk binary format,
                                                 file name: "spinxyz_xxxxx.vtk"
                     "numpy" : numpy array ,     file name: "spinxyz_xxxxx.npy"
                     "torch" : torch Tensor,     file name: "spinxyz_xxxxx.pt"
                     # default = None

        Returns
        -------
        MHhead  : String(6)
                  ["Hext", "Mext", "mext", "mx", "my", "mz"]
        MHdata  : Float(6,:)
                  Format:  Hext0, Hext1, Hext2, Hext3, ...
                           Mext0, Mext1, Mext2, Mext3, ...
                           mext0, mext1, mext2, mext3, ...
                           mx0,   mx1,   mx2,   mx3,   ...
                           my0,   my1,   my2,   my3,   ...
                           mz0,   mz1,   mz2,   mz3,   ...
        """
        print("\nBegin M-H loop calculation:\n")

        Hext_range = np.array(Hext_range, dtype=float)
        Hext_vec = np.array(Hext_vec, dtype=float)
        Hext_vec /= np.linalg.norm(Hext_vec)

        if save_MH :
            MH_file = "outMHloop.txt"
            try:
                mf = open(file=MH_file, mode='r', encoding='utf-8')
                mf.close()
                mf = open(file=MH_file, mode='a', encoding='utf-8')
                mf.write('\n')
            except:
                mf = open(file=MH_file, mode='w', encoding='utf-8')
                mf.write("%-10s %12s %12s %12s %12s %12s %12s %12s %12s %23s\n"
                    % ('Records', 'Hext[Oe]', 'Mext[emu/cc]', 'mext', 
                       'mx', 'my', 'mz', 'Error', 'Steps', 'Localtime'))
                mf.flush()

        MHdata = np.array([[],[],[],[],[],[]], dtype=float)
        for n, Hext_val in enumerate(Hext_range):
            Hext = Hext_val * Hext_vec
            spin, spin_sum, error, steps = self.GetStableState(
                                                Hext=Hext, method=method, 
                                                error_limit=error_limit, 
                                                iters_max=iters_max, 
                                                iters_check=iters_check)

            mext = np.dot(spin_sum, Hext_vec)
            MHdata = np.append(MHdata, [[Hext_val], [mext * self.Msavg], 
                                        [mext], [spin_sum[0]], [spin_sum[1]], 
                                        [spin_sum[2]]],  axis=1)

            if save_MH :
                mf.write("%-10d %12.3f %12.3f %12f %12f %12f %12f %12.3e %12d %23s\n" 
                    % (n, Hext_val, mext * self.Msavg, mext, 
                       spin_sum[0], spin_sum[1], spin_sum[2], 
                       error, steps, 
                       time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) ))
                mf.flush()

            if save_spin is not None:
                file='spinxyz_' + str(n).zfill(5)
                if type(save_spin) == bool:
                    if save_spin :
                        self.OutputSpin(file=file, form='vtkb', spin=spin)
                elif type(save_spin) == str:
                    self.OutputSpin(file=file, form=save_spin, spin=spin)
                else :
                    print('[Warning] Unknown save_spin type! No spin files to be recorded!')

        if save_MH :
            mf.close()

        return ["Hext", "Mext", "mext", "mx", "my", "mz"], MHdata


    # =============================================================================
    # PART V - Output data
    # =============================================================================


    def OutputSpin(self, file='spinxyz_test', form='txt', spin=None):
        """
        To write down the spin state in a file

        Arguments
        ---------
        file : String
               File name
        form : String
               "txt"  : txt format
               "vtk"  : vtk format
               "vtkb" : vtk binary format
               "numpy": numpy array
               "torch": torch Tensor
        spin : Float(self.size,3)
               Spin data to be written down
        """
        if form == 'txt':

            file = file + '.txt'
            sf = open(file=file, mode='w', encoding='utf-8')
            for ijk in np.ndindex(spin.shape[:-1]):
                sf.write("%10d %10d %10d %15.8f %15.8f %15.8f %10d\n" 
                    % (ijk[0], ijk[1], ijk[2], 
                       spin[ijk][0], spin[ijk][1], spin[ijk][2], self.model[ijk]))
            sf.close()

        elif form == 'vtk':
            file = file + '.vtk'
            sf = open(file=file, mode='w', encoding='utf-8')

            sf.write("# vtk DataFile Version 2.0\n")
            sf.write("Magnetization\n")
            sf.write("ASCII\n")
            sf.write("DATASET STRUCTURED_POINTS\n")
            sf.write("DIMENSIONS %5d %5d %5d\n" % ( spin.shape[0], 
                                                   spin.shape[1], 
                                                   spin.shape[2]) )
            sf.write("ASPECT_RATIO %5.2f %5.2f %5.2f\n" % ( self.cell[0]/self.cell[0], 
                                                           self.cell[1]/self.cell[0], 
                                                           self.cell[2]/self.cell[0]) )
            sf.write("ORIGIN  0 0 0\n")
            sf.write("POINT_DATA %12d\n" % ( self.size.prod() ))
            sf.write("VECTORS spin double\n")

            spin = spin.transpose(0,2)
            for ijk in np.ndindex(spin.shape[:-1]):
                sf.write("%15.8f %15.8f %15.8f\n" 
                    % ( spin[ijk][0], spin[ijk][1], spin[ijk][2]) )

            sf.write("\nSCALARS matter float 1\n")
            sf.write("LOOKUP_TABLE default\n")

            model = self.model.transpose(2,1,0)
            for ijk in np.ndindex(model.shape):
                nn = 1 + ijk[2] + ijk[1]*self.size[0] + ijk[0]*self.size[0]*self.size[1]
                if nn%6 == 0:
                    sf.write("%6d\n" % ( model[ijk] ))
                else :
                    sf.write("%6d "  % ( model[ijk] ))

            sf.close()

        elif form == 'vtkb':
            file = file + '.vtk'
            sf = open(file=file, mode='w', encoding='utf-8')

            sf.write("# vtk DataFile Version 2.0\n")
            sf.write("Magnetization\n")
            sf.write("BINARY\n")
            sf.write("DATASET STRUCTURED_POINTS\n")
            sf.write("DIMENSIONS %5d %5d %5d\n" % ( spin.shape[0], 
                                                   spin.shape[1], 
                                                   spin.shape[2]) )
            sf.write("ASPECT_RATIO %5.2f %5.2f %5.2f\n" % ( self.cell[0]/self.cell[0], 
                                                           self.cell[1]/self.cell[0], 
                                                           self.cell[2]/self.cell[0]) )
            sf.write("ORIGIN  0 0 0\n")
            sf.write("POINT_DATA %12d\n" % ( self.size.prod() ))
            sf.write("VECTORS spin double\n")
            sf.close()

            sf = open(file=file, mode='ab')
            spin = spin.transpose(0,2)
            for ijk in np.ndindex( spin.shape[:-1] ):
                sf.write( struct.pack('>d', spin[ijk][0]) )
                sf.write( struct.pack('>d', spin[ijk][1]) )
                sf.write( struct.pack('>d', spin[ijk][2]) )
            sf.close()

            sf = open(file=file, mode='a', encoding='utf-8')
            sf.write("\nSCALARS matter int 1\n")
            sf.write("LOOKUP_TABLE default\n")
            sf.close()

            sf = open(file=file, mode='ab')
            model = self.model.transpose(2,1,0)
            for ijk in np.ndindex( model.shape ):
                sf.write( struct.pack('>i', model[ijk]) )
            sf.close()

        elif form == 'numpy':
            np.save(file, spin.numpy())

        elif form == 'torch':
            np.save(spin, file + '.pt')

        else :
            print('[Warning] Unknown OutputSpin format! No spin files to be recorded!')

            sf.close()

        return None
