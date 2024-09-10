import os.path

import numpy as np
import pygrappa.cgsense
import scipy.io
from tqdm import tqdm

from pygrappa import grappa

# TODO: Change the range to the amount of slices (x-direction) (default = 40)
n_slices = 1
# TODO: Change the pc to the correct amount if periodicity activated (default = 18)
n_pc = 20
# TODO: Change the following parameters to the correct image size
size_all_slices = (n_slices, 352, 176, n_pc)
# TODO: Adapt Kernel-size (default = [7, 7])
grappa_kernel = [7, 7]

data_structure = np.zeros(size_all_slices, dtype='complex')

''' Read the Calibration Matrix from Matlab
calib.mat is a 4 dimensional matrix built in matlab
'''
calib = scipy.io.loadmat('calib.mat')
calib = calib.get('calib')

''' Read the undersampled k-space from Matlab
kspace_under.mat is a n dimensional matrix built in matlab
'''
kspace_under = scipy.io.loadmat('kspace_under.mat')
kspace_under = kspace_under.get('kspace_under')


''' generates the empty matrices with type complex '''
recon = np.zeros(kspace_under.shape, dtype='complex_')
recon_k = np.zeros(kspace_under.shape, dtype='complex_')
recon_real = np.zeros(kspace_under.shape, dtype='complex_')
recon_imag = np.zeros(kspace_under.shape, dtype='complex_')
ax = (0, 1)



''' check if phase-cycles and coils are combined in one dimension'''
if len(kspace_under.shape) > 3:

    ''' iterate through the phase-cycles '''
    for idx in tqdm(range(kspace_under.shape[2]), desc="Reconstructing...", colour='green'):
        ''' Reconstruction using GRAPPA '''
        recon_k[:, :, idx, :] = grappa(kspace_under[:, :, idx, :], calib[:, :, idx, :],
                                       kernel_size=grappa_kernel, coil_axis=-1, lamda=0.01, memmap=False)

        ''' Backtransformation to image space '''
        recon[:, :, idx, :] = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(recon_k[:, :, idx, :],
                                                                             axes=ax), axes=ax), axes=ax))
        recon_real[:, :, idx, :] = np.real(recon[:, :, idx, :]).astype('float64')
        recon_imag[:, :, idx, :] = np.imag(recon[:, :, idx, :]).astype('float64')

    ''' Combine Coils: squeeze(mean(img.*exp(-1i*angle(mean(img,2))),3)) '''
    exponent = np.exp(-1j * np.angle(np.mean(recon, 2)))
    new_size = ((kspace_under.shape[0]), (kspace_under.shape[1]), 1, (kspace_under.shape[3]))
    exponent = np.reshape(exponent, new_size)
    combined = np.squeeze(np.mean(recon * exponent, 3))
    data_structure[0, :, :, :] = combined

else:
    ''' if phase-cycles and coils are combined and treated equally '''
    recon_k = grappa(kspace_under, calib, kernel_size=grappa_kernel, coil_axis=-1, lamda=0.01, memmap=False)

    ''' Backtransformation to image space '''
    recon = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(recon_k, axes=ax), axes=ax), axes=ax))

    recon_real = np.real(recon).astype('float64')
    recon_imag = np.imag(recon).astype('float64')

'''Correct for shift'''
padded_struct = np.pad(data_structure, [(0, 0), (1, 1), (1, 1), (0, 0)], 'constant', constant_values=(complex(0)))
data_structure = padded_struct[:, 2:, 2:, :]


''' Write files for visualization use in Matlab '''
#scipy.io.savemat('real.mat', {'matrix_name': recon_real})
#scipy.io.savemat('imag.mat', {'matrix_name': recon_imag})
scipy.io.savemat('temp/combined.mat', {'matrix_name': data_structure})

