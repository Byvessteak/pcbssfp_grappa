import numpy as np
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

''' Iterate trough all the slices '''
for img_slice in range(n_slices):
    print("Slice number:", (img_slice+1))
    name_calib = f'temp/calib_{img_slice + 1}.mat'
    name_kspace = f'temp/kspace_under_{img_slice + 1}.mat'

    ''' Read the Calibration Matrix from Matlab
    calib.mat is a 4 dimensional matrix built in matlab
    '''
    calib = scipy.io.loadmat(name_calib)
    calib = calib.get('calib')

    ''' Read the undersampled k-space from Matlab
    kspace_under.mat is a n dimensional matrix built in matlab
    '''
    kspace_under = scipy.io.loadmat(name_kspace)
    kspace_under = kspace_under.get('kspace_under')

    ''' generates the empty matrices with type complex '''
    recon = np.zeros(kspace_under.shape, dtype='complex_')
    recon_k = np.zeros(kspace_under.shape, dtype='complex_')
    ax = (0, 1)

    ''' iterate through the image plane '''
    for idx in tqdm(range(kspace_under.shape[1]), desc="Reconstructing..."):
        ''' Reconstruction using GRAPPA '''
        recon_k[:, idx, :, :] = grappa(kspace_under[:, idx, :, :], calib[:, idx, :, :],
                                       kernel_size=grappa_kernel, coil_axis=-1, lamda=0.01, memmap=False, silent=True)

    ''' iterate through the phase-cycles '''
    for idx in range(kspace_under.shape[2]):
        ''' Backtransformation from k-space to image-space '''
        recon[:, :, idx, :] = (np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(recon_k[:, :, idx, :],
                                                                             axes=ax), axes=ax), axes=ax))

    ''' Combine Coils: squeeze(mean(img.*exp(-1i*angle(mean(img,2))),3)) '''
    exponent = np.exp(-1j * np.angle(np.mean(recon, 2)))
    new_size = ((kspace_under.shape[0]), (kspace_under.shape[1]), 1, (kspace_under.shape[3]))
    exponent = np.reshape(exponent, new_size)
    combined = np.squeeze(np.mean(recon * exponent, 3))
    data_structure[img_slice, :, :, :] = combined

scipy.io.savemat('temp/combined.mat', {'matrix_name': data_structure})
