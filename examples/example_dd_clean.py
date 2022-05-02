"""
Example script for running data domain clean
"""

import sys
import ehtim
from ehtim.imaging import clean # import *


def main():
    # Data domain clean with complex visibilities
    im = ehtim.image.load_txt('./models/avery_sgra_eofn.txt')
    arr = ehtim.array.load_txt('./arrays/EHT2017.txt') 
    #arr = ehtim.array.load_txt('./arrays/EHT2025.txt')
    obs = im.observe(arr, 1000, 600, 0, 24., 4.e10, add_th_noise=False, phasecal=True)
    prior = ehtim.image.make_square(obs, 128, 1.5*im.fovx())
    # data domain clean with visibilities
    outvis = clean.dd_clean_vis(obs, prior, niter=100, loop_gain=0.1, method='min_chisq',weighting='uniform')
    # Data domain clean directly with bispectrum
    # trial image 2 -- 2 Gaussians
    im2 = ehtim.image.make_square(obs, 256, 3*im.fovx())
    im2 = im2.add_gauss(1., (1*ehtim.RADPERUAS, 1*ehtim.RADPERUAS, 0, 0, 0))
    im2 = im2.add_gauss(.7, (1*ehtim.RADPERUAS, 1*ehtim.RADPERUAS, 0, -75*ehtim.RADPERUAS, 30*ehtim.RADPERUAS))
    obs2 = im2.observe(arr, 600, 600, 0, 24., 4.e9, add_th_noise=False, phasecal=False)
    prior = ehtim.image.make_square(obs, 50, 3*im.fovx())
    outbs = clean.dd_clean_amp_cphase(obs2, prior, niter=50, loop_gain=0.1, loop_gain_init=.01,phaseweight=2, weighting='uniform', bscount="min")


if __name__ == "__main__":
    main()

