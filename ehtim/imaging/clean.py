""""
Clean-like imagers

Copyright (C) 2022 Andrew Chael

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import string
import time
import numpy
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial
import image
sys.path.extend(["../", "../observing", "../imaging"])
import const_def
from observing import obs_helpers
from imaging import imager_utils

try:
    from pynfft.nfft import NFFT
except ImportError:
    print("Warning: No NFFT installed! Deprecated!!!")


# Constants & Definitions
NHIST = 50 # number of steps to store for hessian approx
MAXIT = 100.

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp']
REGULARIZERS = ['gs', 'tv', 'tv2','l1', 'patch', 'simple', 'compact', 'compact2']

NFFT_KERSIZE_DEFAULT = 20
GRIDDER_P_RAD_DEFAULT = 2
GRIDDER_CONV_FUNC_DEFAULT = 'gaussian'
FFT_PAD_DEFAULT = 2
FFT_INTERP_DEFAULT = 3

nit = 0 # global variable to track the iteration number in the plotting callback

# Imagers
def plot_i(Image, nit, chi2, fig=1, cmap='afmhot'):
    """Plot the total intensity image at each iteration
    """

    plt.ion()
    plt.figure(fig)
    plt.pause(0.00001)
    plt.clf()

    plt.imshow(Image.imvec.reshape(Image.ydim,Image.xdim), cmap=plt.get_cmap(cmap), interpolation='gaussian')
    xticks = obs_helpers.ticks(Image.xdim, Image.psize/const_def.RADPERAS/1e-6)
    yticks = obs_helpers.ticks(Image.ydim, Image.psize/const_def.RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title("step: %i  $\chi^2$: %f " % (nit, chi2), fontsize=20)

def dd_clean_vis(Obsdata, InitIm, niter=1, clipfloor=-1, ttype="direct", loop_gain=1, method='min_chisq', weighting='uniform',
                 fft_pad_factor=FFT_PAD_DEFAULT, p_rad=NFFT_KERSIZE_DEFAULT, show_updates=False):

    # limit imager range to prior values > clipfloor
    embed_mask = InitIm.imvec >= clipfloor

    # get data
    vis = Obsdata.data['vis']
    sigma = Obsdata.data['sigma']
    uv = numpy.hstack((Obsdata.data['u'].reshape(-1,1), Obsdata.data['v'].reshape(-1,1)))

    # necessary nfft infos
    npad = int(fft_pad_factor * numpy.max((InitIm.xdim, InitIm.ydim)))
    nfft_info = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv)
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    # Weights
    weights_nat = 1./(sigma**2)

    if weighting=='uniform':
        weights = numpy.ones(len(weights_nat))
    elif weighting=='natural':
        weights=weights_nat
    else:
        raise Exception("weighting must be 'uniform' or 'natural'")
    weights_norm = numpy.sum(weights)

    # Coordinate matrix
    coord = InitIm.psize * numpy.array([[[x,y] for x in numpy.arange(InitIm.xdim//2,-InitIm.xdim//2,-1)]
                                            for y in numpy.arange(InitIm.ydim//2,-InitIm.ydim//2,-1)])
    coord = coord.reshape(InitIm.ydim*InitIm.xdim, 2)
    coord = coord[embed_mask]

    # Initial imvec and visibilities
    # TODO currently always initialized to zero!!
    OutputIm = InitIm.copy()
    DeltasIm = InitIm.copy()
    ChisqIm = InitIm.copy()

    res = Obsdata.res()
    beamparams = Obsdata.fit_beam()

    imvec_init = 0*InitIm.imvec[embed_mask]
    vis_init = numpy.zeros(len(vis),dtype='complex128')

    imvec_current = imvec_init
    vis_current = vis_init

    chisq_init = numpy.sum(weights*numpy.abs(vis-vis_init)**2)
    rchisq_init = numpy.sum(weights_nat*numpy.abs(vis-vis_init)**2)/(2*len(weights_nat))
    chisq_current = chisq_init
    rchisq_current = rchisq_init

    # clean loop
    print("\n")
    for it in range(niter):

        resid_current = vis - vis_current

        plan.f =  weights * resid_current
        plan.adjoint()
        out = numpy.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))
        deltas_all = out/ weights_norm
        deltas = deltas_all[embed_mask]

        chisq_map_all = chisq_current - (deltas_all**2)*weights_norm
        chisq_map = chisq_map_all[embed_mask]

        # Visibility space clean
        if method=='min_chisq':
            component_loc_idx = numpy.argmin(chisq_map)

        # Image space clean
        elif method=='max_delta':
            component_loc_idx = numpy.argmax(deltas)

        else:
            raise Exception("method should be 'min_chisq' or 'max_delta'!")

        # display images of delta and chisq
        if show_updates:
            DeltasIm.imvec = deltas_all
            plot_i(DeltasIm, it, chisq_current,fig=0, cmap='afmhot')

            ChisqIm.imvec = -chisq_map_all
            plot_i(ChisqIm, it, chisq_current,fig=1, cmap='cool')

        # clean component location
        component_loc_x = coord[component_loc_idx][0]
        component_loc_y = coord[component_loc_idx][1]
        component_strength = loop_gain*deltas[component_loc_idx]

        # update vis and imvec
        imvec_current[component_loc_idx] += component_strength

        #TODO how to incorporate pulse function?
        vis_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv[:,0]*component_loc_x + uv[:,1]*component_loc_y))

        # update chi^2 and output image
        chisq_current = numpy.sum(weights*numpy.abs(vis-vis_current)**2)
        rchisq_current = numpy.sum(weights_nat*numpy.abs(vis-vis_current)**2)/(2*len(weights_nat))

        print(it+1,component_strength, chisq_current, rchisq_current, component_loc_x/const_def.RADPERUAS, component_loc_y/const_def.RADPERUAS)

        OutputIm.imvec = imvec_current
        if show_updates:
            OutputIm.imvec = imager_utils.embed(OutputIm.imvec, embed_mask, clipfloor=0., randomfloor=False)
            OutputImBlur = OutputIm.blur_gauss(beamparams)
            plot_i(OutputImBlur, it, rchisq_current, fig=2)

    OutputIm.imvec = imager_utils.embed(OutputIm.imvec, embed_mask, clipfloor=0., randomfloor=False)
    return OutputIm


#solve full 5th order polynomial
def dd_clean_bispec_full(Obsdata, InitIm, niter=1, clipfloor=-1, loop_gain=.1,
                        weighting='uniform', bscount="min",show_updates=True,
                        fft_pad_factor=FFT_PAD_DEFAULT, p_rad=NFFT_KERSIZE_DEFAULT):


    # limit imager range to prior values > clipfloor
    embed_mask = InitIm.imvec >= clipfloor

    # get data
    biarr = Obsdata.bispectra(mode="all", count=bscount)
    uv1 = numpy.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = numpy.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = numpy.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bs = biarr['bispec']
    sigma = biarr['sigmab']

    # necessary nfft infos
    npad = int(fft_pad_factor * numpy.max((InitIm.xdim, InitIm.ydim)))
    nfft_info1 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv1)
    nfft_info2 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv2)
    nfft_info3 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv3)

    nfft_info11 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv1)
    nfft_info22 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv2)
    nfft_info33 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv3)

    nfft_info12 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv1-uv2)
    nfft_info23 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv2-uv3)
    nfft_info31 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv3-uv1)

    # TODO do we use pulse factors?
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    plan11 = nfft_info11.plan
    pulsefac11 = nfft_info11.pulsefac
    plan22 = nfft_info22.plan
    pulsefac22 = nfft_info22.pulsefac
    plan33 = nfft_info33.plan
    pulsefac33 = nfft_info33.pulsefac

    plan12 = nfft_info12.plan
    pulsefac12 = nfft_info12.pulsefac
    plan23 = nfft_info23.plan
    pulsefac23 = nfft_info23.pulsefac
    plan31 = nfft_info31.plan
    pulsefac31 = nfft_info31.pulsefac

    # Weights
    weights_nat = 1./(sigma**2)
    if weighting=='uniform':
        weights = numpy.ones(len(weights_nat))
    elif weighting=='natural':
        weights=weights_nat
    else:
        raise Exception("weighting must be 'uniform' or 'natural'")
    weights_norm = numpy.sum(weights)

    # Coordinate matrix
    # TODO what if the image is odd?
    coord = InitIm.psize * numpy.array([[[x,y] for x in numpy.arange(InitIm.xdim//2,-InitIm.xdim//2,-1)]
                                            for y in numpy.arange(InitIm.ydim//2,-InitIm.ydim//2,-1)])
    coord = coord.reshape(InitIm.ydim*InitIm.xdim, 2)
    coord = coord[embed_mask]

    # Initial imvec and visibilities
    # TODO currently initialized to zero!!
    OutputIm = InitIm.copy()
    DeltasIm = InitIm.copy()
    ChisqIm = InitIm.copy()

    res = Obsdata.res()
    beamparams = Obsdata.fit_beam()

    imvec_init = 0*InitIm.imvec[embed_mask]
    vis1_init = numpy.zeros(len(bs), dtype='complex128')
    vis2_init = numpy.zeros(len(bs), dtype='complex128')
    vis3_init = numpy.zeros(len(bs), dtype='complex128')
    bs_init = vis1_init*vis2_init*vis3_init
    chisq_init =  numpy.sum(weights*numpy.abs(bs - bs_init)**2)
    rchisq_init =  numpy.sum(weights_nat*numpy.abs(bs - bs_init)**2)/(2*len(weights_nat))

    imvec_current = imvec_init
    vis1_current = vis1_init
    vis2_current = vis2_init
    vis3_current = vis3_init
    bs_current = bs_init
    chisq_current = chisq_init
    rchisq_current = rchisq_init

    # clean loop
    print("\n")
    for it in range(niter):
        t = time.time()
        # compute delta at each location
        resid_current = bs - bs_current
        vis12_current = vis1_current*vis2_current
        vis13_current = vis1_current*vis3_current
        vis23_current = vis2_current*vis3_current

        # center the first component automatically
        # since initial image is empty, must go to higher order in delta in solution
        # TODO generalize to non-empty initial image!
        if it==0:

            A = numpy.sum(weights*numpy.real(resid_current))
            B = numpy.sum(weights)

            component_strength = numpy.cbrt(A/B)
            component_loc_idx = (InitIm.ydim//2)*InitIm.xdim  + InitIm.xdim//2 #TODO is this right for odd images??+/- 1??

        else:
            # First calculate P (1st order)
            plan1.f =  weights * resid_current * vis23_current.conj()
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * resid_current * vis13_current.conj()
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * resid_current * vis12_current.conj()
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            P = -2 * (out1 + out2 + out3)

            # Then calculate Q (2nd order)
            plan12.f =  weights * vis13_current*vis23_current.conj()
            plan12.adjoint()
            out12 = numpy.real((plan12.f_hat.copy().T).reshape(nfft_info12.xdim*nfft_info12.ydim))
            plan23.f =  weights * vis12_current*vis13_current.conj()
            plan23.adjoint()
            out23 = numpy.real((plan23.f_hat.copy().T).reshape(nfft_info23.xdim*nfft_info23.ydim))
            plan31.f =  weights * vis23_current*vis12_current.conj()
            plan31.adjoint()
            out31 = numpy.real((plan31.f_hat.copy().T).reshape(nfft_info31.xdim*nfft_info31.ydim))

            plan1.f =  weights * resid_current.conj() * vis1_current
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * resid_current.conj() * vis2_current
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * resid_current.conj() * vis3_current
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            Q0 = numpy.sum(weights*(numpy.abs(vis12_current)**2 + numpy.abs(vis23_current)**2 + numpy.abs(vis13_current)**2))
            Q1 = 2  * (out12 + out23 + out31)
            Q2 = -2 * (out1 + out2 + out3)
            Q = 2*(Q0 + Q1 + Q2)

            #Calculate R (3rd order)
            plan11.f =  weights * vis1_current.conj() * vis23_current
            plan11.adjoint()
            out11 = numpy.real((plan11.f_hat.copy().T).reshape(nfft_info11.xdim*nfft_info11.ydim))
            plan22.f =  weights * vis2_current.conj() * vis13_current
            plan22.adjoint()
            out22 = numpy.real((plan22.f_hat.copy().T).reshape(nfft_info22.xdim*nfft_info22.ydim))
            plan33.f =  weights * vis3_current.conj() * vis12_current
            plan33.adjoint()
            out33 = numpy.real((plan33.f_hat.copy().T).reshape(nfft_info33.xdim*nfft_info33.ydim))

            plan1.f =  weights * vis1_current * (numpy.abs(vis2_current)**2 + numpy.abs(vis3_current)**2)
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * vis2_current * (numpy.abs(vis1_current)**2 + numpy.abs(vis3_current)**2)
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * vis3_current * (numpy.abs(vis1_current)**2 + numpy.abs(vis2_current)**2)
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            R0 = -numpy.sum(weights*numpy.real(resid_current))
            R1 = numpy.real(out11 + out22 + out33)
            R2 = numpy.real(out1 + out2 + out3)
            R = 6*(R0 + R1 + R2)

            # Now find S (4th order)
            plan12.f =  weights * vis1_current*vis2_current.conj()
            plan12.adjoint()
            out12 = numpy.real((plan12.f_hat.copy().T).reshape(nfft_info12.xdim*nfft_info12.ydim))
            plan23.f =  weights * vis2_current*vis3_current.conj()
            plan23.adjoint()
            out23 = numpy.real((plan23.f_hat.copy().T).reshape(nfft_info23.xdim*nfft_info23.ydim))
            plan31.f =  weights * vis3_current*vis1_current.conj()
            plan31.adjoint()
            out31 = numpy.real((plan31.f_hat.copy().T).reshape(nfft_info31.xdim*nfft_info31.ydim))

            plan1.f =  weights * vis23_current.conj()
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * vis13_current.conj()
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * vis12_current.conj()
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            S0 = numpy.sum(weights*(numpy.abs(vis1_current)**2 + numpy.abs(vis2_current)**2 + numpy.abs(vis3_current)**2))
            S1 = 2  * (out12 + out23 + out31)
            S2 = 2 * (out1 + out2 + out3)
            S = 4*(S0 + S1 + S2)


            # T (5th order)
            plan1.f =  weights * vis1_current
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * vis2_current
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * vis3_current
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            T = 10*(out1 + out2 + out3)

            # Finally U (6th order)
            U = 6*weights_norm * numpy.ones(T.shape)

            # Find Component
            deltas = numpy.zeros(len(P))
            chisq_map = numpy.zeros(len(P))
            for i in range(len(P)):
                polynomial_params = numpy.array([P[i], Q[i], R[i], S[i], T[i], U[i]])
                allroots = polynomial.polyroots(polynomial_params)

                # test roots to see which minimizes chi^2
                newchisq = chisq_current
                delta = 0
                for j in range(len(allroots)):
                    root = allroots[j]
                    if numpy.imag(root)!=0: continue

                    trialchisq = chisq_current + P[i]*root + 0.5*Q[i]*root**2 + (1./3.)*R[i]*root**3 + 0.25*S[i]*root**4 + 0.2*T[i]*root**5 + (1./6.)*U[i]*root**6
                    if trialchisq < newchisq:
                        delta = root
                        newchisq = trialchisq

                deltas[i] = delta
                chisq_map[i] = newchisq

            #plot deltas and chi^2 map
            if show_updates:
                DeltasIm.imvec = deltas
                plot_i(DeltasIm, it, chisq_current,fig=0, cmap='afmhot')

                ChisqIm.imvec = -chisq_map
                plot_i(ChisqIm, it, chisq_current,fig=1, cmap='cool')


            component_loc_idx = numpy.argmin(chisq_map[embed_mask])
            component_strength = loop_gain*(deltas[embed_mask])[component_loc_idx]

        # clean component location
        component_loc_x = coord[component_loc_idx][0]
        component_loc_y = coord[component_loc_idx][1]

        # update imvec, vis, bispec
        imvec_current[component_loc_idx] += component_strength

        #TODO how to incorporate pulse function?
        vis1_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv1[:,0]*component_loc_x + uv1[:,1]*component_loc_y))
        vis2_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv2[:,0]*component_loc_x + uv2[:,1]*component_loc_y))
        vis3_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv3[:,0]*component_loc_x + uv3[:,1]*component_loc_y))
        bs_current = vis1_current * vis2_current * vis3_current

        # update chi^2 and output image
        chisq_current = numpy.sum(weights*numpy.abs(bs - bs_current)**2)
        rchisq_current = numpy.sum(weights_nat*numpy.abs(bs - bs_current)**2)/(2*len(weights_nat))

        print("it %i: %f (%.2f , %.2f) %.4f" % (it+1, component_strength, component_loc_x/const_def.RADPERUAS, component_loc_y/const_def.RADPERUAS, chisq_current))

        OutputIm.imvec = imvec_current
        if show_updates:
            OutputIm.imvec = imager_utils.embed(OutputIm.imvec, embed_mask, clipfloor=0., randomfloor=False)
            OutputImBlur = OutputIm.blur_gauss(beamparams)
            plot_i(OutputImBlur, it, rchisq_current,fig=2)

    OutputIm.imvec = imager_utils.embed(OutputIm.imvec, embed_mask, clipfloor=0., randomfloor=False)
    return OutputIm




#solve full 5th order polynomial
#weight imaginary term differently
def dd_clean_bispec_imweight(Obsdata, InitIm, niter=1, clipfloor=-1, ttype="direct", loop_gain=.1, loop_gain_init=1,
                             weighting='uniform', bscount="min", imweight=1, show_updates=True,
                             fft_pad_factor=FFT_PAD_DEFAULT, p_rad=NFFT_KERSIZE_DEFAULT):


    imag_weight=imweight
    # limit imager range to prior values > clipfloor
    embed_mask = InitIm.imvec >= clipfloor

    # get data
    biarr = Obsdata.bispectra(mode="all", count=bscount)
    uv1 = numpy.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = numpy.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = numpy.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bs = biarr['bispec']
    sigma = biarr['sigmab']

    # necessary nfft infos
    npad = int(fft_pad_factor * numpy.max((InitIm.xdim, InitIm.ydim)))
    nfft_info1 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv1)
    nfft_info2 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv2)
    nfft_info3 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv3)

    nfft_info11 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv1)
    nfft_info22 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv2)
    nfft_info33 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv3)

    nfft_info12 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv1-uv2)
    nfft_info23 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv2-uv3)
    nfft_info31 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv3-uv1)

    # TODO do we use pulse factors?
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    plan11 = nfft_info11.plan
    pulsefac11 = nfft_info11.pulsefac
    plan22 = nfft_info22.plan
    pulsefac22 = nfft_info22.pulsefac
    plan33 = nfft_info33.plan
    pulsefac33 = nfft_info33.pulsefac

    plan12 = nfft_info12.plan
    pulsefac12 = nfft_info12.pulsefac
    plan23 = nfft_info23.plan
    pulsefac23 = nfft_info23.pulsefac
    plan31 = nfft_info31.plan
    pulsefac31 = nfft_info31.pulsefac

    # Weights
    weights_nat = 1./(sigma**2)
    if weighting=='uniform':
        weights = numpy.ones(len(weights_nat))
    elif weighting=='natural':
        weights=weights_nat
    else:
        raise Exception("weighting must be 'uniform' or 'natural'")
    weights_norm = numpy.sum(weights)

    # Coordinate matrix
    # TODO do we need to make sure this corresponds exactly with what NFFT is doing?
    # TODO what if the image is odd?
    coord = InitIm.psize * numpy.array([[[x,y] for x in numpy.arange(InitIm.xdim//2,-InitIm.xdim//2,-1)]
                                            for y in numpy.arange(InitIm.ydim//2,-InitIm.ydim//2,-1)])
    coord = coord.reshape(InitIm.ydim*InitIm.xdim, 2)
    coord = coord[embed_mask]

    # Initial imvec and visibilities
    # TODO currently initialized to zero!!
    OutputIm = InitIm.copy()
    DeltasIm = InitIm.copy()
    ChisqIm = InitIm.copy()

    res = Obsdata.res()
    beamparams = Obsdata.fit_beam()

    imvec_init = 0*InitIm.imvec[embed_mask]

    vis1_init = numpy.zeros(len(bs), dtype='complex128')
    vis2_init = numpy.zeros(len(bs), dtype='complex128')
    vis3_init = numpy.zeros(len(bs), dtype='complex128')
    bs_init = vis1_init*vis2_init*vis3_init
    chisq_init =  numpy.sum(weights*(numpy.real(bs - bs_init)**2 + imweight*numpy.imag(bs-bs_init)**2))
    rchisq_init =  numpy.sum(weights_nat*numpy.abs(bs - bs_init)**2)/(2*len(weights_nat))

    imvec_current = imvec_init
    vis1_current = vis1_init
    vis2_current = vis2_init
    vis3_current = vis3_init
    bs_current = bs_init
    chisq_current = chisq_init
    rchisq_current = rchisq_init

    # clean loop
    print("\n")
    for it in range(niter):
        t = time.time()
        # compute delta at each location
        resid_current = bs - bs_current
        vis12_current = vis1_current*vis2_current
        vis13_current = vis1_current*vis3_current
        vis23_current = vis2_current*vis3_current

        # center the first component automatically
        # since initial image is empty, must go to higher order in delta in solution
        # TODO generalize to non-empty initial image!
        if it==0:

            A = numpy.sum(weights*numpy.real(resid_current))
            B = numpy.sum(weights)

            component_strength = loop_gain_init*numpy.cbrt(A/B)
            component_loc_idx = (InitIm.ydim//2)*InitIm.xdim  + InitIm.xdim//2 #TODO is this right for odd images??+/- 1??

        else:
            # First calculate P (1st order)
            plan1.f =  weights * numpy.real(resid_current.conj() * vis23_current)
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * numpy.real(resid_current.conj() * vis13_current)
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * numpy.real(resid_current.conj() * vis12_current)
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            PRe = -2 * (out1 + out2 + out3)

            plan1.f =  weights * numpy.imag(resid_current.conj() * vis23_current)
            plan1.adjoint()
            out1 = numpy.imag((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * numpy.imag(resid_current.conj() * vis13_current)
            plan2.adjoint()
            out2 = numpy.imag((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * numpy.imag(resid_current.conj() * vis12_current)
            plan3.adjoint()
            out3 = numpy.imag((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            PIm = -2 * (out1 + out2 + out3)

            P = PRe + imag_weight*PIm

            # Then calculate Q (2nd order)
            plan12.f =  weights * numpy.real(vis13_current.conj()*vis23_current)
            plan12.adjoint()
            out12 = numpy.real((plan12.f_hat.copy().T).reshape(nfft_info12.xdim*nfft_info12.ydim))
            plan23.f =  weights * numpy.real(vis12_current.conj()*vis13_current)
            plan23.adjoint()
            out23 = numpy.real((plan23.f_hat.copy().T).reshape(nfft_info23.xdim*nfft_info23.ydim))
            plan31.f =  weights * numpy.real(vis23_current.conj()*vis12_current)
            plan31.adjoint()
            out31 = numpy.real((plan31.f_hat.copy().T).reshape(nfft_info31.xdim*nfft_info31.ydim))

            plan1.f =  weights * numpy.real(resid_current * vis1_current.conj())
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * numpy.real(resid_current * vis2_current.conj())
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * numpy.real(resid_current * vis3_current.conj())
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            Q0Re = numpy.sum(weights*(numpy.real(vis12_current)**2 + numpy.real(vis23_current)**2 + numpy.real(vis13_current)**2))
            Q1Re = 2  * (out12 + out23 + out31)
            Q2Re = -2*(out1 + out2 + out3)
            QRe = 2*(Q0Re + Q1Re + Q2Re)

            plan12.f =  weights * numpy.imag(vis13_current.conj()*vis23_current)
            plan12.adjoint()
            out12 = numpy.imag((plan12.f_hat.copy().T).reshape(nfft_info12.xdim*nfft_info12.ydim))
            plan23.f =  weights * numpy.imag(vis12_current.conj()*vis13_current)
            plan23.adjoint()
            out23 = numpy.imag((plan23.f_hat.copy().T).reshape(nfft_info23.xdim*nfft_info23.ydim))
            plan31.f =  weights * numpy.imag(vis23_current.conj()*vis12_current)
            plan31.adjoint()
            out31 = numpy.imag((plan31.f_hat.copy().T).reshape(nfft_info31.xdim*nfft_info31.ydim))

            plan1.f =  weights * numpy.imag(resid_current * vis1_current.conj())
            plan1.adjoint()
            out1 = numpy.imag((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * numpy.imag(resid_current * vis2_current.conj())
            plan2.adjoint()
            out2 = numpy.imag((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * numpy.imag(resid_current * vis3_current.conj())
            plan3.adjoint()
            out3 = numpy.imag((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            Q0Im = numpy.sum(weights*(numpy.imag(vis12_current)**2 + numpy.imag(vis23_current)**2 + numpy.imag(vis13_current)**2))
            Q1Im = 2  * (out12 + out23 + out31)
            Q2Im = -2*(out1 + out2 + out3)
            QIm = 2*(Q0Im + Q1Im + Q2Im)

            Q = QRe + imag_weight*QIm

            #Calculate R (3rd order)
            plan11.f =  weights * numpy.real(vis1_current * vis23_current.conj())
            plan11.adjoint()
            out11 = numpy.real((plan11.f_hat.copy().T).reshape(nfft_info11.xdim*nfft_info11.ydim))
            plan22.f =  weights * numpy.real(vis2_current * vis13_current.conj())
            plan22.adjoint()
            out22 = numpy.real((plan22.f_hat.copy().T).reshape(nfft_info22.xdim*nfft_info22.ydim))
            plan33.f =  weights * numpy.real(vis3_current * vis12_current.conj())
            plan33.adjoint()
            out33 = numpy.real((plan33.f_hat.copy().T).reshape(nfft_info33.xdim*nfft_info33.ydim))

            plan1.f =  weights * numpy.real(vis1_current.conj()) * (numpy.abs(vis2_current)**2 + numpy.abs(vis3_current)**2)
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * numpy.real(vis2_current.conj()) * (numpy.abs(vis1_current)**2 + numpy.abs(vis3_current)**2)
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * numpy.real(vis3_current.conj()) * (numpy.abs(vis1_current)**2 + numpy.abs(vis2_current)**2)
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            R0Re = -numpy.sum(weights*numpy.real(resid_current))
            R1Re = out11 + out22 + out33
            R2Re = out1 + out2 + out3
            RRe = 6*(R0Re + R1Re + R2Re)

            plan11.f =  weights * numpy.imag(vis1_current * vis23_current.conj())
            plan11.adjoint()
            out11 = numpy.imag((plan11.f_hat.copy().T).reshape(nfft_info11.xdim*nfft_info11.ydim))
            plan22.f =  weights * numpy.imag(vis2_current * vis13_current.conj())
            plan22.adjoint()
            out22 = numpy.imag((plan22.f_hat.copy().T).reshape(nfft_info22.xdim*nfft_info22.ydim))
            plan33.f =  weights * numpy.imag(vis3_current * vis12_current.conj())
            plan33.adjoint()
            out33 = numpy.imag((plan33.f_hat.copy().T).reshape(nfft_info33.xdim*nfft_info33.ydim))

            plan1.f =  weights * numpy.imag(vis1_current.conj()) * (numpy.abs(vis2_current)**2 + numpy.abs(vis3_current)**2)
            plan1.adjoint()
            out1 = numpy.imag((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * numpy.imag(vis2_current.conj()) * (numpy.abs(vis1_current)**2 + numpy.abs(vis3_current)**2)
            plan2.adjoint()
            out2 = numpy.imag((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * numpy.imag(vis3_current.conj()) * (numpy.abs(vis1_current)**2 + numpy.abs(vis2_current)**2)
            plan3.adjoint()
            out3 = numpy.imag((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            R0Im = 0
            R1Im = out11 + out22 + out33
            R2Im = out1 + out2 + out3
            RIm = 6*(R0Im + R1Im + R2Im)

            R = RRe + imag_weight*RIm

            # Now find S (4th order)
            plan12.f =  weights * numpy.real(vis1_current.conj()*vis2_current)
            plan12.adjoint()
            out12 = numpy.real((plan12.f_hat.copy().T).reshape(nfft_info12.xdim*nfft_info12.ydim))
            plan23.f =  weights * numpy.real(vis2_current.conj()*vis3_current)
            plan23.adjoint()
            out23 = numpy.real((plan23.f_hat.copy().T).reshape(nfft_info23.xdim*nfft_info23.ydim))
            plan31.f =  weights * numpy.real(vis3_current.conj()*vis1_current)
            plan31.adjoint()
            out31 = numpy.real((plan31.f_hat.copy().T).reshape(nfft_info31.xdim*nfft_info31.ydim))

            plan1.f =  weights * vis23_current.conj()
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * vis13_current.conj()
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * vis12_current.conj()
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            S0Re = numpy.sum(weights*(numpy.real(vis1_current)**2 + numpy.real(vis2_current)**2 + numpy.real(vis3_current)**2))
            S1Re = 2  * (out12 + out23 + out31)
            S2Re = 2 * (out1 + out2 + out3)
            SRe = 4*(S0Re + S1Re + S2Re)

            plan12.f =  weights * numpy.imag(vis1_current.conj()*vis2_current)
            plan12.adjoint()
            out12 = numpy.imag((plan12.f_hat.copy().T).reshape(nfft_info12.xdim*nfft_info12.ydim))
            plan23.f =  weights * numpy.imag(vis2_current.conj()*vis3_current)
            plan23.adjoint()
            out23 = numpy.imag((plan23.f_hat.copy().T).reshape(nfft_info23.xdim*nfft_info23.ydim))
            plan31.f =  weights * numpy.imag(vis3_current.conj()*vis1_current)
            plan31.adjoint()
            out31 = numpy.imag((plan31.f_hat.copy().T).reshape(nfft_info31.xdim*nfft_info31.ydim))

            S0Im = numpy.sum(weights*(numpy.imag(vis1_current)**2 + numpy.imag(vis2_current)**2 + numpy.imag(vis3_current)**2))
            S1Im = 2  * (out12 + out23 + out31)
            S2Im = 0
            SIm = 4*(S0Im + S1Im + S2Im)

            S = SRe + imag_weight*SIm

            # T (5th order)
            plan1.f =  weights * vis1_current
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights * vis2_current
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights * vis3_current
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            TRe = 10*(out1 + out2 + out3)
            TIm = 0.
            T = TRe + imag_weight*TIm

            # Finally U (6th order)
            URe = 6*weights_norm * numpy.ones(T.shape)
            UIm = 0.
            U = URe + imag_weight*UIm

            #Find Component
            deltas = numpy.zeros(len(P))
            chisq_map = numpy.zeros(len(P))
            for i in range(len(P)):
                if embed_mask[i]:
                    polynomial_params = numpy.array([P[i], Q[i], R[i], S[i], T[i], U[i]])
                    allroots = polynomial.polyroots(polynomial_params)

                    # test roots to see which minimizes chi^2
                    newchisq = chisq_current
                    delta = 0
                    for j in range(len(allroots)):
                        root = allroots[j]
                        if numpy.imag(root)!=0: continue

                        trialchisq = chisq_current + P[i]*root + 0.5*Q[i]*root**2 + (1./3.)*R[i]*root**3 + 0.25*S[i]*root**4 + 0.2*T[i]*root**5 + (1./6.)*U[i]*root**6
                        if trialchisq < newchisq:
                            delta = root
                            newchisq = trialchisq

                    deltas[i] = delta
                    chisq_map[i] = newchisq
                else:
                    deltas[i]=0
                    chisq_map[i]=chisq_current

            #print ("step time %i: %f s" % (it+1, time.time() -t))
            #chisq_map = chisq_current + P*deltas + 0.5*Q*deltas**2 + (1./3.)*R*deltas**3 + 0.25*S*deltas**4 + 0.2*T*deltas**5 + (1./6.)*U*deltas**6

            #Plot delta and chi^2 map
            if show_updates:
                DeltasIm.imvec = deltas
                plot_i(DeltasIm, it, chisq_current,fig=0, cmap='afmhot')

                ChisqIm.imvec = -chisq_map
                plot_i(ChisqIm, it, chisq_current,fig=1, cmap='cool')

            component_loc_idx = numpy.argmin(chisq_map[embed_mask])
            component_strength = loop_gain*(deltas[embed_mask])[component_loc_idx]

        # clean component location
        component_loc_x = coord[component_loc_idx][0]
        component_loc_y = coord[component_loc_idx][1]

        # update imvec, vis, bispec
        imvec_current[component_loc_idx] += component_strength

        #TODO how to incorporate pulse function?
        vis1_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv1[:,0]*component_loc_x + uv1[:,1]*component_loc_y))
        vis2_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv2[:,0]*component_loc_x + uv2[:,1]*component_loc_y))
        vis3_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv3[:,0]*component_loc_x + uv3[:,1]*component_loc_y))
        bs_current = vis1_current * vis2_current * vis3_current

        # update chi^2 and output image
        chisq_current = numpy.sum(weights*numpy.abs(bs - bs_current)**2)
        rchisq_current = numpy.sum(weights_nat*numpy.abs(bs - bs_current)**2)/(2*len(weights_nat))

        print("it %i: %f (%.2f , %.2f) %.4f" % (it+1, component_strength, component_loc_x/const_def.RADPERUAS, component_loc_y/const_def.RADPERUAS, chisq_current))

        OutputIm.imvec = imvec_current
        if show_updates:
            OutputIm.imvec = imager_utils.embed(OutputIm.imvec, embed_mask, clipfloor=0., randomfloor=False)
            OutputImBlur = OutputIm.blur_gauss(beamparams)
            plot_i(OutputImBlur, it, rchisq_current,fig=2)
    return OutputIm


#amplitude and "closure phase" term
def dd_clean_amp_cphase(Obsdata, InitIm, niter=1, clipfloor=-1, loop_gain=.1, loop_gain_init=1,phaseweight=1,
                        weighting='uniform', bscount="min",no_neg_comps=False,
                        fft_pad_factor=FFT_PAD_DEFAULT, p_rad=NFFT_KERSIZE_DEFAULT, show_updates=True):


    # limit imager range to prior values > clipfloor
    embed_mask = InitIm.imvec >= clipfloor

    # get data
    amp2 = numpy.abs(Obsdata.data['vis'])**2 #TODO debias??
    sigma_amp2 = Obsdata.data['sigma']**2
    uv = numpy.hstack((Obsdata.data['u'].reshape(-1,1), Obsdata.data['v'].reshape(-1,1)))

    biarr = Obsdata.bispectra(mode="all", count=bscount)
    uv1 = numpy.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = numpy.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = numpy.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bs = biarr['bispec']
    sigma_bs = biarr['sigmab']

    # necessary nfft infos
    npad = int(fft_pad_factor * numpy.max((InitIm.xdim, InitIm.ydim)))

    nfft_infoA = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv)
    nfft_infoB = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, 2*uv)

    nfft_info1 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv1)
    nfft_info2 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv2)
    nfft_info3 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv3)

    nfft_info11 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv1)
    nfft_info22 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv2)
    nfft_info33 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, -2*uv3)

    nfft_info12 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv1-uv2)
    nfft_info23 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv2-uv3)
    nfft_info31 = obs_helpers.NFFTInfo(InitIm.xdim, InitIm.ydim, InitIm.psize, InitIm.pulse, npad, p_rad, uv3-uv1)

    # TODO do we use pulse factors?
    planA = nfft_infoA.plan
    pulsefacA = nfft_infoA.pulsefac
    planB = nfft_infoB.plan
    pulsefacB = nfft_infoB.pulsefac

    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    plan11 = nfft_info11.plan
    pulsefac11 = nfft_info11.pulsefac
    plan22 = nfft_info22.plan
    pulsefac22 = nfft_info22.pulsefac
    plan33 = nfft_info33.plan
    pulsefac33 = nfft_info33.pulsefac

    plan12 = nfft_info12.plan
    pulsefac12 = nfft_info12.pulsefac
    plan23 = nfft_info23.plan
    pulsefac23 = nfft_info23.pulsefac
    plan31 = nfft_info31.plan
    pulsefac31 = nfft_info31.pulsefac

    # Weights
    weights_amp2_nat = 1./(sigma_amp2**2)
    if weighting=='uniform':
        weights_amp2 = numpy.ones(len(weights_amp2_nat)) * numpy.median(weights_amp2_nat) #TODO for scaling weights are all given by median natural weight??
    elif weighting=='natural':
        weights_amp2=weights_amp2_nat
    else:
        raise Exception("weighting must be 'uniform' or 'natural'")
    weights_amp2 = weights_amp2 / float(len(weights_amp2_nat))
    weights_amp2_norm = numpy.sum(weights_amp2)

    weights_bs_nat = (numpy.abs(bs)**2) / (sigma_bs**2)
    if weighting=='uniform':
        weights_bs = numpy.ones(len(weights_bs_nat)) * numpy.median(weights_bs_nat)     #TODO for scaling weights are all given by median natural weight??
    elif weighting=='natural':
        weights_bs = weights_bs_nat
    else:
        raise Exception("weighting must be 'uniform' or 'natural'")
    weights_bs = weights_bs / numpy.abs(bs)**2 #weight down by 1/bs^2 only works for uniform??
    weights_bs = weights_bs / float(len(weights_bs_nat))
    weights_bs_norm = numpy.sum(weights_bs)

    # Coordinate matrix
    # TODO do we need to make sure this corresponds exactly with what NFFT is doing?
    # TODO what if the image is odd?
    coord = InitIm.psize * numpy.array([[[x,y] for x in numpy.arange(InitIm.xdim//2,-InitIm.xdim//2,-1)]
                                            for y in numpy.arange(InitIm.ydim//2,-InitIm.ydim//2,-1)])
    coord = coord.reshape(InitIm.ydim*InitIm.xdim, 2)
    coord = coord[embed_mask]

    # Initial imvec and visibilities
    # TODO currently initialized to zero!!
    OutputIm = InitIm.copy()
    DeltasIm = InitIm.copy()
    ChisqIm = InitIm.copy()

    res = Obsdata.res()
    beamparams = Obsdata.fit_beam()

    imvec_init = 0*InitIm.imvec[embed_mask]
    vis_init =  numpy.zeros(len(amp2), dtype='complex128')
    chisq_amp2_init = numpy.sum(weights_amp2*(amp2 - numpy.abs(vis_init)**2)**2)
    rchisq_amp2_init = numpy.sum(weights_amp2_nat*(amp2 - numpy.abs(vis_init)**2)**2)

    vis1_init = numpy.zeros(len(bs), dtype='complex128')
    vis2_init = numpy.zeros(len(bs), dtype='complex128')
    vis3_init = numpy.zeros(len(bs), dtype='complex128')
    bs_init = vis1_init*vis2_init*vis3_init
    chisq_bs_init =  numpy.sum(weights_bs*numpy.abs(bs - bs_init)**2)
    rchisq_bs_init =  numpy.sum(weights_bs_nat*numpy.abs(bs - bs_init)**2)

    chisq_init = chisq_amp2_init + phaseweight*chisq_bs_init

    imvec_current = imvec_init.copy()
    vis_current = vis_init.copy()
    vis1_current = vis1_init.copy()
    vis2_current = vis2_init.copy()
    vis3_current = vis3_init.copy()
    bs_current = bs_init.copy()
    chisq_amp2_current = chisq_amp2_init
    rchisq_amp2_current = rchisq_amp2_init
    chisq_bs_current = chisq_bs_init
    rchisq_bs_current = rchisq_bs_init
    chisq_current = chisq_init

    # clean loop
    print("\n")
    for it in range(niter):
        t = time.time()

        # center the first component automatically
        # since initial image is empty, must go to higher order in delta in solution
        # TODO generalize to non-empty initial image!
        # BASE INITIAL POINT SOURCE ENTIRELY ON VISIBILITY AMPLITUDES
        if it==0:

            A = numpy.sum(weights_amp2*amp2)
            B = weights_amp2_norm

            component_strength = loop_gain_init*numpy.sqrt(A/B)
            component_loc_idx = (InitIm.ydim//2)*InitIm.xdim  + InitIm.xdim//2 #TODO is this right for odd images??+/- 1??

        else:

            #Amplitude part
            # First calculate A (1st order)
            planA.f =  weights_amp2 *(amp2_current-amp2)*vis_current
            planA.adjoint()
            out = numpy.real((planA.f_hat.copy().T).reshape(nfft_infoA.xdim*nfft_infoA.ydim))
            A = 4*out

            # Then calculate B (2nd order)
            planB.f =  weights_amp2 * vis_current*vis_current
            planB.adjoint()
            out = numpy.real((planB.f_hat.copy().T).reshape(nfft_infoB.xdim*nfft_infoB.ydim))

            B0 = numpy.sum(weights_amp2*(2*amp2_current - amp2))
            B1 = out
            B = 4*(B0 + B1)

            #Calculate C (3rd order)
            planA.f =  weights_amp2 * vis_current
            planA.adjoint()
            out = numpy.real((planA.f_hat.copy().T).reshape(nfft_infoA.xdim*nfft_infoA.ydim))

            C = 12*out

            # Now find D (4th order)
            D = 4*weights_amp2_norm * numpy.ones(C.shape)

            #"Closure Phase" part
            resid_current = bs - bs_current
            vis12_current = vis1_current*vis2_current
            vis13_current = vis1_current*vis3_current
            vis23_current = vis2_current*vis3_current

            # First calculate P (1st order)
            plan1.f =  weights_bs * resid_current * vis23_current.conj()
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights_bs * resid_current * vis13_current.conj()
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights_bs * resid_current * vis12_current.conj()
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            P = -2 * (out1 + out2 + out3)

            # Then calculate Q (2nd order)
            plan12.f =  weights_bs * vis13_current*vis23_current.conj()
            plan12.adjoint()
            out12 = numpy.real((plan12.f_hat.copy().T).reshape(nfft_info12.xdim*nfft_info12.ydim))
            plan23.f =  weights_bs * vis12_current*vis13_current.conj()
            plan23.adjoint()
            out23 = numpy.real((plan23.f_hat.copy().T).reshape(nfft_info23.xdim*nfft_info23.ydim))
            plan31.f =  weights_bs * vis23_current*vis12_current.conj()
            plan31.adjoint()
            out31 = numpy.real((plan31.f_hat.copy().T).reshape(nfft_info31.xdim*nfft_info31.ydim))

            plan1.f =  weights_bs * resid_current.conj() * vis1_current
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights_bs * resid_current.conj() * vis2_current
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights_bs * resid_current.conj() * vis3_current
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            Q0 = numpy.sum(weights_bs*(numpy.abs(vis12_current)**2 + numpy.abs(vis23_current)**2 + numpy.abs(vis13_current)**2))
            Q1 = 2  * (out12 + out23 + out31)
            Q2 = -2 * (out1 + out2 + out3)
            Q = 2*(Q0 + Q1 + Q2)

            #Calculate R (3rd order)
            plan11.f =  weights_bs * vis1_current.conj() * vis23_current
            plan11.adjoint()
            out11 = numpy.real((plan11.f_hat.copy().T).reshape(nfft_info11.xdim*nfft_info11.ydim))
            plan22.f =  weights_bs * vis2_current.conj() * vis13_current
            plan22.adjoint()
            out22 = numpy.real((plan22.f_hat.copy().T).reshape(nfft_info22.xdim*nfft_info22.ydim))
            plan33.f =  weights_bs * vis3_current.conj() * vis12_current
            plan33.adjoint()
            out33 = numpy.real((plan33.f_hat.copy().T).reshape(nfft_info33.xdim*nfft_info33.ydim))

            plan1.f =  weights_bs * vis1_current * (numpy.abs(vis2_current)**2 + numpy.abs(vis3_current)**2)
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights_bs * vis2_current * (numpy.abs(vis1_current)**2 + numpy.abs(vis3_current)**2)
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights_bs * vis3_current * (numpy.abs(vis1_current)**2 + numpy.abs(vis2_current)**2)
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            R0 = -numpy.sum(weights_bs*numpy.real(resid_current))
            R1 = numpy.real(out11 + out22 + out33)
            R2 = numpy.real(out1 + out2 + out3)
            R = 6*(R0 + R1 + R2)

            # Now find S (4th order)
            plan12.f =  weights_bs * vis1_current*vis2_current.conj()
            plan12.adjoint()
            out12 = numpy.real((plan12.f_hat.copy().T).reshape(nfft_info12.xdim*nfft_info12.ydim))
            plan23.f =  weights_bs * vis2_current*vis3_current.conj()
            plan23.adjoint()
            out23 = numpy.real((plan23.f_hat.copy().T).reshape(nfft_info23.xdim*nfft_info23.ydim))
            plan31.f =  weights_bs * vis3_current*vis1_current.conj()
            plan31.adjoint()
            out31 = numpy.real((plan31.f_hat.copy().T).reshape(nfft_info31.xdim*nfft_info31.ydim))

            plan1.f =  weights_bs * vis23_current.conj()
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights_bs * vis13_current.conj()
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights_bs * vis12_current.conj()
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            S0 = numpy.sum(weights_bs*(numpy.abs(vis1_current)**2 + numpy.abs(vis2_current)**2 + numpy.abs(vis3_current)**2))
            S1 = 2  * (out12 + out23 + out31)
            S2 = 2 * (out1 + out2 + out3)
            S = 4*(S0 + S1 + S2)

            # T (5th order)
            plan1.f =  weights_bs * vis1_current
            plan1.adjoint()
            out1 = numpy.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))
            plan2.f =  weights_bs * vis2_current
            plan2.adjoint()
            out2 = numpy.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))
            plan3.f =  weights_bs * vis3_current
            plan3.adjoint()
            out3 = numpy.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

            T = 10*(out1 + out2 + out3)

            # Finally U (6th order)
            U = 6*weights_bs_norm * numpy.ones(T.shape)

            # Find Component based on minimizing chi^2
            deltas = numpy.zeros(len(A))
            chisq_map = numpy.zeros(len(A))
            for i in range(len(P)):
                if embed_mask[i]:
                    coeffs = numpy.array([A[i] + phaseweight*P[i],
                                      B[i] + phaseweight*Q[i],
                                      C[i] + phaseweight*R[i],
                                      D[i] + phaseweight*S[i],
                                      phaseweight*T[i],
                                      phaseweight*U[i]
                                     ])
                    allroots = polynomial.polyroots(coeffs)

                    # test roots to see which minimizes chi^2
                    newchisq = chisq_current
                    delta = 0
                    for j in range(len(allroots)):
                        root = allroots[j]
                        if numpy.imag(root)!=0: continue
                        if (no_neg_comps and root<0):continue
                        trialchisq = chisq_current + coeffs[0]*root + 0.5*coeffs[1]*root**2 + (1./3.)*coeffs[2]*root**3 + 0.25*coeffs[3]*root**4 + 0.2*coeffs[4]*root**5 + (1./6.)*coeffs[5]*root**6
                        if trialchisq < newchisq:
                            delta = root
                            newchisq = trialchisq

                else:
                    delta = 0.
                    newchisq = chisq_current
                deltas[i] = delta
                chisq_map[i] = newchisq

            #plot deltas and chi^2 map
            if show_updates:
                DeltasIm.imvec = deltas
                plot_i(DeltasIm, it, chisq_current,fig=0, cmap='afmhot')
                ChisqIm.imvec = -chisq_map
                plot_i(ChisqIm, it, chisq_current,fig=1, cmap='cool')

            #chisq_map = chisq_current + P*deltas + 0.5*Q*deltas**2 + (1./3.)*R*deltas**3 + 0.25*S*deltas**4 + 0.2*T*deltas**5 + (1./6.)*U*deltas**6
            component_loc_idx = numpy.argmin(chisq_map[embed_mask])
            component_strength = loop_gain*(deltas[embed_mask])[component_loc_idx]

        # clean component location
        component_loc_x = coord[component_loc_idx][0]
        component_loc_y = coord[component_loc_idx][1]

        # update imvec, vis, bispec
        imvec_current[component_loc_idx] += component_strength

        #TODO how to incorporate pulse function?
        vis_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv[:,0]*component_loc_x + uv[:,1]*component_loc_y))
        amp2_current = numpy.abs(vis_current)**2

        vis1_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv1[:,0]*component_loc_x + uv1[:,1]*component_loc_y))
        vis2_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv2[:,0]*component_loc_x + uv2[:,1]*component_loc_y))
        vis3_current += component_strength*numpy.exp(2*numpy.pi*1j*(uv3[:,0]*component_loc_x + uv3[:,1]*component_loc_y))
        bs_current = vis1_current * vis2_current * vis3_current

        # update chi^2 and output image
        chisq_amp2_current = numpy.sum(weights_amp2*(amp2 - amp2_current)**2)
        rchisq_amp2_current = numpy.sum(weights_amp2_nat*(amp2 - amp2_current)**2)

        chisq_bs_current =  numpy.sum(weights_bs*numpy.abs(bs - bs_current)**2)
        rchisq_bs_current =  numpy.sum(weights_bs_nat*numpy.abs(bs - bs_current)**2)

        chisq_current = chisq_amp2_current + phaseweight*chisq_bs_current

        print("it %i| %.4e (%.1f , %.1f) | %.4e %.4e | %.4e" % (it+1, component_strength, component_loc_x/const_def.RADPERUAS, component_loc_y/const_def.RADPERUAS, chisq_amp2_current, chisq_bs_current, chisq_current))

        OutputIm.imvec = imvec_current
        if show_updates:
            OutputIm.imvec = imager_utils.embed(OutputIm.imvec, embed_mask, clipfloor=0., randomfloor=False)
            OutputImBlur = OutputIm.blur_gauss(beamparams)
            plot_i(OutputImBlur, it, chisq_current, fig=2)

    return OutputIm
