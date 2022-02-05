import sys
sys.path.extend(["../ehtim"])
import ehtim # as eh

def main():
    # Load the image and the array
    im = ehtim.image.load_txt('../models/avery_sgra_eofn.txt')
    eht = ehtim.array.load_txt('../arrays/EHT2017.txt')
    # Observe the image
    tint_sec = 30
    tadv_sec = 1200*3
    tstart_hr = 0
    tstop_hr = 24
    bw_hz = 4e9
    obs = im.observe(
        eht, tint_sec, tadv_sec, tstart_hr,
        tstop_hr, bw_hz, sgrscat=False,
        ampcal=False, phasecal=False
        )
    # Test without multiprocessing
    obs_nc = ehtim.calibrating.network_cal.network_cal(
        obs, im.total_flux(), processes=-1,msgtype='bh'
        )
    obs_sc = ehtim.calibrating.self_cal.self_cal(obs, im, processes=-1)
    ehtim.comp_plots.plot_bl_obs_im_compare(
        [obs,obs_nc,obs_sc],im,'SMA','ALMA','amp'
        )
    ehtim.comp_plots.plotall_obs_im_compare(
        [obs,obs_nc,obs_sc],im,'uvdist','amp'
        )
    ehtim.comp_plots.plotall_obs_im_compare(
        [obs,obs_nc,obs_sc],im,'uvdist','phase'
        )
    # Test with multiprocessing
    obs_nc = ehtim.calibrating.network_cal.network_cal(
        obs, im.total_flux(), processes=0,msgtype='bh'
        )
    obs_sc = ehtim.calibrating.self_cal.self_cal(obs, im, processes=0)
    ehtim.comp_plots.plot_bl_obs_im_compare([obs,obs_nc,obs_sc],im,'SMA','ALMA','amp')
    ehtim.comp_plots.plotall_obs_im_compare([obs,obs_nc,obs_sc],im,'uvdist','amp')
    ehtim.comp_plots.plotall_obs_im_compare([obs,obs_nc,obs_sc],im,'uvdist','phase')

if __name__ == "__main__":
    main()