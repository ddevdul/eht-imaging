"""
A interferometric telescope array class

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

import numpy
from ehtim.observing import obs_simulate  # as simobs
from ehtim.io import save, load # ehtim.io.load ehtim.io.save
from ehtim import const_def # as ehc
from ehtim.obsdata import Obsdata


class Array:

    """
    A VLBI array of telescopes with site locations, SEFDs, and other data.

       Attributes:
           tarr (numpy.recarray): The array of telescope data with datatype DTARR
           tkey (dict): A dictionary of rows in the tarr for each site name
           ephem (dict): A dictionary of TLEs for each space antenna,
                        Space antennas have x=y=z=0 in the tarr
    """

    # fix the types that go in the dictionary
    def __init__(self, tarr: numpy.recarray, ephem: dict[str, None]):
        self.tarr = tarr
        self.ephem = ephem
        self.tkey = self._check_ephemeris() 
    
    def _check_ephemeris(self) -> dict[str, None]:
        array_indicies = dict()
        for i, line in enumerate(self.tarr):
            if numpy.any(numpy.isnan([line['x'], line['y'], line['z']])):
                sitename = str(line['site'])
                try:
                    elen = len(self.ephem[sitename])
                except NameError:
                    raise Exception(f'No ephemeris for site {sitename} !')
                if elen != 3:
                    raise Exception('Wrong ephemeris format for site {sitename} !')
            array_indicies[self.tarr[i]['site']] = i

    def listbls(self) -> numpy.array[list]:
        """
        List all baselines.
        returns:
            numpy.array : array of baselines in a
        """
        sort_tarr = sorted(self.tarr["site"])
        bls = list()
        for i, _ in enumerate(sort_tarr):
            for j in range((i+1), len(sort_tarr)):
                if sort_tarr[i] != sort_tarr[j]:
                    if ([sort_tarr[i], sort_tarr[j]] not in bls and
                        [sort_tarr[j], sort_tarr[i]] not in bls):
                        bls.append([sort_tarr[i], sort_tarr[j]])
        return bls

    def obsdata(self, **kwargs) -> Obsdata:
        """
        Generate u,v points and baseline uncertainties.
        Required Keyword Arguements:
            ra (float): the source right ascension in fractional hours
            dec (float): the source declination in fractional degrees
            tint (float): the scan integration time in seconds
            tadv (float): the uniform cadence between scans in seconds
            tstart (float): the start time of the observation in hours
            tstop (float): the end time of the observation in hours
            mjd (int): the mjd of the observation
            timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
            polrep (str): polarization representation, either 'stokes' or 'circ'
            elevmin (float): station minimum elevation in degrees
            elevmax (float): station maximum elevation in degrees
            tau (float): the base opacity at all sites, or a dict giving one opacity per site
        returns:
            Obsdata: an observation object with no data
        """
        obsarr = obs_simulate.make_uvpoints(self, **kwargs)
        uniquetimes = numpy.sort(numpy.unique(obsarr['time']))
        scans = numpy.array([[time - 0.5 * kwargs["tadv"], time + 0.5 * kwargs["tadv"]] for time in uniquetimes])
        source = str(kwargs["ra"]) + ":" + str(kwargs["dec"])
        obs = Obsdata(obsarr, self.tarr, source=source,
                                    ampcal=True, phasecal=True, 
                                    opacitycal=True, dcal=True, frcal=True,
                                    scantable=scans, **kwargs)
        return obs

    def make_subarray(self, sites):
        """Make a subarray from the Array object array that only includes the sites listed.

           Args:
               sites (list) : list of sites in the subarray
           Returns:
               Array: an Array object with specified sites and metadata
        """
        all_sites = [t[0] for t in self.tarr]
        mask = np.array([t in sites for t in all_sites])
        subarr = Array(self.tarr[mask], ephem=self.ephem)
        return subarr

    def save_txt(self, fname):
        """Save the array data in a text file.

           Args:
               fname (str) : path to output array file
        """
        ehtim.io.save.save_array_txt(self, fname)
        return

##########################################################################
# Array creation functions
##########################################################################


def load_txt(fname, ephemdir='ephemeris'):
    """Read an array from a text file.
       Sites with x=y=z=0 are spacecraft, TLE ephemerides read from ephemdir.

       Args:
           fname (str) : path to input array file
           ephemdir (str) : path to directory with 2TLE ephemerides for spacecraft
       Returns:
           Array: an Array object loaded from file
    """

    return ehtim.io.load.load_array_txt(fname, ephemdir=ephemdir)
