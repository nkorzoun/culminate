#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body
from astropy.time import Time


def main(args):
    # location
    if args.location == "WAO":
        location = EarthLocation(lat=30.052984*u.deg, lon=35.040677*u.deg, height=400*u.m)
    elif args.location == "Bochum":
        location = EarthLocation(lat=51.44192*u.deg, lon=7.26275*u.deg, height=70*u.m) # this is a complete guess

    # load catalog
    # Compilation of optical starlight polarization
    # (Panopoulou+, 2025)
    # http://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/276/15
    try:
        hdul = fits.open('asu.fit')
        catalog = pd.DataFrame(hdul[1].data)
        cols = {
            'Name': 'Name',
            'Gmag': 'G',
            'RAJ2000': 'RA J2000',
            'DEJ2000': 'Dec J2000',
            'Pol': 'Pol',
            'e_Pol': 'e_Pol',
            'PA': 'PA',
            'e_PA': 'e_PA',
            'N': 'N Obs'
        }
        catalog['N'] = catalog.groupby('Star')['Star'].transform('size') # number of observations before dedup
        catalog=catalog.drop_duplicates(subset="Star") # duplicates -> same method to generate Table 5, but keeps star names
        catalog = catalog[[c for c in cols if c in catalog.columns]]
        catalog = catalog.rename(columns={k: v for k, v in cols.items() if k in catalog.columns})
        if 'G' in catalog.columns:
            catalog.loc[catalog['G'] == 0, 'G'] = np.nan
    except FileNotFoundError:
        print("Error: asu.fit not found")
        return
    except Exception as e:
        print(f"Error reading catalog: {e}")
        return    
    
    n_initial = len(catalog)
    
    # observation time
    observation_time = Time.now() if args.time is None else Time(args.time, format='iso', scale='utc')
    
    # apply magnitude mask
    mag_mask = (catalog['G'].isna() | (catalog['G'] <= args.magnitude)) if 'G' in catalog.columns else np.ones(len(catalog), dtype=bool)
    catalog = catalog[mag_mask].copy()
    n_mag = len(catalog)
    
    # apply polarization mask
    pol_mask = catalog['Pol'] >= args.min_polarization if 'Pol' in catalog.columns else np.ones(len(catalog), dtype=bool)
    if args.max_polarization is not None and 'Pol' in catalog.columns:
        pol_mask &= catalog['Pol'] <= args.max_polarization
    catalog = catalog[pol_mask].copy()
    n_pol = len(catalog)

    # apply observation count mask
    obs_mask = catalog['N Obs'] >= args.observations if 'N Obs' in catalog.columns else np.ones(len(catalog), dtype=bool)
    catalog = catalog[obs_mask].copy()
    n_obs = len(catalog)

    if len(catalog) == 0:
        print("No stars match magnitude/polarization/observation criteria")
        return
    
    # dark time - find the sunset/sunrise bracketing the observation time
    # (search starts a day before observation_time so an already-dark observation time
    # still resolves to the sunset that started the night, not the observation time itself)
    times = observation_time - 1 * u.day + np.linspace(0, 2.5, 3600) * u.day
    sun = get_body('sun', times, location)
    frame = AltAz(obstime=times, location=location)
    sun_alt = sun.transform_to(frame).alt.deg

    below_horizon = sun_alt < 0
    sunset_idxs = np.where(np.diff(below_horizon.astype(int)) == 1)[0] + 1
    sunrise_idxs = np.where(np.diff(below_horizon.astype(int)) == -1)[0] + 1

    obs_sun_alt = get_body('sun', observation_time, location).transform_to(
        AltAz(obstime=observation_time, location=location)).alt.deg
    currently_night = obs_sun_alt < 0

    if currently_night:
        past_sunsets = sunset_idxs[times[sunset_idxs].jd <= observation_time.jd]
        sunset_idx = past_sunsets[-1] if len(past_sunsets) > 0 else (sunset_idxs[0] if len(sunset_idxs) > 0 else len(times) - 1)
    else:
        future_sunsets = sunset_idxs[times[sunset_idxs].jd > observation_time.jd]
        sunset_idx = future_sunsets[0] if len(future_sunsets) > 0 else (sunset_idxs[-1] if len(sunset_idxs) > 0 else len(times) - 1)
    sunset = times[sunset_idx]

    future_sunrises = sunrise_idxs[times[sunrise_idxs].jd > sunset.jd]
    sunrise = times[future_sunrises[0]] if len(future_sunrises) > 0 else times[-1]

    # find rise/peak/set times
    night_times = sunset + np.linspace(0, 1, 1000) * (sunrise - sunset)
    
    peak_elevations = np.zeros(len(catalog))
    rise_times = np.zeros(len(catalog), dtype=object)
    peak_times = np.zeros(len(catalog), dtype=object)
    set_times = np.zeros(len(catalog), dtype=object)
    
    # calculate elevations for all stars at all times
    all_elevations = np.zeros((len(catalog), len(night_times)))
    all_stars = SkyCoord(ra=catalog['RA J2000'].values*u.deg, dec=catalog['Dec J2000'].values*u.deg, frame='icrs')
    
    for j, night_time in enumerate(night_times):
        frame = AltAz(obstime=night_time, location=location)
        altaz = all_stars.transform_to(frame)
        all_elevations[:, j] = altaz.alt.deg
    
    # find peak, rise, and set for each star
    for i in range(len(catalog)):
        elevations = all_elevations[i, :]
        
        # peak
        peak_idx = np.argmax(elevations)
        peak_elevations[i] = elevations[peak_idx]
        peak_times[i] = night_times[peak_idx]
        
        # rise and set
        above_threshold = elevations >= args.elevation
        if np.any(above_threshold):
            rise_idx = np.argmax(above_threshold)
            set_idx = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])
            rise_times[i] = night_times[rise_idx]
            set_times[i] = night_times[set_idx]
    
    # masks - only keep stars that reach the elevation threshold
    elevation_mask = peak_elevations >= args.elevation
    catalog = catalog[elevation_mask].reset_index(drop=True)
    peak_elevations = peak_elevations[elevation_mask]
    rise_times = rise_times[elevation_mask]
    peak_times = peak_times[elevation_mask]
    set_times = set_times[elevation_mask]
    n_elev = len(catalog)
    n_dark = len(catalog)
    
    # calculate moon distance at peak time for each star
    moon_distances = np.zeros(len(catalog))
    for i, peak_time in enumerate(peak_times):
        moon = get_body('moon', peak_time, location)
        star = SkyCoord(ra=catalog['RA J2000'].iloc[i]*u.deg, dec=catalog['Dec J2000'].iloc[i]*u.deg, frame='icrs')
        moon_distances[i] = star.separation(moon.icrs).deg
    
    # add columns
    catalog["Start Time (UT)"] = [t.iso.split('.')[0][:-3] if t is not None else "N/A" for t in rise_times]
    catalog["Peak Time (UT)"] = [t.iso.split('.')[0][:-3] for t in peak_times]
    catalog["Elevation"] = np.round(peak_elevations, 1)
    catalog["Moon Angle"] = np.round(moon_distances, 1)
    catalog["End Time (UT)"] = [t.iso.split('.')[0][:-3] if t is not None else "N/A" for t in set_times]

    # sort
    sort_map = {
        'start': 'Start Time (UT)',
        'peak': 'Peak Time (UT)',
        'end': 'End Time (UT)',
        'G': 'G',
        'RA': 'RA J2000',
        'Dec': 'Dec J2000',
        'p': 'Pol',
        'PA': 'PA',
        'El': 'Elevation',
        'Moon': 'Moon Angle',
        'N': 'N Obs'
    }
    sort_col = sort_map.get(args.sort, 'Peak Time (UT)')
    catalog = catalog.sort_values(by=[sort_col])
    
    # print
    print(catalog.to_string(index=False, na_rep='--'))
    print()
    print(f"Observing date (UT): {observation_time}")
    print(f"Location: {args.location}")
    print(f"Sunset: {sunset.iso}")
    print(f"Sunrise: {sunrise.iso}")

    if args.csv:
        catalog.to_csv(args.csv, index=False)
        print(f"Saved table to {args.csv}")

    if args.verbose:
        print()
        print(f"Total catalog: {n_initial} stars")
        print(f"Magnitude mask (Gmag <= {args.magnitude}): {n_mag} stars")
        pol_range = f"{args.min_polarization} <= Pol" + (f" <= {args.max_polarization}" if args.max_polarization is not None else "")
        print(f"Polarization mask ({pol_range}): {n_pol} stars")
        print(f"Observation count mask (N Obs >= {args.observations}): {n_obs} stars")
        print(f"Elevation mask (>= {args.elevation}°): {n_elev} stars")
        print(f"Dark time mask: {n_dark} stars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Culminate - find observable stars by peak elevation.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose debug output.")
    parser.add_argument("--elevation", "-e", type=float, default=60, metavar="DEGREES", help="Minimum peak elevation.")
    parser.add_argument("--time", "-t", type=str, default=None, metavar="ISO (UT)", help="Observation time (UT) in ISO format, e.g. \"2026-04-14 12:00:00\" (default: now).")
    parser.add_argument("--location", "-l", type=str, default="WAO", choices=["WAO", "Bochum"], help="Location. WAO (default) or Bochum.")
    parser.add_argument("--magnitude", "-m", type=float, default=8, metavar="G-BAND", help="Limiting magnitude.")
    parser.add_argument("--min-polarization", "-pmin", type=float, default=1e-2, metavar="FRACTION", help="Minimum polarization fraction.")
    parser.add_argument("--max-polarization", "-pmax", type=float, default=None, metavar="FRACTION", help="Maximum polarization fraction.")
    parser.add_argument("--observations", "-n", type=int, default=1, metavar="COUNT", help="Minimum number of data points (observations) for a star.")
    parser.add_argument("--sort", "-s", type=str, default="RA", choices=["start", "peak", "end", "G", "RA", "Dec", "Pol", "PA", "El", "Moon", "N"], help="Column to sort min to max.")
    parser.add_argument("--csv", type=str, default=None, metavar="FILE", help="Export the table to a CSV file.")
    args = parser.parse_args()
    
    main(args)

