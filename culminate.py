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
            'e_PA': 'e_PA'
        }
        catalog=catalog.drop_duplicates(subset="Star") # duplicates -> same method to generate Table 5, but keeps star names
        catalog = catalog[[c for c in cols if c in catalog.columns]]
        catalog = catalog.rename(columns={k: v for k, v in cols.items() if k in catalog.columns})
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
    mag_mask = catalog['Gmag'] <= args.magnitude if 'Gmag' in catalog.columns else np.ones(len(catalog), dtype=bool)
    catalog = catalog[mag_mask].copy()
    n_mag = len(catalog)
    
    # apply polarization mask
    pol_mask = catalog['Pol'] >= args.polarization if 'Pol' in catalog.columns else np.ones(len(catalog), dtype=bool)
    catalog = catalog[pol_mask].copy()
    n_pol = len(catalog)
    
    if len(catalog) == 0:
        print("No stars match magnitude/polarization criteria")
        return

    # culmination time (when star reaches meridian == hour angle is 0)
    star_coords = SkyCoord(
        ra=catalog['RA J2000'],
        dec=catalog['Dec J2000'],
        unit=u.deg,
        frame='icrs'
    )
    lst = observation_time.sidereal_time('apparent', longitude=location.lon)
    ra_hours = star_coords.ra.hour
    hour_angle = (lst.hour - ra_hours) % 24
    hour_angle_centered = np.where(hour_angle > 12, hour_angle - 24, hour_angle)
    seconds_to_culmination = -hour_angle_centered * 3600
    culmination_time = observation_time + (seconds_to_culmination * u.second)
    
    # if culmination is in the past, shift to next sidereal day (~23h 56m 4s)
    sidereal_day = 86164.0905 * u.second  # sidereal day in seconds
    culmination_time = np.where(culmination_time < observation_time, culmination_time + sidereal_day, culmination_time)
    
    # calculate el at culmination
    frame = AltAz(obstime=culmination_time, location=location)
    star_altaz = star_coords.transform_to(frame)
    elevations = star_altaz.alt.deg
        
    # moon angle
    moon = get_body('moon', observation_time, location)
    moon_distance = star_coords.separation(moon.icrs)
    
    # dark time - find next sunset and sunrise
    times = observation_time + np.linspace(0, 1.5, 2160) * u.day
    sun = get_body('sun', times, location)
    frame = AltAz(obstime=times, location=location)
    sun_alt = sun.transform_to(frame).alt.deg
    
    sunset_idx = np.where(sun_alt < 0)[0]
    sunset = times[sunset_idx[0]] if len(sunset_idx) > 0 else times[-1]
    
    if len(sunset_idx) > 0:
        after_sunset = np.where(sun_alt[sunset_idx[0]:] > 0)[0]
        sunrise = times[sunset_idx[0] + after_sunset[0]] if len(after_sunset) > 0 else times[-1]
    else:
        sunrise = times[-1]

    # visibility masks
    elevation_mask = elevations >= args.elevation
    catalog = catalog[elevation_mask].reset_index(drop=True)
    elevations = elevations[elevation_mask]
    moon_distance = moon_distance[elevation_mask]
    culmination_time = culmination_time[elevation_mask]
    n_elev = len(catalog)
    
    dark_time_mask = (culmination_time >= sunset) & (culmination_time <= sunrise)
    catalog = catalog[dark_time_mask].reset_index(drop=True)
    elevations = elevations[dark_time_mask]
    moon_distance = moon_distance[dark_time_mask]
    culmination_time = culmination_time[dark_time_mask]
    n_dark = len(catalog)
    
    # apply mask
    visible_stars = catalog.copy()
    
    # add columns
    visible_stars["Elevation"] = np.round(elevations, 1)
    visible_stars["Moon Angle"] = np.round(moon_distance.deg, 1)
    visible_stars["Culmination Time (UT)"] = [t.iso for t in culmination_time]

    # sort
    sort_map = {
        'time': 'Culmination Time (UT)',
        'G': 'G',
        'RA': 'RA J2000',
        'Dec': 'Dec J2000',
        'p': 'Pol',
        'PA': 'PA',
        'El': 'Elevation',
        'Moon': 'Moon Angle'
    }
    sort_col = sort_map.get(args.sort, 'Culmination Time (UT)')
    visible_stars = visible_stars.sort_values(by=[sort_col])
    visible_stars["Culmination Time (UT)"] = visible_stars["Culmination Time (UT)"].apply(lambda t: t.split()[1].split('.')[0])
    
    # print
    print(visible_stars.to_string(index=False))
    print()
    print(f"Observing date (UT): {observation_time}")
    print(f"Location: {args.location}")
    print(f"Sunset: {sunset.iso}")
    print(f"Sunrise: {sunrise.iso}")
    
    if args.verbose:
        print()
        print(f"Total catalog: {n_initial} stars")
        print(f"Magnitude mask (Gmag <= {args.magnitude}): {n_mag} stars")
        print(f"Polarization mask (Pol >= {args.polarization}): {n_pol} stars")
        print(f"Elevation mask (>= {args.elevation}°): {n_elev} stars")
        print(f"Dark time mask: {n_dark} stars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Culminate - calculate star culmination times.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose debug output.")
    parser.add_argument("--elevation", "-e", type=float, default=60, metavar="DEGREES", help="Minimum culminating elevation.")
    parser.add_argument("--time", "-t", type=str, default=None, metavar="ISO (UT)", help="Observation time (UT) in ISO format, e.g. \"2026-04-14 12:00:00\" (default: now).")
    parser.add_argument("--location", "-l", type=str, default="WAO", choices=["WAO", "Bochum"], help="Location. WAO (default) or Bochum.")
    parser.add_argument("--magnitude", "-m", type=float, default=8, metavar="G-BAND", help="Limiting magnitude.")
    parser.add_argument("--polarization", "-p", type=float, default=1e-2, metavar="FRACTION", help="Minimum polarization fraction.")
    parser.add_argument("--sort", "-s", type=str, default="RA", choices=["G", "RA", "Dec", "Pol", "PA", "El", "Moon"], help="Column to sort min to max.")
    args = parser.parse_args()
    
    main(args)

