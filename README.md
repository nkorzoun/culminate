# culminate.py

A tool that calculates and displays visible stars with their polarization properties and culmination times.
Inspired by software written by Henrike Fleischhack for the VERITAS observatory.

# Installation

Install dependencies (`python`, `numpy`, `pandas`, `astropy`):

**Option 1: Using conda**
```
conda create -n culminate python=3.10 numpy pandas astropy -y
conda activate culminate
```

**Option 2: Using pip**
```
pip install numpy pandas astropy
```

# Running the script

Make the script executable and run it:

```bash
chmod +x culminate.py
./culminate.py
```

For available arguments:

```bash
./culminate.py --help
```

# Data

The `asu.fit` catalog is sourced from Table 4 (extended catalog) of:

[*A Compilation of Optical Starlight Polarization Catalogs*](https://iopscience.iop.org/article/10.3847/1538-4365/ad8b21) (Panopoulou et al. 2025)

Also available on [ViZieR](http://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/276/15)