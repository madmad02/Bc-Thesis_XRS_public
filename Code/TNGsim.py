# %%
# ------- Import -------
from soxs.utils import soxs_cfg
soxs_cfg.set("soxs", "bkgnd_nH", "0.01") # avoid configparser error by specifying here
import soxs
print("soxs version:", soxs.__version__)

import yt
import pyxsim
print("pyxsim version:", pyxsim.__version__)

import h5py
import numpy as np
import illustris_python as il

import os
from regions import RectangleSkyRegion
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import wcs
from astropy.io import fits
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.constants import m_p

# %%
# ------- Load data -------
basePath = "/space/IllustrisTNG/TNG100-3/output/"
snap = 99
haloID = 0
# %%
halo = il.groupcat.loadSingle(basePath, snap, haloID=haloID)
# %%
groupcat = il.groupcat.loadHalos(basePath, snap)
# %%
snapshot = il.snapshot.loadSubset(basePath, snap, "gas", fields=['Coordinates','GFM_CoolingRate','Density','InternalEnergy','ElectronAbundance','StarFormationRate'])
# %%
fields = ['Coordinates','GFM_CoolingRate','Density','InternalEnergy','ElectronAbundance','StarFormationRate']
# gas = il.snapshot.loadSubset(bas'ePath, snap, 'gas')
gas = il.snapshot.loadHalo(basePath, snap, haloID, "gas")
header = il.groupcat.loadHeader(basePath, snap)

with h5py.File(il.snapshot.snapPath(basePath, snap),'r') as f:
    header_snap = dict(f['Header'].attrs)
# %%
# ------- yt settings -------
filename = "halo_%d.hdf5" % haloID
# %%
with h5py.File(filename,'w') as f:
    for key in gas.keys():
        f['PartType0/' + key] = gas[key]
        
    # some metadata that yt demands
    f.create_group('Header')
    f['Header'].attrs['NumFilesPerSnapshot'] = 1
    f['Header'].attrs['MassTable'] = header_snap['MassTable']
    f['Header'].attrs['BoxSize'] = header['BoxSize']
    f['Header'].attrs['Time'] = header['Time']
    f['Header'].attrs['NumPart_ThisFile'] = np.array([gas['count'],0,0,0,0,0])
    
    # Must have the next six for correct units
    f["Header"].attrs["HubbleParam"] = header["HubbleParam"]
    f["Header"].attrs["Omega0"] = header["Omega0"]
    f["Header"].attrs["OmegaLambda"] = header["OmegaLambda"]

    # These correspond to the values from the TNG simulations
    f["Header"].attrs["UnitLength_in_cm"] = header_snap['UnitLength_in_cm']
    f["Header"].attrs["UnitMass_in_g"] = header_snap['UnitMass_in_g']
    f["Header"].attrs["UnitVelocity_in_cm_per_s"] = header_snap['UnitVelocity_in_cm_per_s']
# %%
# ------- Load yt file -------
# ds = yt.load(filename)
ds = yt.load(filename, default_species_fields="ionized")
# %%
# ------- Add X-ray filter -------
def hot_gas(pfilter, data):
    pfilter1 = data[pfilter.filtered_type, "temperature"] > 3.0e5
    pfilter2 = data["PartType0", "StarFormationRate"] == 0.0
    pfilter3 = data["PartType0", "GFM_CoolingRate"] < 0.0
    return (pfilter1 & pfilter2) & pfilter3

yt.add_particle_filter("hot_gas", function=hot_gas,
                       filtered_type='gas', requires=["temperature","density"])
# %%
ds.add_particle_filter("hot_gas")
# %%
# ------- Projection and visualization -------
c = ds.arr([halo["GroupPos"][0], halo["GroupPos"][1], halo["GroupPos"][2]], "code_length")
# %%
###### SHIFT SHIFT SHIFT?
# Define shift values
# dx, dy, dz = 0.0, 0.0, 0.0
# dx, dy, dz = 0.0, -7600.0, -400.0 # halo 0 first pos
# dx, dy, dz = 0.0, 0.0, 185.0 # halo 20 r500
# dx, dy, dz = 0.0, 0.0, 230.0 # halo 20 in btw
# dx, dy, dz = 0.0, 0.0, 295.0 # halo 20 r200
# dx, dy, dz = 0.0, 0.0, 375.0 # halo 20 outskirts
# dx, dy, dz = 0.0, 0.0, 685.0 # halo 20 gas e-5
# dx, dy, dz = 0.0, 0.0, 800.0 # halo 20 thin
# dx, dy, dz = 0.0, 0.0, -2000.0 # halo 0 filament
dx, dy, dz = 0.0, 0.0, 600.0 # halo 0 beyond r200
# dx, dy, dz = 0.0, 0.0, 470.0 # halo 0 r200
# dx, dy, dz = 0.0, 0.0, 295.0 # halo 0 r500
# Shift the center of projection
c_shifted = ds.arr([halo["GroupPos"][0] + dx, halo["GroupPos"][1] + dy, halo["GroupPos"][2] + dz], "code_length")
# c = c_shifted
###### SHIFT SHIFT SHIFT?
# %%
# Total gas density
prj = yt.ProjectionPlot(ds, "x", ("gas","density"), width=(5, "Mpc"), center=c)
prj.set_zlim(("gas","density"), 1.0e-6, 1.0)
prj.show()
# %%
sphere_center = c
r0_pt = ds.quan(10, 'kpc')
r200_sp = ds.quan(696.5179446533203, 'kpc')
r500_sp = ds.quan(436.4346303222656, 'kpc')
prj = yt.ProjectionPlot(ds, "x", ("gas","density"), width=(7, "Mpc"), center=c_shifted)
prj.annotate_sphere(c_shifted, radius=r0_pt, circle_args={'color':'black', 'linestyle':'solid'})
prj.annotate_sphere(sphere_center, radius=r200_sp, circle_args={'color':'red', 'linestyle':'dashed'})
prj.annotate_sphere(sphere_center, radius=r500_sp, circle_args={'color':'black', 'linestyle':'dashed'})
# prj.set_zlim(("gas","density"), 1.0e-8, 1.0e1)
prj.set_zlim(("gas","density"), 1.0e-8, 1.0e-2)
prj.show()
# %%
sphere_center = c
r0_pt = ds.quan(10, 'kpc')
r200_sp = ds.quan(696.517944653320, 'kpc')
r500_sp = ds.quan(436.4346303222656, 'kpc')
slc = yt.SlicePlot(ds, "x", ("gas","temperature"), width=(7, "Mpc"), center=c_shifted)
slc.annotate_sphere(c_shifted, radius=r0_pt, circle_args={'color':'black', 'linestyle':'solid'})
slc.annotate_sphere(sphere_center, radius=r200_sp, circle_args={'color':'red', 'linestyle':'dashed'})
slc.annotate_sphere(sphere_center, radius=r500_sp, circle_args={'color':'black', 'linestyle':'dashed'})
slc.set_zlim(("gas","temperature"), 1.0e2, 1.0e8)
slc.show()
# %%
# Off-axis projection
prj = yt.OffAxisProjectionPlot(ds, [1,1,1], ("gas","density"), width=(10, "Mpc"), center=c)
prj.set_zlim(("gas","density"), 1.0e-5, 1.0)
prj.show()
# %%
# Hot gas density
sphere_center = c
r0_pt = ds.quan(10, 'kpc')
r200_sp = ds.quan(696.5179446533203 , 'kpc')
r500_sp = ds.quan(436.4346303222656 , 'kpc')
prj = yt.ProjectionPlot(ds, "x", ("hot_gas","density"), width=(2, "Mpc"), center=c_shifted)
prj.annotate_sphere(c_shifted, radius=r0_pt, circle_args={'color':'black', 'linestyle':'solid'})
# prj.annotate_sphere(sphere_center, radius=r200_sp, circle_args={'color':'red', 'linestyle':'dashed'})
# prj.annotate_sphere(sphere_center, radius=r500_sp, circle_args={'color':'black', 'linestyle':'dashed'})
# prj.set_zlim(("gas","density"), 1.0e-8, 1.0e1)
prj.set_zlim(("hot_gas","density"), 1.0e-6, 1.0e-2)
prj.show()
# %%
prj.save(f"prj_h{haloID}_filament.png")
# %%
slc.save(f"slc_h{haloID}_filament.png")
# %%
# ------- Mock observation - configuration -------
emin = 0.2
emax = 12.0
nbins = 4000
source_model = pyxsim.CIESourceModel(
    "apec", emin, emax, nbins, ("hot_gas","metallicity"),
    temperature_field=("hot_gas","temperature"),
    emission_measure_field=("hot_gas", "emission_measure"),
)
# %%
exp_time = (5000, "ks") # exposure time
area = (3000.0, "cm**2") # collecting area
redshift = 0.01
# %%
width = ds.quan(0.15, "Mpc")
le = c_shifted - 0.5*width
re = c_shifted + 0.5*width
box = ds.box(le, re)
# %%
# Nice plots
plt.scatter(box[('hot_gas', 'temperature')]*8.617333262145e-8, (box[('PartType0', 'density')])*((2e+40)/1.67e-27)/(((3.09e+19)**3)*(0.6774**2)*100**3), c=box[('hot_gas', 'metallicity')]/0.0127, marker="o", s=10)
plt.xlabel("kT (keV)")
plt.ylabel("n$ _{ \\text{e}} \\times 10^{-2}$ (cm$^{-3}$)")
plt.xscale("log")
plt.yscale("log")
plt.yticks([1.5e-2,2e-2,3e-2,4e-2],[1.5,2.0,3.0,4.0])
plt.xticks([2.5, 3,4,5,6,7],[2.5, 3,4,5,6,7])
cb = plt.colorbar()
cb.set_label('Z (Z$_{\odot}$)')
plt.savefig("file.pdf", bbox_inches="tight")
# %%
# ------- Create and project photons -------
n_photons, n_cells = pyxsim.make_photons(f"halo_{haloID}_photons", box, redshift, area, exp_time, source_model)
# %%
n_events = pyxsim.project_photons(f"halo_{haloID}_photons", f"halo_{haloID}_events", "x", (45.,30.),  # projection angle i guess
                                  absorb_model="wabs", nH=0.01)
# %%
# ROTATION ALONG PROJECTION AXIS
original_axis = np.array([1, 0, 0])
rotation_angle_x = 30
rotation_angle_y = 20
rotation_angle_z = 0

r_x = R.from_rotvec(np.radians(rotation_angle_x) * np.array([1, 0, 0]))  # rotation around "x" axis
r_y = R.from_rotvec(np.radians(rotation_angle_y) * np.array([0, 1, 0]))  # rotation around "y" axis
r_z = R.from_rotvec(np.radians(rotation_angle_z) * np.array([0, 0, 1]))  # rotation around "z" axis

rotated_axis = r_y.apply(r_x.apply(original_axis))

n_events = pyxsim.project_photons(f"halo_{haloID}_photons", f"halo_{haloID}_events", rotated_axis, (45.,30.),
                                  absorb_model="wabs", nH=0.01)
# %%
events = pyxsim.EventList(f"halo_{haloID}_events.h5")
events.write_to_simput(f"halo_{haloID}", overwrite=True)
# %%
# ------- Create mock image with SOXS ------
exposure = (100.0, "ks") # exposure time of the observation

# instrument = "xrism_xtend"            # XRISM imager
# instrument = "athena_wfi"             # Athena imager
# instrument = "xrism_resolve"          # the XRISM we use
# instrument = "NewAthena"              # the Athena we use
# instrument = "chandra_aciss_cy22"     # the Chandra we use
# instrument = "lem_inner_array"        # the LEM we use
# instrument = "hubs"                   # the HUBS we use
# %%
# ------- Prepare image and spectrum -------
soxs.instrument_simulator(f"halo_{haloID}_simput.fits", f"evts_{instrument}_expks:{exposure[0]}.fits", exposure, instrument, (45,30.), overwrite=True, foreground=True, ptsrc_bkgnd=True, instr_bkgnd=True)
# %%
soxs.write_image(f"evts_{instrument}_expks:{exposure[0]}.fits", f"img_{instrument}_expks:{exposure[0]}.fits", emin=0.1, emax=12.0, overwrite=True)
center_sky = SkyCoord(45, 30, unit='deg', frame='fk5')
region_sky = RectangleSkyRegion(center=center_sky, width=32* u.arcmin, height=32*u.arcmin)
with fits.open(f"img_{instrument}_expks:{exposure[0]}.fits") as f:
    w = wcs.WCS(header=f[0].header)
    fig, ax = soxs.plot_image(f"img_{instrument}_expks:{exposure[0]}.fits", stretch='log', cmap='afmhot', vmax=1500.0, width=0.1)
ax.add_artist(region_sky.to_pixel(w).as_artist())
# %%
soxs.write_spectrum(f"evts_{instrument}_expks:{exposure[0]}.fits", f"spectrum_{instrument}_expks:{exposure[0]}.pi", overwrite=True)
fig, ax = soxs.plot_spectrum(f"spectrum_{instrument}_expks:{exposure[0]}.pi", xmin=0.3, xmax=10.0, xscale="log", yscale="log")
# %%
abundance = np.average(box[('hot_gas', 'metallicity')])/0.0127
abundance_err = np.std(box[('hot_gas', 'metallicity')])/0.0127
print("Z :: Abundance :", abundance.value, "pm", abundance_err.value, "Z_solar")
# %%
temperature = np.average(box[('hot_gas', 'temperature')], weights=box[('hot_gas', 'density')])
average = np.average(box[('hot_gas', 'temperature')], weights=box[('hot_gas', 'density')])
variance = np.average((box[('hot_gas', 'temperature')]-average)**2, weights=box[('hot_gas', 'density')])
weighted_std = np.sqrt(variance)*8.617333262145e-8
tempkev = temperature*8.617333262145e-8
print("Temperature in keV :", tempkev.value, "pm", weighted_std.value, "keV")
# %%
density = np.average(box[('PartType0', 'Density')])*((2e+40)/1.67e-27)/(((3.09e+19)**3)*(0.6774**2)*100**3)
density_err = np.std(box[('PartType0', 'Density')])*((2e+40)/1.67e-27)/(((3.09e+19)**3)*(0.6774**2)*100**3)
print("mean density :", density.value, "pm", density_err.value, "particles/cm3")
# halo_mass = groupcat['Group_M_Crit200'][haloID]*0.6774*1e10
# print("halo_mass :", halo_mass, "M_solar")
r200 = halo["Group_R_Crit200"]*0.6774
print("r200 :", r200, "kpc")
r500 = halo["Group_R_Crit500"]*0.6774
print("r500 :", r500, "kpc")
# %%