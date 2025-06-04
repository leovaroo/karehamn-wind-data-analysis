import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Open dataset and extract wind data
nc_file = xr.open_dataset('cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_1743095861588.nc')
eastward_wind = nc_file['eastward_wind']
northward_wind = nc_file['northward_wind']
time = np.array(nc_file['time'].values, dtype='datetime64')

# Create a directory to save the plots
output_dir = "wind_data_output_images"
os.makedirs(output_dir, exist_ok=True)

####################################################################################
# DIRECT USE OF DATA AT KAREHAMN LOCATION
####################################################################################

# Calculate wind speed at 10 meters
wind_speed = np.sqrt(eastward_wind**2 + northward_wind**2)
wind_speed_flat = wind_speed.values.flatten()
wind_speed_flat = wind_speed_flat[~np.isnan(wind_speed_flat)]

# Normal wind profile model (NWP) - Power law for wind speed at hub height (130 meters)
def adjust_wind_speed_to_hub(wind_speed_10m, alpha=0.2, z_hub=130, z_ref=10):
    return wind_speed_10m * (z_hub / z_ref)**alpha

# Calculate wind speed at 30 meters
wind_speed_hub = adjust_wind_speed_to_hub(wind_speed)
wind_speed_hub_flat = wind_speed_hub.values.flatten()
wind_speed_hub_flat = wind_speed_hub_flat[~np.isnan(wind_speed_hub_flat)]

# Time intervals (delta_t) and observations (N)
if len(time) > 1:
    total_observation_time_hours = (time[-1] - time[0]) / np.timedelta64(1, 'h')
    delta_t = np.diff(time) / np.timedelta64(1, 's')
    unique_deltas, counts = np.unique(delta_t, return_counts=True)
    N = len(time) if len(time) > 1 else len(wind_speed_flat)
else:
    total_observation_time_hours = len(time)
    delta_t = np.array([])
print(f"Total number of observations: N = {N}")
if len(delta_t) > 0:
    for dt, count in zip(unique_deltas, counts):
        print(f"Time interval: Δt = {dt:.2f}")
# Convert total observation time to months
total_observation_time_months = total_observation_time_hours / (24 * 30.44)
print(f"Total observation time: {total_observation_time_hours:.2f} hours = {total_observation_time_months:.2f} months")

# Plot: Average Wind Speed Over Space at 10m 
average_wind_speed_time = wind_speed.mean(dim='time')
fig, ax = plt.subplots(figsize=(10, 6))
mappable = average_wind_speed_time.plot(ax=ax, cmap='viridis', add_colorbar=False)
plt.colorbar(mappable, label='Wind speed (m/s)')
plt.title('Average wind speed at 10m above sea level')
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
# Kårehamn location red rectangle
lat_min, lat_max = 56.933333, 57.022222
lon_min, lon_max = 16.977778, 17.066667
rect_width = lon_max - lon_min
rect_height = lat_max - lat_min
rect = Rectangle((lon_min, lat_min), rect_width, rect_height, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
legend_elements = [Line2D([0], [0], color='red', lw=2, label='Kårehamn location area')]
ax.legend(handles=legend_elements, loc='upper right')
# Save and display plot
plt.savefig(os.path.join(output_dir, "average_wind_speed_10m.png"), dpi=1200)
plt.show(block=False)

# Plot: Average Wind Speed at Over Space at 130m (hub height)
wind_speed_hub_time = wind_speed_hub.mean(dim='time')
fig, ax = plt.subplots(figsize=(10, 6))
mappable = wind_speed_hub_time.plot(ax=ax, cmap='plasma', add_colorbar=False)
plt.colorbar(mappable, label='Wind speed (m/s)')
plt.title('Average wind speed at 130m above sea level')
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
# Kårehamn location red rectangle
rect = Rectangle((lon_min, lat_min), rect_width, rect_height, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.legend([Line2D([0], [0], color='red', lw=2)], ['Kårehamn location'], loc='upper right')
# Save and display plot
plt.savefig(os.path.join(output_dir, "adjusted_wind_speed_130m.png"), dpi=1200)
plt.show(block=False)

# Histogram of wind speed occurrences at 10m above sea level
# Define bin edges from 0 to the max wind speed, ensuring whole number intervals
max_speed = np.ceil(np.max(wind_speed_flat))
bins = np.arange(0, max_speed + 1, 1)  # Bins: 0-1, 1-2, 2-3, ...
# Compute histogram
counts, bin_edges = np.histogram(wind_speed_flat, bins=bins)
# Convert frequency to percentage
percentages = (counts / N) * 100
# Plot histogram as a step profile
plt.figure(figsize=(10, 6))
plt.step(bin_edges[:-1], percentages, where='post', color='blue', linewidth=2)
# Labels and formatting
plt.title('Histogram of wind speed occurrences at 10m above sea level')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency (% of total)')
plt.xticks(bins)  # Ensure x-axis labels match bin edges
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Save and display the plot
plt.savefig(os.path.join(output_dir, "wind_speed_histogram_10m.png"), dpi=1200)
plt.show(block=False)

# Histogram of wind speed occurrences at 130m above sea level
# Define bin edges from 0 to the max wind speed, ensuring whole number intervals
max_speed = np.ceil(np.max(wind_speed_hub_flat))
bins = np.arange(0, max_speed + 1, 1)  # Bins: 0-1, 1-2, 2-3, ...
# Compute histogram
counts, bin_edges = np.histogram(wind_speed_hub_flat, bins=bins)
# Convert frequency to percentage
percentages = (counts / N) * 100
# Plot histogram as a step profile
plt.figure(figsize=(10, 6))
plt.step(bin_edges[:-1], percentages, where='post', color='blue', linewidth=2)
# Labels and formatting
plt.title('Histogram of wind speed occurrences at 130m above sea level')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Frequency (% of total)')
plt.xticks(bins)  # Ensure x-axis labels match bin edges
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Save and display the plot
plt.savefig(os.path.join(output_dir, "wind_speed_histogram_130m.png"), dpi=1200)
plt.show(block=False)

# karehamn location
lat_decimal = 56.9800
lon_decimal = 17.0200
# Select the nearest grid point in the dataset
wind_at_karehamn = wind_speed.sel(latitude=lat_decimal, longitude=lon_decimal, method='nearest')
wind_at_karehamn_130m = wind_speed_hub.sel(latitude=lat_decimal, longitude=lon_decimal, method='nearest')
# wind speed time series at kårehamn location at 10m
plt.figure(figsize=(10, 6))
plt.plot(time, wind_at_karehamn, color='green', linewidth=0.3)
plt.title(f'Wind speed time series at Kårehamn location ({lat_decimal}, {lon_decimal}) at 10m')
plt.xlabel('Time')
plt.ylabel('Wind speed (m/s)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "wind_speed_time_series_karehamn_10m.png"), dpi=1200)
plt.show(block=False)
# wind speed time series at kårehamn location at 130m
plt.figure(figsize=(10, 6))
plt.plot(time, wind_at_karehamn_130m, color='red', linewidth=0.3)
plt.title(f'Wind speed time series at Kårehamn location ({lat_decimal}, {lon_decimal}) at 130m')
plt.xlabel('Time')
plt.ylabel('Wind speed (m/s)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "wind_speed_time_series_karehamn_130m.png"), dpi=1200)
plt.show(block=False)

# CONSTANTS
rho = 1.225  # kg/m^3
dt_year = 365 * 24 * 3600  # seconds in a year

# Mean wind speed and standard deviation at 130m height at karehamn location
average_wind_speed_karehamn_130m = wind_at_karehamn_130m.mean(dim='time')
standard_deviation_karehamn_130m = wind_at_karehamn_130m.std(dim='time', ddof=1)
print(f"Average wind speed at Kårehamn location (130m): {average_wind_speed_karehamn_130m:.2f} m/s")
print(f"Standard deviation of wind speed at Kårehamn location (130m): {standard_deviation_karehamn_130m:.2f} m/s")
# Available wind power and annual energy density at karehamn location at 130m
available_power_density_karehamn_130m = 0.5 * rho * average_wind_speed_karehamn_130m**3
energy_density_per_area_karehamn_130m = available_power_density_karehamn_130m * dt_year
print(f"Available wind power density at Kårehamn location (130m): {available_power_density_karehamn_130m:.2f} W/m²")
#print(f"Energy density per unit area at Kårehamn location (130m): {((energy_density_per_area_karehamn_130m)*(10)**-9):.2f} GJ/m²")
print(f"Energy density per unit area at Kårehamn location (130m): {((energy_density_per_area_karehamn_130m)*((10)**(-9))/3.6):.2f} MWh/m²")

####################################################################################
# WIND SPEED DISTRIBUTIONS (Normal wind conditions)
#####################################################################################

# Rayleigh Cumulative Distribution Function (CDF)
def rayleigh_cdf(V_hub, V_ave):
    return 1 - np.exp(-np.pi * (V_hub / (2 * V_ave))**2)
# Rayleigh Probability Density Function (PDF)
def rayleigh_pdf(V_hub, V_ave):
    return np.pi/2*(V_hub / V_ave**2) * np.exp(-np.pi * (V_hub / (2 * V_ave))**2)

# Average wind speeds for the IEC classes
V_ave1 = 10   # IEC Class 1
V_ave2 = 8.5  # IEC Class 2
V_ave3 = 7.5  # IEC Class 3
# Average wind speed at hub height at karehamn location
V_ave_karehamn = round(float(average_wind_speed_karehamn_130m),1)

# Generate wind speed values
V_hub = np.linspace(0, 35, 500)

# Rayleigh CDF for each IEC class
cdf_rayleigh1 = rayleigh_cdf(V_hub, V_ave1)
cdf_rayleigh2 = rayleigh_cdf(V_hub, V_ave2)
cdf_rayleigh3 = rayleigh_cdf(V_hub, V_ave3)
# Rayleigh CDF for the average wind speed at Kårehamn location
cdf_rayleigh_karehamn = rayleigh_cdf(V_hub, V_ave_karehamn)
# Rayleigh PDF for each IEC class
pdf_rayleigh1 = rayleigh_pdf(V_hub, V_ave1)
pdf_rayleigh2 = rayleigh_pdf(V_hub, V_ave2)
pdf_rayleigh3 = rayleigh_pdf(V_hub, V_ave3)
# Rayleigh PDF for the average wind speed at Kårehamn location
pdf_rayleigh_karehamn = rayleigh_pdf(V_hub, V_ave_karehamn)

# Plot CDF
plt.figure(figsize=(10, 6))
plt.plot(V_hub, cdf_rayleigh1, label='IEC Class 1 (V_ave=10 m/s)', color='blue')
plt.plot(V_hub, cdf_rayleigh2, label='IEC Class 2 (V_ave=8.5 m/s)', color='green')
plt.plot(V_hub, cdf_rayleigh3, label='IEC Class 3 (V_ave=7.5 m/s)', color='red')
plt.plot(V_hub, cdf_rayleigh_karehamn, label='Kårehamn (V_ave=9.7 m/s)', color='orange')
plt.xlabel('Wind speed at hub height [m/s]')
plt.ylabel('Rayleigh CDF')
plt.title('Rayleigh Cumulative Distribution Function (CDF) for different IEC classes')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "rayleigh_cdf.png"), dpi=1200)
plt.show(block=False)

# Plot PDF
plt.figure(figsize=(10, 6))
plt.plot(V_hub, pdf_rayleigh1, label='IEC Class 1 (V_ave=10 m/s)', color='blue')
plt.plot(V_hub, pdf_rayleigh2, label='IEC Class 2 (V_ave=8.5 m/s)', color='green')
plt.plot(V_hub, pdf_rayleigh3, label='IEC Class 3 (V_ave=7.5 m/s)', color='red')
plt.plot(V_hub, pdf_rayleigh_karehamn, label=f'Kårehamn (V_ave={V_ave_karehamn} m/s)', color='orange')
plt.xlabel('Wind speed at hub height [m/s]')
plt.ylabel('Rayleigh PDF')
plt.title('Rayleigh Probability Density Function (PDF) for different IEC classes')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "rayleigh_pdf.png"), dpi=1200)
plt.show(block=False)


####################################################################################
# VELOCITY DURATION CURVES (VDCs) 
#####################################################################################
# Sort wind speed data for Kårehamn location at 130m
sorted_wind_speed = np.sort(wind_at_karehamn_130m)[::-1]
duration_percentage = (np.arange(1, N + 1) / N) * 100

# Plot: Velocity Duration Curve (VDC)
plt.figure(figsize=(10, 6))
plt.plot(duration_percentage, sorted_wind_speed, label='Wind speed (m/s)', color='blue')
plt.title('Velocity Duration Curve (VDC)')
plt.xlabel('Duration (%)')
plt.ylabel('Wind speed (m/s)')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, "velocity_duration_curve.png"), dpi=1200)
plt.show(block=False)

####################################################################################
# POWER DURATION CURVES (PDCs) 
#####################################################################################
# Generalized wind turbine power curve function
def power_curve(U, cut_in_speed, cut_out_speed, rated_speed, rated_power):
    P = np.zeros_like(U)
    for i in range(len(U)):
        if U[i] < cut_in_speed or U[i] >= cut_out_speed:
            P[i] = 0
        elif cut_in_speed <= U[i] < rated_speed:
            P[i] = rated_power * ((U[i] - cut_in_speed) / (rated_speed - cut_in_speed))**3
        else:  # rated_speed <= U < cut_out_speed
            P[i] = rated_power
    return P

def wind_power_curve(U, rho = 1.225):
    P = np.zeros_like(U)
    for i in range(len(U)):
        if U[i] > 0:
            P[i] = 0.5*rho*U[i]**3
    return P
    
turbines = {
    'Vestas V236 1x15MW': {'cut_in_speed': 3, 'cut_out_speed': 31, 'rated_speed': 11.1, 'rated_power': 15},
    'GE Haliade 1x14MW': {'cut_in_speed': 3, 'cut_out_speed': 34, 'rated_speed': 11.1, 'rated_power': 14.0},
    'Nezzy2 1x10MW': {'cut_in_speed': 4, 'cut_out_speed': 25, 'rated_speed': 11.8, 'rated_power': 10.0},
    'IEA 1x10MW': {'cut_in_speed': 4, 'cut_out_speed': 25, 'rated_speed': 11, 'rated_power': 10.0}
}

wind_speeds = wind_at_karehamn_130m.values.flatten()
wind_speeds = wind_speeds[~np.isnan(wind_speeds)]  # Remove NaN values
wind_speeds_sorted = np.sort(wind_speeds)[::-1]

plt.figure(figsize=(10, 6))
mean_power_outputs = {}
capacity_factors = {}
annual_energy_production = {}

for name, specs in turbines.items():
    powers = power_curve(
        wind_speeds,
        cut_in_speed=specs['cut_in_speed'],
        cut_out_speed=specs['cut_out_speed'],
        rated_speed=specs['rated_speed'],
        rated_power=specs['rated_power']
    )


    # Mean power output
    mean_power = np.mean(powers)
    mean_power_outputs[name] = mean_power

    # Capacity Factor
    cf = mean_power / specs['rated_power']
    capacity_factors[name] = cf

    # Annual Energy Production (AEP in MWh)
    aep = mean_power * 8760  # hours in a year
    annual_energy_production[name] = aep
    
    # Plot the power duration curve
    powers_sorted = np.sort(powers)[::-1]
    duration_percentage = (np.arange(1, len(powers_sorted) + 1) / len(powers_sorted)) * 100
    plt.plot(duration_percentage, powers_sorted, label=name)

available_power_sorted = wind_power_curve(wind_speeds_sorted)

# Output results
print("Turbine Performance Summary:")
for name in turbines.keys():
    print(f"\n{name}:")
    print(f"  - Mean Power Output: {mean_power_outputs[name]:.2f} MW")
    print(f"  - Capacity Factor: {capacity_factors[name]*100:.1f}%")
    #print(f"  - Annual Energy Production: {annual_energy_production[name]:,.0f} MWh")
    print(f"  - Annual Energy Production: {annual_energy_production[name]/1000:,.0f} GWh")

#plt.plot(duration_percentage, available_power_sorted, label='Available Wind Power', color='black', linestyle='--')
plt.title('Power Duration Curve (PDC) at Kårehamn (130m)')
plt.xlabel('Duration (%)')
plt.ylabel('Power Output (MW)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "power_duration_curves_all_turbines.png"), dpi=1200)
plt.show(block=False)

# # Specific wind turbine power curves
# # Vestas V236 1x15MW
# def power_curve_vestas(U):
#     return power_curve(U, cut_in_speed=3, cut_out_speed=31, rated_speed = 11.1, rated_power=15)
# # GE Haliade 1x14MW
# def power_curve_haliade(U):
#     return power_curve(U, cut_in_speed=3, cut_out_speed=34, rated_speed = 11.1, rated_power=14.0)
# # Nezzy2 1x10MW
# def power_curve_nezzy2(U):
#     return power_curve(U, cut_in_speed=4, cut_out_speed=25, rated_speed = 11.8, rated_power=10.0)
# # IEA 1x10MW 
# def power_curve_iea(U):
#     return power_curve(U, cut_in_speed=4, cut_out_speed=25, rated_speed = 11, rated_power=10.0)

# V_hub = np.linspace(0, 35, 500)

# # Power curves for different turbines
# plt.figure(figsize=(10, 6))
# plt.plot(V_hub, power_curve_vestas(V_hub), label='Vestas V236 1x15MW', color='blue')
# plt.plot(V_hub, power_curve_haliade(V_hub), label='GE Haliade 1x14MW', color='green')
# plt.plot(V_hub, power_curve_nezzy2(V_hub), label='Nezzy2 1x10MW', color='red')
# plt.plot(V_hub, power_curve_iea(V_hub), label='IEA 1x10MW', color='orange')
# plt.title('Power Duration Curve (PDC) for different wind turbines')
# plt.xlabel('Wind speed (m/s)')
# plt.ylabel('Power (MW)')
# plt.grid(True)
# plt.legend()
# plt.savefig(os.path.join(output_dir, "power_duration_curve_turbines.png"), dpi=1200)
# plt.show(block=False)

#####################################################################################
# TURBULENCE INTENSITY AND IEC REFERENCE VALUE (I_ref) 
#####################################################################################

TI = standard_deviation_karehamn_130m/average_wind_speed_karehamn_130m
TI = TI.where(average_wind_speed_karehamn_130m > 0)  # Avoid division by zero
# Define wind speed bins (e.g., 0–1, 1–2, ..., max)
bin_edges = np.arange(0, np.ceil(np.max(wind_speed_hub_flat)) + 1, 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
TI_per_bin = []
for i in range(len(bin_edges) - 1):
    # Indices where wind speed is within the bin
    bin_mask = (wind_speed_hub_flat >= bin_edges[i]) & (wind_speed_hub_flat < bin_edges[i + 1])
    speeds_in_bin = wind_speed_hub_flat[bin_mask]
    
    if len(speeds_in_bin) > 1:
        U_mean_bin = np.mean(speeds_in_bin)
        sigma_u_bin = np.std(speeds_in_bin, ddof=1)
        TI = sigma_u_bin / U_mean_bin
        TI_per_bin.append(TI)
    else:
        TI_per_bin.append(np.nan)  # Not enough data in bin

# Plot: TI vs Wind Speed
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, TI_per_bin, marker='o', linestyle='-', color='darkgreen')
plt.title('Turbulence Intensity (TI) vs wind speed')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Turbulence Intensity (σ₍ᵤ₎ / Ū)')
plt.grid(True)
plt.ylim(bottom=0)
plt.savefig(os.path.join(output_dir, "turbulence_intensity_vs_wind_speed.png"), dpi=1200)
plt.show(block=False)

####################################################################################
# NORMAL TURBULENCE MODEL (NTM) - (Normal wind conditions)
#####################################################################################
# constants
b = 5.6
# Reference turbulence intensities for different IEC classes
I_refAplus = 0.18 # IEC Class A+
I_refA = 0.16 # IEC Class A
I_refB = 0.14 # IEC Class B
I_refC = 0.12 # IEC Class C

def standard_deviation_ntm(I_ref, V_hub, b):
    """
    Calculate the standard deviation of wind speed using the Normal Turbulence Model (NTM).
    
    Parameters:
    I_ref (float): Reference turbulence intensity
    V_hub (float): Wind speed at hub height
    b (float): Coefficient for the NTM (5.6m/s)
    
    Returns:
    float: Standard deviation of wind speed
    """
    return I_ref*(0.75*V_hub+b)

# Calculate standard deviation for each IEC class
std_dev_Aplus = standard_deviation_ntm(I_refAplus, V_hub, b)
std_dev_A = standard_deviation_ntm(I_refA, V_hub, b)
std_dev_B = standard_deviation_ntm(I_refB, V_hub, b)
std_dev_C = standard_deviation_ntm(I_refC, V_hub, b)

# Plot standard deviation for each IEC class
plt.figure(figsize=(10, 6))
plt.plot(V_hub, std_dev_Aplus, label='IEC Class A+', color='blue')
plt.plot(V_hub, std_dev_A, label='IEC Class A', color='green')
plt.plot(V_hub, std_dev_B, label='IEC Class B', color='red')
plt.plot(V_hub, std_dev_C, label='IEC Class C', color='orange')
plt.xlabel('Wind speed at hub height [m/s]')
plt.ylabel('Standard deviation of wind speed [m/s]')
plt.title('Standard deviation of wind speed using NTM for different IEC classes')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "standard_deviation_ntm.png"), dpi=1200)
plt.show(block=False)

V_hub = np.linspace(0.1, 35, 500)

def turbulence_intensity(I_ref, V_hub, b):
    """
    Calculate the turbulence intensity for different wind speeds using the standard deviation.
    
    Parameters:
    I_ref (float): Reference turbulence intensity
    V_hub (float): Wind speed at hub height
    b (float): Coefficient for the NTM
    
    Returns:
    float: Turbulence intensity
    """
    # Calculate standard deviation
    std_dev = standard_deviation_ntm(I_ref, V_hub, b)
    # Calculate turbulence intensity
    return std_dev / V_hub

# Calculate turbulence intensity for each IEC class
TI_Aplus = turbulence_intensity(I_refAplus, V_hub, b)
TI_A = turbulence_intensity(I_refA, V_hub, b)
TI_B = turbulence_intensity(I_refB, V_hub, b)
TI_C = turbulence_intensity(I_refC, V_hub, b)

# Plot turbulence intensity for each IEC class
plt.figure(figsize=(10, 6))
plt.plot(V_hub, TI_Aplus, label='IEC Class A+', color='blue')
plt.plot(V_hub, TI_A, label='IEC Class A', color='green')
plt.plot(V_hub, TI_B, label='IEC Class B', color='red')
plt.plot(V_hub, TI_C, label='IEC Class C', color='orange')
plt.xlabel('Wind speed at hub height [m/s]')
plt.ylabel('Turbulence intensity [ ]')
plt.ylim(bottom=0, top=0.5)
plt.title('Turbulence Intensity for Different IEC Classes')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "turbulence_intensity_ntm.png"), dpi=1200)
plt.show(block=False)

####################################################################################
# GUST WIND SPEEDS
#####################################################################################
t = np.logspace(np.log10(3600), np.log10(1), 100)

def gust_factor(I_ref, t):
    return 1+0.42*I_ref*np.log(3600/t)

# Calculate gust wind speeds for each IEC class
gust_Aplus = gust_factor(I_refAplus, t)
gust_A = gust_factor(I_refA, t)
gust_B = gust_factor(I_refB, t)
gust_C = gust_factor(I_refC, t)
# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, gust_Aplus, label='IEC Class A+', color='blue')
plt.plot(t, gust_A, label='IEC Class A', color='green')
plt.plot(t, gust_B, label='IEC Class B', color='red')
plt.plot(t, gust_C, label='IEC Class C', color='orange')
# Set log-log scale
plt.xscale('log')
# Invert x-axis: so 3600s is left, 1s is right
plt.gca().invert_xaxis()
# Custom ticks (in seconds) and labels
custom_ticks = [3600, 1800, 600, 300, 60, 30, 10, 3, 1]
custom_labels = ["1 hr", "30 min", "10 min", "5 min", "60 s", "30 s", "10 s", "3 s", "1 s"]
plt.xticks(custom_ticks, custom_labels)
plt.xlabel('Gust duration [s]')
plt.ylabel('Gust factor, G [-]')
plt.title('Gust factor (G) vs gust duration')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(bottom=1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gust_wind_speeds.png"), dpi=1200)
plt.show(block=False)

####################################################################################
# EXTREME WIND SPEEDS
#####################################################################################
V_ref = 5*V_ave_karehamn
z_hub = 130
z = np.linspace(0.1, 200, 100)
V_e50 = 1.4*V_ref*(z/z_hub)**0.11
V_e1 = 0.8*V_e50
# Plot
plt.figure(figsize=(10, 6))
plt.plot(z, V_e50, label='50-year return period', color='blue')
plt.plot(z, V_e1, label='1-year return period', color='green')
plt.xlabel('Height above sea-level [m]')
plt.ylabel('Extreme wind speed [m/s]')
plt.title('Extreme wind speeds at different heights')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "extreme_winds.png"), dpi=1200)
plt.show(block=False)

# === Keep all figures open ===
plt.pause(0.1)
input("Press Enter to close all figures...")
plt.close('all')