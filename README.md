import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# File paths
momsis_sst_file_path = "C:/Users/Janki/Downloads/MOMSIS/MOMSIS_SST.nc"
avhrr_file_path = "C:/Users/Janki/Downloads/AVHRR_SST_SIC.nc"

# Open the NetCDF files
momsis_sst_data = netCDF4.Dataset(momsis_sst_file_path, 'r')
avhrr_data = netCDF4.Dataset(avhrr_file_path, 'r')

# Read the specific variables you want to plot for MOMSIS and AVHRR
lon_momsis = momsis_sst_data.variables['xt_ocean'][:]
lat_momsis = momsis_sst_data.variables['yt_ocean'][:]

# Read the daily SST data for the period 2000 to 2019
sst_daily_momsis = momsis_sst_data.variables['SST_REGRID'][:]
sst_daily_avhrr = avhrr_data.variables['SST_REGRID'][:]

# Function to handle missing values in chunks
def fill_missing_values_in_chunks(data, chunk_size=100):
    num_chunks = data.shape[0] // chunk_size + 1
    filled_data = np.empty_like(data)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, data.shape[0])
        chunk = data[start_idx:end_idx, :, :]
        mean_val = np.nanmean(chunk)
        filled_chunk = np.nan_to_num(chunk, nan=mean_val)
        filled_data[start_idx:end_idx, :, :] = filled_chunk
    return filled_data

# Handle missing values in chunks to reduce memory usage
sst_daily_momsis = fill_missing_values_in_chunks(sst_daily_momsis)
sst_daily_avhrr = fill_missing_values_in_chunks(sst_daily_avhrr)

# Define the coordinates for the sectors (IO, PO, RS, BAS, WS)
regions = {
    "IO": ((20, 90), (-90, -60)),
    "PO": ((90, 160), (-90, -60)),
    "RS": ((160, 230), (-90, -60)),
    "BAS": ((230, 300), (-90, -60)),
    "WS": ([(300, 360), (0, 20)], (-90, -60))  # Wraparound longitude
}

# Function to compute the daily mean for a given region
def compute_daily_mean(data, lon, lat, lon_range, lat_range):
    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
    if isinstance(lon_range, list):  # Handle wraparound
        lon_mask = np.logical_or((lon >= lon_range[0][0]) & (lon <= lon_range[0][1]),
                                 (lon >= lon_range[1][0]) & (lon <= lon_range[1][1]))
    else:
        lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])
    region_data = data[:, lat_mask, :][:, :, lon_mask]
    daily_mean = np.nanmean(region_data, axis=(1, 2))
    return daily_mean

# Compute daily means for each sector
sst_daily_mean_momsis = {}
sst_daily_mean_avhrr = {}

for region, (lon_range, lat_range) in regions.items():
    print(f"Computing daily means for {region} sector...")
    sst_daily_mean_momsis[region] = compute_daily_mean(sst_daily_momsis, lon_momsis, lat_momsis, lon_range, lat_range)
    sst_daily_mean_avhrr[region] = compute_daily_mean(sst_daily_avhrr, lon_momsis, lat_momsis, lon_range, lat_range)
    print(f"MOMSIS daily mean for {region}: {sst_daily_mean_momsis[region][:10]}")  # Print first 10 values for debugging
    print(f"AVHRR daily mean for {region}: {sst_daily_mean_avhrr[region][:10]}")  # Print first 10 values for debugging

# Create subplots for each sector
fig, axes = plt.subplots(5, 1, figsize=(12, 30), sharex=True)
fig.suptitle('FFT Comparison of SST between AVHRR and MOMSIS (2000-2019) for Different Sectors')

for i, region in enumerate(regions.keys()):
    print(f"Processing region: {region}")

    # Apply FFT to the daily mean SST values for the region
    fft_momsis = fft(sst_daily_mean_momsis[region])
    fft_avhrr = fft(sst_daily_mean_avhrr[region])

    # Compute the frequencies
    N = len(sst_daily_mean_momsis[region])
    T = 1.0  # Sampling interval (1 day)
    frequencies = fftfreq(N, T)[:N // 2]

    # Remove zero frequencies to avoid division by zero
    non_zero_freqs = frequencies != 0
    frequencies = frequencies[non_zero_freqs]
    fft_momsis = fft_momsis[:N // 2][non_zero_freqs]
    fft_avhrr = fft_avhrr[:N // 2][non_zero_freqs]

    # Plot the FFT results for the region
    axes[i].plot(frequencies, 2.0 / N * np.abs(fft_momsis), 'r', label='MOMSIS')
    axes[i].plot(frequencies, 2.0 / N * np.abs(fft_avhrr), 'k', label='AVHRR')
    axes[i].set_title(f'FFT Comparison for {region} Sector')
    axes[i].set_ylabel('Amplitude')
    axes[i].legend()
    axes[i].grid()

    # Set x-axis to log scale
    axes[i].set_xscale('log')
    axes[i].set_xlim(1e-4, 1e-1)  # Adjusted range to show more detail
    axes[i].set_ylim(0,2)

    # Find peaks and their amplitudes
    amplitude_momsis = 2.0 / N * np.abs(fft_momsis)
    amplitude_avhrr = 2.0 / N * np.abs(fft_avhrr)

    peaks_momsis, _ = find_peaks(amplitude_momsis)
    peaks_avhrr, _ = find_peaks(amplitude_avhrr)

    # Get the top 5 peaks by amplitude
    top_peaks_momsis = sorted(peaks_momsis, key=lambda x: amplitude_momsis[x], reverse=True)[:5]
    top_peaks_avhrr = sorted(peaks_avhrr, key=lambda x: amplitude_avhrr[x], reverse=True)[:5]

    # Annotate the top 5 peaks for MOMSIS
    for peak in top_peaks_momsis:
        axes[i].annotate(f'{1 / frequencies[peak]:.2f} days',
                         xy=(frequencies[peak], amplitude_momsis[peak]),
                         xytext=(5, 5), textcoords='offset points', fontsize=5)

    # Annotate the top 5 peaks for AVHRR
    for peak in top_peaks_avhrr:
        axes[i].annotate(f'{1 / frequencies[peak]:.2f} days',
                         xy=(frequencies[peak], amplitude_avhrr[peak]),
                         xytext=(5, 5), textcoords='offset points', fontsize=5)

axes[-1].set_xlabel('Frequency (1/days)')
plt.show()

# Close the NetCDF files
momsis_sst_data.close()
avhrr_data.close()
