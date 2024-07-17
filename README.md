import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal as signal

# Define the Butterworth filter function
def butterworth_lowpass_filter(data, cutoff_time, axis=0):
    cutoff_freq = 1 / cutoff_time
    nyfreq = cutoff_freq * 2
    order = 4
    B, A = signal.butter(order, nyfreq, output='ba')
    return signal.filtfilt(B, A, data, axis=axis)

# Define the filter_data function
def filter_data(data, var, cutoff_time=400):
    data_low = butterworth_lowpass_filter(data, cutoff_time)
    data_low = xr.DataArray(data_low, coords={"time": data.TAX}, name=str(var), dims=["TAX"])
    return data_low

# Load the NetCDF file
section_sla_file_path = "C:/Users/Janki/Downloads/section_sla.nc"
section_sla_data = xr.open_dataset(section_sla_file_path)

# Calculate the mean of the data over latitude and longitude
mean_sla = section_sla_data['SLA'].mean(dim=["YAX", "XAX"])

# Apply the 400-day low-pass filter using filter_data function
mean_sla_lowpass = filter_data(mean_sla, "sla_lowpass", cutoff_time=400)

# Load the pre-filtered data for comparison
section_sla_low_file_path = "C:/Users/Janki/Downloads/section_sla_low.nc"
section_sla_low_data = xr.open_dataset(section_sla_low_file_path)
mean_sla_low = section_sla_low_data['SLA_LOW'].mean(dim=["YAX", "XAX"])

# Plot the unfiltered, filtered, and pre-filtered data
plt.figure(figsize=(14, 7))
plt.plot(mean_sla['TAX'], mean_sla, label='Unfiltered SLA', color='orange')
plt.plot(mean_sla['TAX'], mean_sla_lowpass, label='400-day Low-pass Filtered SLA', color='blue')
plt.plot(mean_sla_low['TAX'], mean_sla_low, label='Pre-filtered SLA', color='green', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('SLA (m)')
plt.title('SLA Time Series for the Bay of Bengal')
plt.legend()
plt.show()
