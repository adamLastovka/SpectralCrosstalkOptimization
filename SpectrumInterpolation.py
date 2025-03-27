import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = "LEDs\XEG-PHOTORED.csv"
out_file_name = "LEDs\XEG-PHOTORED_Processed.csv"

raw_data = np.genfromtxt(file_name,delimiter=',',skip_header=True)

wavelengths = np.arange(250,750,1,dtype=np.float64)

interpolated_data = np.interp(wavelengths, raw_data[:,0], raw_data[:,1])/100

output_data = np.column_stack((wavelengths,interpolated_data))
df = pd.DataFrame(output_data,columns=("Wavelength","Transmittance"))

df.to_csv(out_file_name, index=False)

plt.plot(wavelengths,interpolated_data)
plt.show()