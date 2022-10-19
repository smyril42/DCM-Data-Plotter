# DCM-Data-Plotter

- Einlesen von Dicom Files
- Einteilen in die einzelnen slices
- Erfassen der mittelwert jedes slices auf jedem bild
- Erfassen der standartjabweichung und des mittelwerts der Bilder für jeden slice
- Erfassen des Signal-Noise-Ratio (SNR)
- Plotten der Daten

### Librarys:
Time, Pydicom, Os, Numpy, Scipy, Math, Matplotlib

### Input: 
Dicom Files, File Count, Slices

### Output:
For every Slice:
Mean value for each picture, mean value for all pictures, standart deviation
Signal to noise ratio

