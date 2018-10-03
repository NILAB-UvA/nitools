# nitools
NeuroImaging tools developed for MRI (pre)processing at the Spinoza Centre, location Roeterseiland. 

## Note to self
dcm2niix crashes for some high-resolution functional scans. In that case, edit `nii_dicom.h` and change the following:
```C
static const int kMaxSlice2D = 70000;
```
