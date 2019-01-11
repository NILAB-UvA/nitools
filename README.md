# nitools
NeuroImaging tools developed for MRI (pre)processing at the Spinoza Centre, location Roeterseiland. 

## Notes to self
dcm2niix crashes for some high-resolution functional scans. In that case, edit `nii_dicom.h` and change the following:
```C
static const int kMaxSlice2D = 70000;
```

`fmriprep-docker` won't run as a cron-job because of the `-it` flag in the docker command.
Edit `fmriprep_docker.py`.
