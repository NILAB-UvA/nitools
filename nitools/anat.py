import os
import click
from nilearn import image


@click.command(name='scale_MP2RAGE')
@click.option('--in_file', help='In-file')
@click.option('--minimum', default=0, type=int, help='minimum value')
@click.option('--mean', default=100, type=int, help='Approximate mean value')
@click.option('--overwrite', is_flag=True, help='Overwrite file?')
def scale_MP2RAGE(in_file, minimum=0, mean=100, overwrite=True):
    """ Scales the MP2RAGE such that does not have any
    negative values and a mean around 100.

    Based on https://mail.nmr.mgh.harvard.edu/pipermail//freesurfer/2017-July/052950.html
    """

    scaled = image.math_img('(img - np.min(img)) * 100', img=in_file)
    json = in_file.replace('.nii.gz', '.json')
    
    if overwrite:
        scaled.to_filename(in_file)
    else:
        scaled.to_filename(in_file.replace('acq-', 'acq-Rescaled'))
        if os.path.isfile(json):
            os.rename(json, json.replace('acq-', 'acq-Rescaled'))
