import os
import click
import os.path as op
import numpy as np
import pandas as pd
from nilearn import image, masking


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


@click.command(name='compute_TSNR')
@click.option('--in_file', help='In-file')
@click.option('--highpass', is_flag=True, help='Whether to highpass the data')
@click.option('--motion_corr', is_flag=True, help='Whether to remove motion params')
def compute_TSNR(in_file, highpass=True, motion_corr=False):
    """ Computes temporal signal-to-noise ratio (TSNR) for
    a functional MRI file preprocessed with FMRIPREP.
    
    Parameters
    ----------
    in_file : str
        Path to functional MRI file
    highpass : bool
        Whether to highpass the data (using the cosine basis
        set in the *confounds.tsv file)
    motion_corr : bool
        Whether to include the six motion parameters from
        motion correction in the confound regression step.
        
    Returns
    -------
    out_file : str
        Path to TSNR file
    """

    base_dir = op.dirname(in_file)
    base_file = op.basename(in_file)
    if 'preproc' not in base_file:
        raise ValueError("In_file should be a preprocessed file!")
    
    # Mask BOLD file
    mask = op.join(base_dir, base_file.split('preproc')[0] + 'brain_mask.nii.gz')
    bold_masked = masking.apply_mask(in_file, mask)

    # Get confounds and regress out of data
    if highpass or motion_corr:
        conf = op.join(base_dir, base_file.split('space')[0] + 'desc-confounds_regressors.tsv')
        conf_df = pd.read_csv(conf, sep='\t')
        
        conf_vars = []
        if highpass:
            conf_vars.extend([col for col in conf_df.columns if 'cosine' in col])
        
        if motion_corr:
            conf_vars.extend(['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])
            
        conf_df = conf_df.loc[:, conf_vars]
        conf_values = np.c_[np.ones(conf_df.shape[0]), conf_df.values]
        conf_params = np.linalg.lstsq(conf_values, bold_masked, rcond=-1)[0]
        bold_masked = bold_masked - conf_values[:, 1:].dot(conf_params[1:, :])

    # Compute tsnr
    tsnr = bold_masked.mean(axis=0) / bold_masked.std(axis=0)
    tsnr_img = masking.unmask(tsnr, mask)
    out_file = in_file.replace('.nii.gz', '_TSNR.nii.gz')
    tsnr_img.to_filename(out_file)
    return out_file
    

def extract_kwargs_from_ctx(ctx):
    """ Extracts kwargs from Click context manager. """
    args = []
    i = 0
    for arg in ctx.args:
        if arg[:2] == '--':
            args.append([arg[2:]])
            i += 1
        else:
            args[(i-1)].append(arg)

    for i, arg in enumerate(args):
        if len(arg) == 1:
            args[i] = [arg[0], True]
        elif len(args) > 2:
            args[i] = [arg[0], ' '.join(arg[1:])]

    keys = [arg[0] for arg in args]
    if len(keys) != len(set(keys)):
        msg = "Your cmd arguments contain a duplicate!"
        raise ValueError(msg)

    kwargs = {arg[0]: arg[1] for arg in args}
    return kwargs

