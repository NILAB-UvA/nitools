import os
import os.path as op
from glob import glob
import shutil


default_args = {
    '--debug': False,
    '--nthreads': 10,
    '--omp-nthreads': 4,
    '--mem_mb': False,
    '--low-mem': False,
    '--use-plugin': False,
    '--anat-only': False,
    '--ignore': 'slicetiming',
    '--ignore-aroma-denoising-errors': True,
    '--verbose': False,
    '--longitudinal': False,
    '--t2s-coreg': False,
    '--bold2t1w-dof': 9,
    '--output-space': 'template',
    '--force-bbr': False,
    '--force-no-bbr': False,
    '--template': 'MNI152NLin2009cAsym',
    '--template-resampling-grid': 'native',
    '--medial-surface-nan': False,
    '--use-aroma': False,
    '--skull-strip-template': 'OASIS',
    '--fmap-bspline': False,
    '--fmap-no-demean': False,
    '--use-syn-sdc': False,
    '--force-syn': False,
    '--fs-no-reconall': True,
    '--no-submm-recon': False,
    '--fs-license-file': '/usr/local/freesurfer/license.txt',
    '-w': False,
    '--resource-monitor': True,
    '--reports-only': False,
    '--run-uuid': False,
    '--write-graph': False,
    '--stop-on-first-crash': False
}


def run_preproc(bids_dir, export_dir=None, **fmriprep_options):
    """ Runs data from BIDS-directory through fmriprep pipeline.

    Parameters
    ----------
    bids_dir: str
        Absolute path to BIDS-directory
    export_dir : str or None
	If string, it points to a path to copy the data to. If None,
        this is ignored
    subs: list of str
        List of subject-identifiers (e.g., sub-0001) which need to be run
        through the pipeline
    **fmriprep_options: kwargs
        Keyword arguments of fmriprep-options
    """

    out_dir = op.join(op.dirname(bids_dir), 'preproc')
    fmriprep_dir = op.join(out_dir, 'fmriprep')
    subs_done = [op.basename(s).split('.html')[0]
                 for s in sorted(glob(op.join(fmriprep_dir, '*html')))]
    bids_subs = [op.basename(f) for f in sorted(glob(op.join(bids_dir, 'sub*')))]
    participant_labels = [sub.split('-')[1] for sub in bids_subs if sub not in subs_done]

    fmriprep_options = {('--' + key): value for key, value in fmriprep_options.items()}
    default_args.update(fmriprep_options)
    fmriprep_options = {key: value for key, value in default_args.items() if value}
    options_str = [key + ' ' + str(value) for key, value in fmriprep_options.items()]
    cmd = f'fmriprep-docker {bids_dir} {out_dir} ' + ' '.join(options_str).replace(' True', '')
    cmd += ' --participant_label %s' % ' '.join(participant_labels)
    #os.system(cmd)

    if export_dir is not None:
        copy_dir = op.join(export_dir, 'preproc')
        if not op.isdir(copy_dir):
            os.makedirs(copy_dir)
        
        proc_sub_data = sorted(glob(op.join(fmriprep_dir, 'sub-*')))
        done_sub_data = [op.basename(f) for f in sorted(glob(op.join(copy_dir, 'sub-*')))]
        
        for f in proc_sub_data:
            if op.basename(f) not in done_sub_data:
                if op.isdir(f):
                    shutil.copytree(f, op.join(copy_dir, op.basename(f)))
                else:
                    shutil.copyfile(f, op.join(copy_dir, op.basename(f)))



