import click
import os
import os.path as op
from glob import glob
import shutil
import subprocess
import yaml
from .utils import extract_kwargs_from_ctx
from .version import FMRIPREP_VERSION


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
    '--stop-on-first-crash': False,
    '--image': 'poldracklab/fmriprep:%s' % FMRIPREP_VERSION
}


@click.command(name='run_preproc', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--bids_dir', default=os.getcwd(), help='BIDS-directory.')
@click.option('--out_dir', default=None, help='output-directory.')
@click.option('--export_dir', default=None, help='Directory to export data.')
@click.option('--run_single', is_flag=True, default=True, help='Run a single subject at the time.')
@click.pass_context
def run_preproc_cmd(ctx, bids_dir, run_single=True, out_dir=None, export_dir=None, **fmriprep_options):
    """ CMD interface """
    fmriprep_options = extract_kwargs_from_ctx(ctx)
    run_preproc(bids_dir, run_single, out_dir, export_dir, **fmriprep_options)
    

def run_preproc(bids_dir, run_single=True, out_dir=None, export_dir=None, **fmriprep_options):
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

    cp_file = op.join(op.dirname(__file__), 'data', 'CURRENT_PROJECTS.yml')
    with open(cp_file, 'r') as cpf:
        curr_projects = yaml.load(cpf)

    par_dir = op.basename(op.dirname(bids_dir))
    if par_dir in curr_projects.keys():
        extra_opts = curr_projects[par_dir]['fmriprep_options']
        if 'version' in extra_opts.keys():
            default_args['--imgage'] = 'poldracklab/fmriprep:%s' % extra_opts['version']

        fmriprep_options.update(extra_opts)

    # make sure is abspath
    bids_dir = op.abspath(bids_dir)

    if out_dir is None:
        out_dir = op.join(op.dirname(bids_dir), 'preproc')

    out_dir = op.abspath(out_dir)

    # Define directories + find subjects
    fmriprep_dir = op.join(out_dir, 'fmriprep')
    subs_done = [op.basename(s).split('.html')[0]
                 for s in sorted(glob(op.join(fmriprep_dir, '*html')))]
    bids_subs = [op.basename(f) for f in sorted(glob(op.join(bids_dir, 'sub*')))]

    # Define subjects which need to be preprocessed
    participant_labels = [sub.split('-')[1] for sub in bids_subs
                          if sub not in subs_done]

    # Merge default arguments and desided arguments from user (which will)
    # overwrite default arguments
    fmriprep_options = {('--' + key): value for key, value in fmriprep_options.items()}
    default_args.update(fmriprep_options)
    all_fmriprep_options = {key: value for key, value in default_args.items() if value}
    options_str = [key + ' ' + str(value) for key, value in all_fmriprep_options.items()]

    # Construct command
    cmd = f'fmriprep-docker {bids_dir} {out_dir} ' + ' '.join(options_str).replace(' True', '') 
    if participant_labels:
        if run_single:
            cmds = [cmd + ' --participant_label %s' % plabel for plabel in participant_labels]
        else:
            cmds = [cmd + ' --participant_label %s' % ' '.join(participant_labels)]
    else:
        cmds = []

    # Only run if there are actually participants to be processed
    if cmds:
        for cmd in cmds:
            sub_label = cmd.split('--participant_label ')[-1]
            print("Running participant(s): %s ..." % sub_label)
            fout = open(op.join(op.dirname(out_dir), 'fmriprep_stdout.txt'), 'a+')
            ferr = open(op.join(op.dirname(out_dir), 'fmriprep_stderr.txt'), 'a+') 
            subprocess.run(cmd.split(' '), stdout=fout, stderr=ferr)
            fout.close()
            ferr.close()
    else:
        print('All subjects seem to have been preprocessed already!')

    # If an export-dir is defined, copy stuff to export-dir (if None, nothing
    # is copied)
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
