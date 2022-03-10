import click
import os
import os.path as op
from glob import glob
import shutil
import subprocess
import yaml
from datetime import datetime
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
    #'--ignore-aroma-denoising-errors': True,
    '--verbose': False,
    '--longitudinal': False,
    '--t2s-coreg': False,
    '--bold2t1w-dof': 9,
    '--output-space': 'template',
    '--output-spaces': 'MNI152NLin6Asym T1w',
    '--force-bbr': False,
    '--force-no-bbr': False,
    '--template': 'MNI152NLin2009cAsym',
    '--template-resampling-grid': 'native',
    '--medial-surface-nan': False,
    '--use-aroma': False,
    # '--skull-strip-template': 'OASIS',  # GIVES AN ERROR >1.3.1
    '--fmap-bspline': False,
    '--fmap-no-demean': False,
    '--use-syn-sdc': False,
    '--force-syn': False,
    '--fs-no-reconall': True,
    '--no-submm-recon': False,
    '--fs-license-file': '/usr/local/freesurfer/license.txt',
    '--resource-monitor': True,
    '--reports-only': False,
    '--run-uuid': False,
    '--write-graph': False,
    '--stop-on-first-crash': False,
    '--image': 'poldracklab/fmriprep:%s' % FMRIPREP_VERSION,
    '--notrack': True
}


@click.command(name='run_preproc', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--bids_dir', default=op.join(os.getcwd(), 'bids'), help='BIDS-directory.')
@click.option('--work_dir', default=op.join(os.getcwd(), 'work', 'fmriprep'), help='Work directory')
@click.option('--out_dir', default=None, help='output-directory.')
@click.option('--export_dir', default=None, help='Directory to export data.')
@click.option('--run_single', is_flag=True, default=True, help='Run a single subject at the time.')
@click.option('--uid', default=None, help='Run container as uid')
@click.option('--nolog', is_flag=True, default=False, help='Wheter to print instead of logging.')
@click.pass_context
def run_preproc_cmd(ctx, bids_dir, work_dir=None, out_dir=None, export_dir=None, run_single=True, uid=None, nolog=False, **fmriprep_options):
    """ CMD interface """
    fmriprep_options = extract_kwargs_from_ctx(ctx)
    run_preproc(bids_dir, work_dir, out_dir, export_dir, run_single, uid, nolog, **fmriprep_options)
    

def run_preproc(bids_dir, work_dir=None, out_dir=None, export_dir=None, run_single=True, uid=None, nolog=False, **fmriprep_options):
    """ Runs data from BIDS-directory through fmriprep pipeline.

    Parameters
    ----------
    bids_dir: str
        Absolute path to BIDS-directory
    work_dir : str
        Absolute path to work-directory
    out_dir : str
        Where to store the results (default: $bids_dir/derivatives)
    export_dir : str or None
        If string, it points to a path to copy the data to. If None,
        this is ignored
    uid : str
        User-id to run the container with
    **fmriprep_options: kwargs
        Keyword arguments of fmriprep-options
    """

    if not op.isdir(bids_dir):
        raise ValueError("%s is not an existing directory!" % bids_dir)
    
    if work_dir is None:
        work_dir = op.join(op.dirname(bids_dir), 'work', 'fmriprep')

    if not op.isdir(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    if uid is None:
        uid = str(os.getuid())
    else:
        uid = str(uid)  # make sure that it's a string

    project_name = op.basename(op.dirname(bids_dir))
    date = datetime.now().strftime("%Y-%m-%d")

    if not nolog:
        log_dir = op.join(op.dirname(op.dirname(bids_dir)), 'logs')
        log_name = op.join(log_dir, 'project-%s_stage-fmriprep_%s' % (project_name, date))

    cp_file = op.join(op.dirname(op.dirname(bids_dir)), 'CURRENT_PROJECTS.yml')
    with open(cp_file, 'r') as cpf:
        curr_projects = yaml.load(cpf)

    if project_name in curr_projects.keys():
        extra_opts = curr_projects[project_name]['fmriprep_options']
        fmriprep_options.update(extra_opts)
        if 'version' in fmriprep_options.keys():
            default_args['--image'] = 'poldracklab/fmriprep:%s' % extra_opts['version']
            del fmriprep_options['version']

    # make sure is abspath
    bids_dir = op.abspath(bids_dir)

    if out_dir is None:
        out_dir = op.join(bids_dir, 'derivatives')

    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    # Define directories + find subjects
    fmriprep_dir = op.join(out_dir, 'fmriprep')

    # Check which subjects need to be preprocessed
    bids_subs = [sd for sd in sorted(glob(op.join(bids_dir, 'sub-*'))) if op.isdir(sd)]
    to_process = []
    for bs in bids_subs:
        sub_base = op.basename(bs)
        ses_dirs = [sesd for sesd in sorted(glob(op.join(bs, 'ses-*'))) if op.isdir(sesd)]
        if ses_dirs:
            ses_proc = []
            for sesd in ses_dirs:
                this_ses = op.basename(sesd)
                fmriprep_fig_dir = op.join(fmriprep_dir, sub_base, this_ses, 'figures')
                html_file = op.join(fmriprep_dir, sub_base + '.html')
                
                if op.isdir(fmriprep_fig_dir) and op.isfile(html_file):
                #if op.isfile(html_file):
                    ses_proc.append(True)
                else:
                    ses_proc.append(False)
            if not all(ses_proc):  # not all sessions preprocessed yet!
                to_process.append(sub_base)
        else:
            fmriprep_fig_dir = op.join(fmriprep_dir, sub_base, 'figures')
            html_file = op.join(fmriprep_dir, sub_base + '.html')
            if not op.isdir(fmriprep_fig_dir) or not op.isfile(html_file):  # not processed yet!
                to_process.append(sub_base)           

    # Define subjects which need to be preprocessed
    participant_labels = [sub.split('-')[1] for sub in to_process]

    # Merge default arguments and desided arguments from user (which will)
    # overwrite default arguments
    fmriprep_options = {('--' + key): value for key, value in fmriprep_options.items()}
    default_args.update(fmriprep_options)
    all_fmriprep_options = {key: value for key, value in default_args.items() if value}

    if all_fmriprep_options['--image'].split(':')[-1] > '1.3.2':
        del all_fmriprep_options['--output-space']
        del all_fmriprep_options['--template']
        del all_fmriprep_options['--template-resampling-grid']
    else:
        del all_fmriprep_options['--output-spaces']

    options_str = [key + ' ' + str(value) for key, value in all_fmriprep_options.items()]

    # Construct command
    cmd = f'fmriprep-docker {bids_dir} {out_dir} -w {work_dir} -u {uid}:{uid} ' + ' '.join(options_str).replace(' True', '') 
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
            func_dirs = glob(op.join(bids_dir, 'sub-' + sub_label, '**', 'func'), recursive=True)
            if len(func_dirs) == 0:
                print("Setting --anat-only, because it doesn't have any functional files!")
                cmd += ' --anat-only'  # no functional files!
            
            anat_dirs = glob(op.join(bids_dir, 'sub-' + sub_label, '**', 'anat'), recursive=True)
            if len(anat_dirs) == 0:
                print("Cannot run %s, because doesn't have an anat folder!" % sub_label)
                continue

            if op.isfile(op.join(out_dir, 'freesurfer', 'sub-' + sub_label, 'scripts', 'IsRunning.lh+rh')):
                print("Skipping %s, because it's currently being preprocessed by freesurfer" % sub_label)

            print("Running participant(s): %s ..." % sub_label)
            print(cmd)

            if not nolog:  # we do want a logfile (instead of stderr/out)!
                fout = open(log_name + '_stdout.txt', 'w')
                ferr = open(log_name + '_stderr.txt', 'w') 
                subprocess.run(cmd.split(' '), stdout=fout, stderr=ferr) 
                fout.close()
                ferr.close()
            else:
                subprocess.run(cmd.split(' '))

    else:
        print('All subjects seem to have been preprocessed already!')

    # If an export-dir is defined, copy stuff to export-dir (if None, nothing
    # is copied)
    if export_dir is not None:
        export_dir_deriv = op.join(export_dir, 'bids', 'derivatives')
        if not op.isdir(export_dir_deriv):
            os.makedirs(export_dir_deriv)

        for output_type in ['fmriprep', 'freesurfer']:
            local_dir_deriv_otype = op.join(out_dir, output_type)
            
            if not op.isdir(local_dir_deriv_otype):
                continue

            export_dir_deriv_otype = op.join(export_dir_deriv, output_type)
            
            if not op.isdir(export_dir_deriv_otype):
                os.makedirs(export_dir_deriv_otype)

            to_copy = sorted(glob(op.join(local_dir_deriv_otype, '*')))
            for src in to_copy:
                dst = op.join(export_dir_deriv_otype, op.basename(src))
                if not op.exists(dst):
                    if op.isfile(src):
                        shutil.copyfile(src, dst)
                    elif op.isdir(src):
                        shutil.copytree(src, dst)
