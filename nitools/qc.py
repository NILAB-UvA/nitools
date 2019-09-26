import click
import os
import shutil
import subprocess
import yaml
import os.path as op
from datetime import datetime
from glob import glob
from .utils import extract_kwargs_from_ctx


default_args = {
    '--n_procs': 10,
    '--verbose-reports': False,
    '--no-sub': True,
    '--verbose': True,
    '--ica': False,
    '--hmc-afni': True,
    '--hmc-fsl': False,
    '--fft-spikes-detector': True,
    '--fd_thres': 0.2,
    '--ants-nthreads': 4,
    '--deoblique': False,
    '--despike': True,
    '--correct-slice-timing': False
}


@click.command(name='run_qc', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--bids_dir', default=os.getcwd(), help='BIDS-directory.')
@click.option('--work_dir', default=op.join(os.getcwd(), 'work', 'mriqc'), help='Work-directory')
@click.option('--out_dir', default=None, help='output-directory.')
@click.option('--export_dir', default=None, help='Directory to export data.')
@click.option('--run_single', is_flag=True, default=True, help='Whether to run a single subject at once')
@click.option('--run_group', default=True, help='Whether to run qc-group.')
@click.option('--uid', default=None, help='Run container as uid')
@click.option('--nolog', is_flag=True, help='Use stderr/out instead of log')
@click.pass_context
def run_qc_cmd(ctx, bids_dir, work_dir=None, out_dir=None, export_dir=None, run_single=True, run_group=True, uid=None, nolog=False, **mriqc_options):
    """ Run qc cmd interface """
    mriqc_options = extract_kwargs_from_ctx(ctx)
    run_qc(bids_dir, out_dir, export_dir, run_single, run_group, uid, nolog, **mriqc_options)


def run_qc(bids_dir, work_dir=None, out_dir=None, export_dir=None, run_single=True, run_group=True, uid=None, nolog=False, **mriqc_options):
    """ Runs data from BIDS-directory through the MRIQC pipeline.

    Parameters
    ----------
    bids_dir: str
        Absolute path to BIDS-directory
    subs: list of str
        List of subject-identifiers (e.g., sub-0001) which need to be run
        through the pipeline
    **mriqc_options: kwargs
        Keyword arguments of mriqc-options
    """

    from nitools.version import MRIQC_VERSION

    if not op.isdir(bids_dir):
        raise ValueError(f"Bids-dir {bids_dir} doesn't exist!")

    if work_dir is None:
        work_dir = op.join(op.dirname(bids_dir), 'work', 'mriqc')

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
        log_name = op.join(log_dir, 'project-%s_stage-mriqc_%s' % (project_name, date))

    cp_file = op.join(op.dirname(op.dirname(bids_dir)), 'CURRENT_PROJECTS.yml')
    with open(cp_file, 'r') as cpf:
        curr_projects = yaml.load(cpf)
    
    if project_name in curr_projects.keys():
        extra_opts = curr_projects[project_name]['mriqc_options']
        mriqc_options.update(extra_opts)
        if 'version' in mriqc_options.keys():  # override default
            MRIQC_VERSION = extra_opts['version']
            del mriqc_options['version']

    # make sure is abspath
    bids_dir = op.abspath(bids_dir)

    if out_dir is None:
        out_dir = op.join(bids_dir, 'derivatives', 'mriqc')

    out_dir = op.abspath(out_dir)

    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    bids_subs = [sd for sd in sorted(glob(op.join(bids_dir, 'sub-*'))) if op.isdir(sd)]
    to_process = []
    for bs in bids_subs:
        sub_base = op.basename(bs)
        ses_dirs = [sesd for sesd in sorted(glob(op.join(bs, 'ses-*'))) if op.isdir(sesd)]
        if ses_dirs:
            ses_proc = []
            for sesd in ses_dirs:
                this_ses = op.basename(sesd)
                if op.isdir(op.join(out_dir, sub_base, this_ses)):
                    ses_proc.append(True)
                else:
                    ses_proc.append(False)
            if not all(ses_proc):  # not all sessions preprocessed yet!
                to_process.append(sub_base)
        else:
            this_dir = op.join(out_dir, sub_base)
            if not op.isdir(this_dir):  # not processed yet!
                to_process.append(sub_base)

    # Define subjects which need to be preprocessed
    participant_labels = [sub.split('-')[1] for sub in to_process]

    # Merge default options and user-defined options
    mriqc_options = {('--' + key): value for key, value in mriqc_options.items()}
    default_args.update(mriqc_options)
    all_mriqc_options = {key: value for key, value in default_args.items() if value}
    options_str = [key + ' ' + str(value) for key, value in all_mriqc_options.items()]

    cmd = f'docker run --rm -u {uid}:{uid} -v {bids_dir}:/data:ro -v {out_dir}:/out -v {work_dir}:/scratch poldracklab/mriqc:{MRIQC_VERSION} /data /out -w /scratch participant ' + ' '.join(options_str).replace(' True', '')
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
            
            if not nolog:
                fout = open(log_name  + '_stdout.txt', 'w')
                ferr = open(log_name + '_stderr.txt', 'w')
                subprocess.run(cmd.split(' '), stdout=fout, stderr=ferr)
                fout.close()
                ferr.close()
            else:
                subprocess.run(cmd.split(' '))
    
    else:
        print("All subjects seem to have been QC'ed already!")

    if run_group:

        cmd = f'docker run --rm -u {uid} -v {bids_dir}:/data:ro -v {out_dir}:/out poldracklab/mriqc:{MRIQC_VERSION} /data /out group'
        if not nolog:
            fout = open(log_name.replace('mriqc', 'mriqcGroup') + '_stdout.txt', 'w')
            ferr = open(log_name.replace('mriqc', 'mriqcGroup') + '_stderr.txt', 'w')
            subprocess.run(cmd.split(' '), stdout=fout, stderr=ferr)
        else:
            subprocess.run(cmd.split(' '))

    # Copy stuff back to server!
    if export_dir is not None:
        export_dir_mriqc = op.join(export_dir, 'bids', 'derivatives', 'mriqc')
        if not op.isdir(export_dir_mriqc):
            os.makedirs(export_dir_mriqc)

        to_copy = sorted(glob(op.join(out_dir, '*')))

        for src in to_copy:
            dst = op.join(export_dir_mriqc, op.basename(src))
            if not op.exists(dst) or 'group_' in op.basename(src):
                if op.isfile(src):
                    shutil.copyfile(src, dst)
                elif op.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    pass
