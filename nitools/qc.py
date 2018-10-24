import click
import os
import os.path as op
from glob import glob
import shutil
import subprocess
import yaml
from .utils import extract_kwargs_from_ctx


default_args = {
    '--n_procs': 10,
    '--verbose-reports': False,
    '--no-sub': True,
    '--verbose': True,
    '--ica': False,
    '--hmc-afni': True,
    '--hmc-fsl': False,
    '--fft-spikes-detector': False,
    '--fd_thres': 0.2,
    '--ants-nthreads': 4,
    '--deoblique': False,
    '--despike': False,
    '--correct-slice-timing': False

}


@click.command(name='run_qc', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--bids_dir', default=os.getcwd(), help='BIDS-directory.')
@click.option('--out_dir', default=None, help='output-directory.')
@click.option('--export_dir', default=None, help='Directory to export data.')
@click.option('--run_single', is_flag=True, default=True, help='Whether to run a single subject at once')
@click.option('--run_group', default=True, help='Whether to run qc-group.')
@click.pass_context
def run_qc_cmd(ctx, bids_dir, out_dir=None, export_dir=None, run_single=True, run_group=True, **mriqc_options):
    """ Run qc cmd interface """
    mriqc_options = extract_kwargs_from_ctx(ctx)
    run_qc(bids_dir, out_dir, export_dir, run_single, run_group, **mriqc_options)


def run_qc(bids_dir, out_dir=None, export_dir=None, run_single=True, run_group=True, **mriqc_options):
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

    cp_file = op.join(op.dirname(__file__), 'data', 'CURRENT_PROJECTS.yml')
    with open(cp_file, 'r') as cpf:
        curr_projects = yaml.load(cpf)
    
    par_dir = op.basename(op.dirname(bids_dir))
    if par_dir in curr_projects.keys():
        mriqc_options.update(curr_projects[par_dir]['mriqc_options'])

    # make sure is abspath
    bids_dir = op.abspath(bids_dir)

    if out_dir is None:
        out_dir = op.join(op.dirname(bids_dir), 'qc')

    out_dir = op.abspath(out_dir)

    # Define directories + find subjects which need to be processed
    subs_done = [op.basename(s).split('.html')[0].split('_')[0]
                 for s in sorted(glob(op.join(out_dir, '*html')))]
    bids_subs = [op.basename(f) for f in sorted(glob(op.join(bids_dir, 'sub*')))]

    # Which subjects aren't processed yet?
    participant_labels = [sub.split('-')[1]
                          for sub in bids_subs if sub not in subs_done]

    # Merge default options and user-defined options
    mriqc_options = {('--' + key): value for key, value in mriqc_options.items()}
    default_args.update(mriqc_options)
    all_mriqc_options = {key: value for key, value in default_args.items() if value}
    options_str = [key + ' ' + str(value) for key, value in all_mriqc_options.items()]

    cmd = f'docker run -it --rm -v {bids_dir}:/data:ro -v {out_dir}:/out poldracklab/mriqc:latest /data /out participant ' + ' '.join(options_str).replace(' True', '')
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
            fout = open(op.join(op.dirname(out_dir), 'mriqc_stdout.txt'), 'a+')
            ferr = open(op.join(op.dirname(out_dir), 'mriqc_stderr.txt'), 'a+')
            subprocess.run(cmd.split(' '), stdout=fout, stderr=ferr)
            fout.close()
            ferr.close()
    else:
        print("All subjects seem to have been QC'ed already!")

    if run_group:
        cmd = f'docker run -it --rm -v {bids_dir}:/data:ro -v {out_dir}:/out poldracklab/mriqc:latest /data /out group'
        subprocess.run(cmd.split(' '))

    # Copy stuff back to server!
    if export_dir is not None:
        copy_dir = op.join(export_dir, 'qc')
        if not op.isdir(copy_dir):
            os.makedirs(copy_dir)

        proc_sub_data = sorted(glob(op.join(qc_dir, 'sub-*')))
        done_sub_data = [op.basename(f) for f in sorted(glob(op.join(copy_dir, 'sub-*')))]

        for f in proc_sub_data:
            if op.basename(f).split('_')[0] not in done_sub_data:
                sub_qc_dir = op.join(copy_dir, op.basename(f).split('_')[0])
                if not op.isdir(sub_qc_dir):
                    os.makedirs(sub_qc_dir)

                shutil.copyfile(f, op.join(sub_qc_dir, op.basename(f)))
