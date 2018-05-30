import os
import os.path as op
from glob import glob
import shutil


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


def run_qc(bids_dir, export_dir=None, **mriqc_options):
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

    # Define directories + find subjects which need to be processed
    qc_dir = op.join(op.dirname(bids_dir), 'qc')
    reports_dir = op.join(qc_dir, 'reports')
    subs_done = [op.basename(s).split('.html')[0].split('_')[0]
                 for s in sorted(glob(op.join(reports_dir, '*html')))]
    bids_subs = [op.basename(f) for f in sorted(glob(op.join(bids_dir, 'sub*')))]

    # Which subjects aren't processed yet?
    participant_labels = [sub.split('-')[1]
                          for sub in bids_subs if sub not in subs_done]

    # Merge default options and user-defined options
    mriqc_options = {('--' + key): value for key, value in mriqc_options.items()}
    default_args.update(mriqc_options)
    all_mriqc_options = {key: value for key, value in default_args.items() if value}
    options_str = [key + ' ' + str(value) for key, value in all_mriqc_options.items()]

    # Run QC!
    if participant_labels:
        print("Running qc for participants: %s" % ' '.join(participant_labels))
        cmd = f'docker run -it --rm -v {bids_dir}:/data:ro -v {qc_dir}:/out poldracklab/mriqc:latest /data /out participant ' + ' '.join(options_str).replace(' True', '')
        cmd += ' --participant_label %s' % ' '.join(participant_labels)
        os.system(cmd)

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
