import os
import os.path as op
from glob import glob

default_args = {
    '--n_procs': 10,
    '--verbose-reports': False,
    '--no-sub': True,
    '--verbose': False,
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


def run_qc(bids_dir, **mriqc_options):
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

    out_dir = op.join(op.dirname(bids_dir), 'qc')
    qc_dir = op.join(out_dir, 'reports')
    subs_done = [op.basename(s).split('.html')[0].split('_')[0]
                 for s in sorted(glob(op.join(qc_dir, '*html')))]
    bids_subs = [op.basename(f) for f in sorted(glob(op.join(bids_dir, 'sub*')))]
    participant_labels = [sub.split('-')[1]
                          for sub in bids_subs if sub not in subs_done]

    mriqc_options = {('--' + key): value for key, value in mriqc_options.items()}
    default_args.update(mriqc_options)
    mriqc_options = {key: value for key, value in default_args.items() if value}
    options_str = [key + ' ' + str(value) for key, value in mriqc_options.items()]
    cmd = f'docker run -it --rm -v {bids_dir}:/data:ro -v {out_dir}:/out poldracklab/mriqc:latest /data /out participant ' + ' '.join(options_str).replace(' True', '')
    cmd += ' --participant_label %s' % ' '.join(participant_labels)
    #print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    import yaml
    cp_file = op.join(op.dirname(__file__), 'data', 'CURRENT_PROJECTS.yml')
    with open(cp_file, 'r') as cpf:
        curr_projects = yaml.load(cpf)

    run_qc('/media/lukas/data/spinoza-rec/spinoza_testdata/SpinozaTest/bids', **curr_projects['SpinozaTest']['mriqc_options'])
