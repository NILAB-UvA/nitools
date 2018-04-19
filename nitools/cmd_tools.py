import yaml
import socket
import warnings
import shutil
import os.path as op
import bidsify
from glob import glob
from bidsify import bidsify as run_bidsify
from .preproc import run_preproc
from .qc import run_qc

TEST = True

env_vars = {
    'uva': dict(
        server_home='/media/lukas/goliath/spinoza_data',
        fmri_proj='/run/user/1000/gvfs/smb-share:server=bigbrother.fmg.uva.nl,share=fmri_projects$',
        dropbox='/run/user/1000/gvfs/smb-share:server=bigbrother.fmg.uva.nl,share=dropbox$'
    ),
    'neuroimaging.lukas-snoek.com': dict(
        server_home='/media/lukas/goliath/spinoza_data',
        fmri_proj='/run/user/1002/gvfs/smb-share:server=bigbrother.fmg.uva.nl,share=fmri_projects$',
        dropbox='/run/user/1002/gvfs/smb-share:server=bigbrother.fmg.uva.nl,share=dropbox$'
    ),
    'MacBook': dict(
        server_home='/Users/lukas/spinoza_data',
        fmri_proj='/Volumes/fMRI_projects$',
        dropbox='/Volumes/dropbox$'
    ),
    'TEST': dict(
        server_home='/Users/lukas/spinoza_data',
        fmri_proj='/Users/lukas/spinoza_data/mock_fmri_proj',
        dropbox='/Users/lukas/spinoza_data/mock_dropbox'
    )
}

hostname = socket.gethostname()
if 'MacBook' in hostname or 'vpn' in hostname:
    hostname = 'MacBook'
elif TEST:
    hostname = 'TEST'

env = env_vars[hostname]


def run_qc_and_preproc():
    """ Main function to run qc and preprocessing of Spinoza Centre (REC)
    data. """

    # Open file with current-projects
    cp_file = op.join(op.dirname(__file__), 'data', 'CURRENT_PROJECTS.yml')
    with open(cp_file, 'r') as cpf:
        curr_projects = yaml.load(cpf)

    # Loop over projects
    for proj_name, settings in curr_projects.items():
        print("Fetching data from %s" % proj_name)

        export_folder = settings['export_folder']
        if 'fMRI Project' in export_folder:
            export_folder = op.join(env['fmri_proj'], export_folder)
        else:
            export_folder = op.join(env['dropbox'], export_folder)

        if not op.isdir(export_folder):
            warnings.warn("The export-folder '%s' doesn't seem to"
                          " exist!" % export_folder)

        proj_dir = op.join(env['server_home'], proj_name)
        # Check for raw subjects
        if settings['multiple_sessions']:
            search_str = op.join(export_folder, 'raw', 'sub-*', 'ses-*')
            subs_in_raw = sorted(glob(search_str))
            print("subjects in raw: %s" % ([f.split('/')[-2:] for f in subs_in_raw],))
        else:
            subs_in_raw = sorted(glob(op.join(export_folder, 'raw', 'sub-*')))
            print("subjects in raw: %s" % ([op.basename(f) for f in subs_in_raw],))

        for sub in subs_in_raw:  # First copy data to server (if necessary)

            if settings['multiple_sessions']:
                ses_idf = op.basename(sub)
                sub_idf = op.basename(op.dirname(sub))
                server_dir = op.join(proj_dir, 'raw', sub_idf, ses_idf)
            else:
                sub_idf = op.basename(sub)
                server_dir = op.join(proj_dir, 'raw', sub_idf)

            if not op.isdir(server_dir):
                shutil.copytree(sub, server_dir)
            else:
                print("This data is already on server")

        # Then bidsify everything
        spinoza_cfg = op.join(op.dirname(bidsify.__file__), 'data', 'spinoza_cfg.yml')
        run_bidsify(cfg_path=spinoza_cfg, directory=op.join(proj_dir, 'raw'), validate=True)

        if settings['qc']:
            run_qc(directory=op.join(proj_dir, 'bids'))

        if settings['preproc']:
            run_preproc(directory=op.join(proj_dir, 'bids'),
                        **settings['fmriprep_options'])
