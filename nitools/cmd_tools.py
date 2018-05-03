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


env_vars = {
    'uva': dict(
        server_home='/media/lukas/goliath/spinoza_data',
        fmri_proj='/run/user/1000/gvfs/smb-share:server=fmgstorage.fmg.uva.nl,share=psychology$/fMRI Projects',
        dropbox='/run/user/1000/gvfs/smb-share:server=fmgstorage.fmg.uva.nl,share=dropbox$'
    ),
    'neuroimaging.lukas-snoek.com': dict(
        server_home='/home/lsnoek1/spinoza_data',
        fmri_proj='/run/user/1002/gvfs/smb-share:server=fmgstorage.fmg.uva.nl,share=psychology$/fMRI Projects',
        dropbox='/run/user/1002/gvfs/smb-share:server=fmgstorage.fmg.uva.nl,share=dropbox$'
    ),
    'MacBook': dict(
        server_home='/Users/lukas/spinoza_data',
        fmri_proj='/Volumes/fMRI_projects$',
        dropbox='/Volumes/dropbox$'
    )
}

hostname = socket.gethostname()
if 'MacBook' in hostname or 'vpn' in hostname:
    hostname = 'MacBook'

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
        print("Looking for data in %s" % proj_name)

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
                print("Copying data from %s to server ... " % sub, end='')
                shutil.copytree(sub, server_dir)
                print("done.")
            else:
                print("This data is already on server")

	# Also check for config.yml
        cfg_file = op.join(export_folder, 'raw', 'config.yml')
        if op.isfile(cfg_file):
            print("Copying config file to server ...")
            shutil.copy2(cfg_file, op.join(proj_dir, 'raw'))
            cfg_file = op.join(proj_dir, 'raw', 'config.yml')

        # Then bidsify everything
        print("Running bidsify ...")
        if op.isfile(cfg_file):
            this_cfg = cfg_file
        else:
            this_cfg = op.join(op.dirname(bidsify.__file__), 'data', 'spinoza_cfg.yml')
        
        run_bidsify(cfg_path=this_cfg, directory=op.join(proj_dir, 'raw'), validate=True)

        if settings['preproc']:
            print("Running fmriprep ...")
            run_preproc(bids_dir=op.join(proj_dir, 'bids'), export_dir=export_folder,
                        **settings['fmriprep_options'])

        if settings['qc']:
            print("Running mriqc ...")
            run_qc(bids_dir=op.join(proj_dir, 'bids'), export_dir=export_folder,
                   **settings['mriqc_options'])
