import os
import yaml
import socket
import warnings
import shutil
import os.path as op
import bidsify
import subprocess
import click
import logging
import sys
from glob import glob
from datetime import datetime
from bidsify import bidsify as run_bidsify
from bidsify.docker import run_from_docker as run_bidsify_docker
from .preproc import run_preproc
from .qc import run_qc

# Define some "environment-variables"
env_vars = {
    'uva': dict(
        server_home='/media/lukas/goliath/spinoza_data',
        fmri_proj='/run/user/1000/gvfs/smb-share:server=fmgstorage.fmg.uva.nl,share=psychology$/fMRI Projects',
        dropbox='/run/user/1000/gvfs/smb-share:server=bigbrother.fmg.uva.nl,share=dropbox$'
    ),
    'neuroimaging.lukas-snoek.com': dict(
        server_home='/home/lsnoek1/spinoza_data',
        fmri_proj='/mnt/lsnoek1/fmgstorage_share/fMRI Projects',
        dropbox='/mnt/lsnoek1/dropbox_share/fMRI Proejcts'
    ),
    'MacBook': dict(
        server_home='/Users/lukas/spinoza_data',
        fmri_proj='/Volumes/fMRI_projects$',
        dropbox='/Volumes/dropbox$'
    )
}

# Check on which platform this is running (work desktop, pers. laptop, server)
hostname = socket.gethostname()
if 'MacBook' in hostname or 'vpn' in hostname:
    hostname = 'MacBook'

env = env_vars[hostname]


@click.command(name='run_qc_and_preproc')
@click.option('--project', default=None, type=str, help='Run for specific project?')
@click.option('--docker', is_flag=True, help='Run docker?')
def run_qc_and_preproc(project=None, docker=False):
    """ Main function to run qc and preprocessing of Spinoza Centre (REC)
    data. """

    # Open file with currently running projects
    cp_file = op.join(op.dirname(__file__), 'data', 'CURRENT_PROJECTS.yml')
    with open(cp_file, 'r') as cpf:
        curr_projects = yaml.load(cpf)
   
    date = datetime.now().strftime("%Y-%m-%d")
    log_dir = op.join(env['server_home'], 'logs')
    log_file = op.join(log_dir, 'nitools_%s.txt' % date)
    formatter = logging.Formatter('%(message)s')

    logger = logging.getLogger("nitools-logger")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)    
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info("Starting nitools at %s" % str(datetime.now()))

    # Loop over projects
    for proj_name, settings in curr_projects.items():

        if project is not None:
            if project != proj_name:
                continue
        else:
            if not settings['run_automatically']:
                continue
    
        logger.info("\n======== PROCESSING DATA FROM PROJECT %s ========" % proj_name)

        export_folder = settings['export_folder']
        if 'fMRI Project' in export_folder:
            export_folder = op.join(env['fmri_proj'], export_folder)
        else:
            export_folder = op.join(env['dropbox'], export_folder)

        if not op.isdir(export_folder):
            msg = "The export-folder '%s' doesn't seem to exist!" % export_folder
            logger.error(msg)
            raise ValueError(msg)

        proj_dir = op.join(env['server_home'], proj_name)
        # Check for raw subjects
        if settings['multiple_sessions']:
            search_str = op.join(export_folder, 'raw', 'sub-*', 'ses-*')
            subs_in_raw = sorted(glob(search_str))
        else:
            subs_in_raw = sorted(glob(op.join(export_folder, 'raw', 'sub-*')))

        for sub in subs_in_raw:  # First copy data to server (if necessary)

            if settings['multiple_sessions']:
                ses_idf = op.basename(sub)
                sub_idf = op.basename(op.dirname(sub))
                server_dir = op.join(proj_dir, 'raw', sub_idf, ses_idf)
            else:
                sub_idf = op.basename(sub)
                server_dir = op.join(proj_dir, 'raw', sub_idf)

            if not op.isdir(server_dir):
                n_files = len(os.listdir(sub))
                if n_files > 0:
                    logger.info("Copying data from %s to server ..." % sub_idf)
                    shutil.copytree(sub, server_dir)
            else:
                pass

        # Also check for config.yml
        cfg_file = op.join(export_folder, 'raw', 'config.yml')
        if op.isfile(cfg_file):
            logger.info("Copying config file to server ...")
            shutil.copy2(cfg_file, op.join(proj_dir, 'raw'))
            cfg_file = op.join(proj_dir, 'raw', 'config.yml')

        # Then bidsify everything
        logger.info("\n-------- RUNNING BIDSIFY --------")
        if op.isfile(cfg_file):
            this_cfg = cfg_file
        else:
            this_cfg = op.join(op.dirname(bidsify.__file__), 'data',
                               'spinoza_cfg.yml')
        raw_dir = op.join(proj_dir, 'raw')
        bidsignore_file = op.join(raw_dir, '.bidsignore')

        if not op.isfile(bidsignore_file):
            with open(bidsignore_file, 'w') as big:
                big.write('**/*.log\n**/*phy')

        if docker:
            run_bidsify_docker(cfg_path=this_cfg, directory=raw_dir,
                               validate=True, out_dir=op.join(proj_dir, 'bids'), spinoza=True)
        else:
            run_bidsify(cfg_path=this_cfg, directory=raw_dir,
                        validate=True, out_dir=op.join(proj_dir, 'bids'))

        # Copy stuff to server
        bids_export_folder = op.join(export_folder, 'bids')
        if not op.isdir(bids_export_folder):
            os.makedirs(bids_export_folder)

        bids_out_dir = op.join(proj_dir, 'bids')
        bids_files_on_server = sorted(glob(op.join(bids_out_dir, 'sub-*')))
        for sub in bids_files_on_server:
            sub_name = op.basename(sub)
            if not op.isdir(op.join(bids_export_folder, sub_name)):
                logger.info('Copying files from %s to export-dir ...' % sub)
                shutil.copytree(sub, op.join(bids_export_folder, sub_name))
        
        participants_file = op.join(bids_out_dir, 'participants.tsv')
        shutil.copyfile(participants_file, op.join(bids_export_folder, op.basename(participants_file)))
        dataset_descr_file = op.join(bids_out_dir, 'dataset_description.json')
        shutil.copyfile(dataset_descr_file, op.join(bids_export_folder, op.basename(dataset_descr_file)))
        
        if settings['preproc']:
            logger.info("\n-------- RUNNING FMRIPREP --------")
            run_preproc(bids_dir=op.join(proj_dir, 'bids'),
                        export_dir=export_folder,
                        **settings['fmriprep_options'])

        if settings['qc']:
            logger.info("\n-------- RUNNING MRIQC --------")
            run_qc(bids_dir=op.join(proj_dir, 'bids'),
                   export_dir=export_folder, **settings['mriqc_options'])
