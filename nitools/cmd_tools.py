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
import joblib as jl
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
        dropbox='/mnt/lsnoek1/dropbox_share/fMRI Projects'
    )
}

# Check on which platform this is running (work desktop, pers. laptop, server)
hostname = socket.gethostname()
ENV = env_vars[hostname]
UID = '1002'

@click.command(name='run_qc_and_preproc')
@click.option('--project', default=None, type=str, help='Run for specific project?')
@click.option('--docker', is_flag=True, help='Run docker?')
@click.option('--n-cpus', default=6, help='Number of CPUs to use in parallel')
def run_qc_and_preproc(project=None, docker=False, n_cpus=6):
    """ Main function to run qc and preprocessing of Spinoza Centre (REC)
    data. """

    # Open file with currently running projects
    cp_file = op.join(ENV['server_home'], 'CURRENT_PROJECTS.yml')
    with open(cp_file, 'r') as cpf:
        curr_projects = yaml.load(cpf)
   
    return_codes = jl.Parallel(n_jobs=n_cpus)(jl.delayed(_run_project)
            (proj_name, settings, project, docker) for proj_name, settings in curr_projects.items()
    )


def _run_project(proj_name, settings, project, docker):

    if project is not None:
        if project != proj_name:
            return None  # skip if not the supplied project name
    else:
        if not settings['run_automatically']:
            return None  # only run if indicated to do so

    export_folder = settings['export_folder']
    if 'fMRI Project' in export_folder:
        export_folder = op.join(ENV['fmri_proj'], export_folder)
    else:
        export_folder = op.join(ENV['dropbox'], export_folder)

    if not op.isdir(export_folder):
        msg = "The export-folder '%s' doesn't seem to exist!" % export_folder
        raise ValueError(msg)

    proj_dir = op.join(ENV['server_home'], proj_name)
    if not op.isdir(proj_dir):
        os.makedirs(proj_dir)

    still_running_f = op.join(proj_dir, 'StillRunning.txt')
    if op.isfile(still_running_f):
        print('Project %s is still running!' % proj_name)
        return None
    else:
        with open(still_running_f, 'w') as f_out:
            f_out.write('Started %s' % datetime.now())

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
                print("Copying data from %s to server ..." % sub_idf)
                shutil.copytree(sub, server_dir)
        else:
            pass

    # Also check for config.yml
    cfg_file = op.join(export_folder, 'raw', 'config.yml')
    if op.isfile(cfg_file):
        print("Copying config file to server ...")
        shutil.copy2(cfg_file, op.join(proj_dir, 'raw'))
        cfg_file = op.join(proj_dir, 'raw', 'config.yml')

    # Then bidsify everything
    print("\n-------- RUNNING BIDSIFY FOR %s --------" % proj_name)
    if op.isfile(cfg_file):
        this_cfg = cfg_file
    else:
        this_cfg = op.join(op.dirname(bidsify.__file__), 'data',
                           'spinoza_cfg.yml')
    raw_dir = op.join(proj_dir, 'raw')
    bids_dir = op.join(proj_dir, 'bids')
    if not op.isdir(bids_dir):
        os.makedirs(bids_dir)

    bidsignore_file = op.join(bids_dir, '.bidsignore')
    if not op.isfile(bidsignore_file):
        with open(bidsignore_file, 'w') as big:
            big.write('**/*.log\n**/*phy\nbids_validator_log.txt\nunallocated\nwork')

    if docker:
        run_bidsify_docker(cfg_path=this_cfg, directory=raw_dir,
                           validate=True, out_dir=op.join(proj_dir, 'bids'), spinoza=True,
                           uid=UID)  # this should not be hard-coded!
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
            
            print('Copying bidsified files from %s to export-dir ...' % sub)
            shutil.copytree(sub, op.join(bids_export_folder, sub_name))
    
    participants_file = op.join(bids_out_dir, 'participants.tsv')
    shutil.copyfile(participants_file, op.join(bids_export_folder, op.basename(participants_file)))
    dataset_descr_file = op.join(bids_out_dir, 'dataset_description.json')
    if not op.isfile(dataset_descr_file):
        shutil.copyfile(dataset_descr_file,
                        op.join(bids_export_folder, op.basename(dataset_descr_file)))

    if settings['preproc']:
        fp_workdir = op.join(bids_out_dir, 'work', 'fmriprep')
        if not op.isdir(fp_workdir):  # otherwise it's created as root!
            os.makedirs(fp_workdir, exist_ok=True)

        print("\n-------- RUNNING FMRIPREP FOR %s --------" % proj_name)
        run_preproc(bids_dir=op.join(proj_dir, 'bids'),
                    export_dir=export_folder, uid=UID,
                    **settings['fmriprep_options'])

    if settings['qc']:
        qc_workdir = op.join(bids_out_dir, 'work', 'mriqc')
        if not op.isdir(qc_workdir):
            os.makedirs(qc_workdir)

        print("\n-------- RUNNING MRIQC FOR %s --------" % proj_name)
        run_qc(bids_dir=op.join(proj_dir, 'bids'), uid=UID,
                export_dir=export_folder, **settings['mriqc_options'])

    if op.isfile(still_running_f):
        os.remove(still_running_f)


@click.command(name='start_nitools')
def start_nitools():
    """ Start nitools service """
    sr_files = glob(op.join(ENV['server_home'], '*', 'StillRunning.txt'))
    for sr_file in sr_files:
        print(f"Removing {sr_file}!")
        #os.remove(sr_file)

    cmd = "sudo mount -t cifs -o username=lsnoek1,uid=1002,gid=1002 //fmgstorage.fmg.uva.nl/psychology$ /mnt/lsnoek1/fmgstorage_share/"
    subprocess.run(cmd.split(' '))
    cmd = "sudo mount -t cifs -o username=lsnoek1,uid=1002,gid=1002 //bigbrother.fmg.uva.nl/dropbox$ /mnt/lsnoek1/dropbox_share/"
    subprocess.run(cmd.split(' '))
