from spynoza.glm.workflows import create_modelgen_workflow
from nipype.interfaces.io import SelectFiles
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline import Workflow, Node
from nipype.interfaces.fsl.model import FEAT
import os.path as op
from glob import glob
from .nodes import ConcatenateIterables, Combine_events_and_confounds, Custom_Level1design_Feat


tasks = ['other', 'self_run-1', 'self_run-2']
base_dir = '/home/lsnoek1/SharedStates'
out_dir = op.join(base_dir, 'firstlevel')
sub_ids = sorted([op.basename(f) for f in glob(op.join(base_dir, 'preproc', 'fmriprep', 'sub-???'))])

meta_wf = Workflow('firstlevel_spynoza')

concat_iterables_node = Node(interface=ConcatenateIterables(fields=['sub_id', 'task']),
                             name='concat_iterables')

input_node = Node(IdentityInterface(fields=['sub_id', 'task']), name='inputspec')
input_node.iterables = [('sub_id', sub_ids), ('task', tasks)]

meta_wf.connect(input_node, 'sub_id', concat_iterables_node, 'sub_id')
meta_wf.connect(input_node, 'task', concat_iterables_node, 'task')

templates = {'func': '{sub_id}/func/{sub_id}_task-{task}*_preproc.nii.gz',
             'func_mask': '{sub_id}/func/{sub_id}_task-{task}*_brainmask.nii.gz',
             'T1': '{sub_id}/anat/*preproc.nii.gz',
             'events': 'LOGS/{sub_id}_task-{task}_events.tsv',
             'confounds': '{sub_id}/func/{sub_id}_task-{task}*_confounds.tsv'}

select_files = Node(SelectFiles(templates=templates), name='selectfiles')

select_files.inputs.base_directory = op.join(base_dir, 'preproc', 'fmriprep')
select_files.inputs.raise_on_empty = False
select_files.inputs.sort_filelist = True

meta_wf.connect(input_node, 'sub_id', select_files, 'sub_id')
meta_wf.connect(input_node, 'task', select_files, 'task')

modelgen_wf = create_modelgen_workflow(skip_specify_model=True)
modelgen_wf.inputs.inputspec.sort_by_onset = True
modelgen_wf.inputs.inputspec.TR = 2
modelgen_wf.inputs.inputspec.extend_motion_pars = False
modelgen_wf.inputs.inputspec.exclude = None
modelgen_wf.inputs.inputspec.hp_filter = 100
modelgen_wf.inputs.inputspec.single_trial = True
modelgen_wf.inputs.inputspec.which_confounds = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']

meta_wf.connect(select_files, 'events', modelgen_wf, 'inputspec.events_file')
meta_wf.connect(select_files, 'confounds', modelgen_wf, 'inputspec.confound_file')
meta_wf.connect(select_files, 'func', modelgen_wf, 'inputspec.func_file')

feat_node = Node(interface=Custom_Level1design_Feat, name='feat')
feat_node.inputs.smoothing = 5
feat_node.inputs.temp_deriv = False

feat_node.inputs.registration = 'none'
feat_node.inputs.slicetiming = 'no'
feat_node.inputs.motion_correction = 0
feat_node.inputs.bet = True
feat_node.inputs.prewhitening = False
feat_node.inputs.motion_regression = False
feat_node.inputs.thresholding = 'uncorrected'
feat_node.inputs.hrf = 'doublegamma'
feat_node.inputs.open_feat_html = False
feat_node.inputs.highpass = 100
feat_node.inputs.contrasts = 'single-trial'

meta_wf.connect(select_files, 'func', feat_node, 'func_file')
meta_wf.connect(modelgen_wf, 'outputspec.session_info', feat_node, 'session_info')
meta_wf.connect(input_node, 'sub_id', feat_node, 'output_dirname')
meta_wf.connect(select_files, 'func_mask', feat_node, 'mask')

run_feat_node = Node(FEAT(), name='run_feat')

meta_wf.connect(feat_node, 'feat_dir', run_feat_node, 'fsf_file')

datasink = Node(interface=DataSink(), name='datasink')
datasink.inputs.parameterization = False
datasink.inputs.base_directory = out_dir
meta_wf.connect(feat_node, 'confound_file', datasink, 'confound_file')
meta_wf.connect(feat_node, 'ev_files', datasink, 'ev_files')
meta_wf.connect(run_feat_node, 'feat_dir', datasink, 'firstlevel_FEAT')

meta_wf.connect(concat_iterables_node, 'out', datasink, 'container')

meta_wf.base_dir = op.join(out_dir, 'workingdir')
meta_wf.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
