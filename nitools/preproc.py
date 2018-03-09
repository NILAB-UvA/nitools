import os
import os.path as op
import click
from glob import glob


@click.command()
@click.option('--bids', help='Bids-directory')
@click.option('--out', help='Out directory')
@click.option('--freesurfer', is_flag=True, help='Use freesurfer')
@click.option('--stc', is_flag=True, help='Do slicetiming correction')
@click.option('--cores', default=10, help='How many cores?')
@click.option('--omp', default=4, help='How many OMP cores?')
@click.option('--nsub', default=20, help='Number of participants to run')
def run(bids, out, freesurfer, stc, cores, omp, nsub):

    cmd = ('fmriprep-docker %s %s%s%s--write-graph --nthreads %i --omp-nthreads %i '
           '--use-aroma --ignore-aroma-denoising-errors')
    cmd = cmd % (bids, out, '' if freesurfer else ' --no-freesurfer ', '' if stc else ' --ignore slicetiming ', cores, omp)

    subs_done = [s.split('.html')[0] for s in sorted(glob(op.join(out, 'fmriprep', '*html')))]
    bids_subs = sorted(glob(op.join(bids, 'sub*')))
    to_process = [sub.split('-')[1] for sub in bids_subs if sub not in subs_done]
    cmd = cmd + ' --participant-label %s' % ' '.join(to_process[:nsub])

    os.system(cmd)

if __name__ == '__main__':

    run()

