import glob
import os
for tarfile in glob.glob('data/clip/mscoco/mscoco/*.tar'):
    tar_idx = os.path.basename(tarfile)[:-4]
    to_dir = f'data/clip/mscoco/images/{tar_idx}'
    os.makedirs(to_dir, exist_ok=True)
    os.system(f'cp {tarfile} {to_dir}/')
    os.system(f'cd {to_dir} && tar xvf {os.path.basename(tarfile)}')