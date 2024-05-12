import os
import argparse

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--save_folder', type=str, default='./tmp',
    help='path for saving rendered image')
parser.add_argument(
    '--dataset_folder', type=str, default='./tmp',
    help='path for downloaded 3d dataset folder')
parser.add_argument(
    '--blender_root', type=str, default='./tmp',
    help='path for blender')
parser.add_argument(
    '--shapenet_version', type=str, default='2',
)
args = parser.parse_args()

save_folder = args.save_folder
dataset_folder = args.dataset_folder
blender_root = args.blender_root
shapenet_version = args.shapenet_version

synset_list = [
    '02958343',  # Car
]
scale_list = [
    0.9,
    # 0.7,
    # 0.9
]

if shapenet_version == '2':
    path_list = [os.path.join(dataset_folder, synset) for synset in synset_list]
    for obj_scale, dataset_folder in zip(scale_list, path_list):
        file_list = sorted(os.listdir(os.path.join(dataset_folder)))
        for file in file_list:
            models_path = os.path.join(dataset_folder, file, 'models')
            # check if models folder exists
            if os.path.exists(models_path):
                # move all files in 'models' to parent directory
                os.system('mv ' + os.path.join(models_path, '*') + ' ' + os.path.join(dataset_folder, file))
                # remove 'models' directory
                os.system('rm -rf ' + models_path)
            
            material_file = os.path.join(dataset_folder, file, 'model_normalized.mtl')
            try:
                with open(material_file, 'r') as f:
                    material_file_text = f.read()
                material_file_text = material_file_text.replace('../images', './images')
                
                with open(material_file, 'w') as f:
                    f.write(material_file_text)
            except FileNotFoundError:
                print("Warning: File not found", material_file)
                continue
            
            
            render_folder = os.path.join(save_folder,"img","02958343",file)
            print(render_folder)
            if not os.path.exists(render_folder) or len(os.listdir(render_folder)) < 25:
                print(render_folder, "does not exist or has less than 24 images. Rendering...")
                render_cmd = '%s -b -P render_shapenet.py -- --output %s %s --scale %f --views 24 --resolution 1024 >> tmp.out' % (
                    blender_root, save_folder, os.path.join(dataset_folder, file, 'model_normalized.obj'), obj_scale
                )
                os.system(render_cmd)
            else:
                print(render_folder, "already exists and has 24 images. Skipping...")

