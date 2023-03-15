import os
import sys
import random
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from concurrent.futures import ThreadPoolExecutor

# Define the input and output directories
input_dir = 'birds'
output_dir = './resized_augmented_birds23wi'

# Define Transforms
rotate=iaa.Affine(rotate=(-30, 30),translate_percent=(0.0,0.2), scale={"x":1.3, "y":1.3})
flip_hr=iaa.Fliplr(p=0.5)
blur = iaa.AverageBlur(k=(0,3))
resize = iaa.Resize({"height": 224, "width": 224})

# Desired number of images
des_num_images = 80000
desired_samples_per_class = des_num_images // 555



# Define the function that will resize the image
def process_image(input_path, output_path, do_augment):
    # Load the image and resize it
    image = imageio.v3.imread(input_path)
    augm_image = resize.augment_image(image)
    if do_augment:   
        augm_image = rotate.augment_image(augm_image)
        augm_image = flip_hr.augment_image(augm_image)
        augm_image =blur.augment_image(augm_image)

    # Save the resized image
    imageio.imwrite(output_path, augm_image[:,:,:3])



def process_save_images(root, files, do_augment):
        for i,file in enumerate(files):
            # Check if the file is an image
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                # Define the input and output paths
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))

                if do_augment:
                  output_path = output_path.replace(file, f'augmented_{i}_'+ file)   

                # Make sure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Submit the image processing task to the thread pool
                process_image(input_path, output_path, do_augment)
            else:
                 print(file)
        



# Traverse through all the directories and subdirectories
for root, dirs, files in os.walk(input_dir):
        num_files =  len(files) 
        process_save_images(root, files, False)

        if 'train' in root and num_files > 0 and num_files < desired_samples_per_class:
            sampled_files = random.choices(files, k=(desired_samples_per_class  - num_files))
            process_save_images(root, sampled_files, True)




