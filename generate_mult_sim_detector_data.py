import os
import random
import numpy as np
from PIL import Image, ImageDraw
import math
import scipy.ndimage
import csv
from tqdm import tqdm  # For the progress bar

##note barrel causes the line to not enter/exit on edges

# ----------------------------
# distortion_model.py content
# ----------------------------

def distortionParameter(types, width, height):
    parameters = []
    
    if types == 'barrel':
        Lambda = np.random.uniform(-5e-6 / 4, -1e-6 / 4)
        x0 = width / 2
        y0 = height / 2
        parameters.extend([Lambda, x0, y0])
        return parameters
    
    elif types == 'pincushion':
        Lambda = np.random.uniform(2.6e-6 / 4, 8.6e-6 / 4)
        x0 = width / 2
        y0 = height / 2
        parameters.extend([Lambda, x0, y0])
        return parameters
    
    elif types == 'rotation':
        theta = np.random.uniform(-15, 15)  # Degrees
        radian = math.radians(theta)
        parameters.append(radian)
        return parameters
    
    elif types == 'shear':
        shear = np.random.uniform(-0.4, 0.4)
        parameters.append(shear)
        return parameters
    
    elif types == 'wave':
        mag = np.random.uniform(2, 8)
        parameters.append(mag)
        return parameters

    elif types == 'radial_distortion':
        # Random k value for distortion
        k = np.random.uniform(-1e-4, 1e-4)  # Adjust range as needed
        parameters.append(k)
        return parameters

    elif types == 'random_perturbation':
        max_displacement = np.random.uniform(1, 5)  # Adjust range as needed
        parameters.append(max_displacement)
        return parameters

def distortionModel(types, coords, W, H, parameter):
    if types == 'barrel' or types == 'pincushion':
        Lambda, x0, y0 = parameter
        xd, yd = coords
        xd = xd - x0
        yd = yd - y0
        r_squared = xd**2 + yd**2
        coeff = 1 + Lambda * r_squared
        # Avoid division by zero
        coeff[coeff == 0] = 1e-6
        xu = xd / coeff + x0
        yu = yd / coeff + y0
        return np.array([xu, yu])
    
    elif types == 'rotation':
        radian = parameter[0]
        cos_theta = np.cos(radian)
        sin_theta = np.sin(radian)
        xd, yd = coords
        x0 = W / 2
        y0 = H / 2
        xd = xd - x0
        yd = yd - y0
        xu = cos_theta * xd - sin_theta * yd + x0
        yu = sin_theta * xd + cos_theta * yd + y0
        return np.array([xu, yu])
    
    elif types == 'shear':
        shear = parameter[0]
        xd, yd = coords
        xu = xd + shear * yd
        yu = yd
        return np.array([xu, yu])
    
    elif types == 'wave':
        mag = parameter[0]
        xd, yd = coords
        xu = xd + mag * np.sin(np.pi * 4 * yd / W)
        yu = yd
        return np.array([xu, yu])

    elif types == 'radial_distortion':
        k = parameter[0]
        xd, yd = coords
        x0 = W / 2
        y0 = H / 2
        xd_c = xd - x0
        yd_c = yd - y0
        r = np.sqrt(xd_c**2 + yd_c**2)
        xu = xd + k * xd_c * r
        yu = yd + k * yd_c * r
        return np.array([xu, yu])

    elif types == 'random_perturbation':
        max_displacement = parameter[0]
        xd, yd = coords
        noise_x = (np.random.rand(*xd.shape) - 0.5) * max_displacement
        noise_y = (np.random.rand(*yd.shape) - 0.5) * max_displacement
        xu = xd + noise_x
        yu = yd + noise_y
        return np.array([xu, yu])

# ----------------------------
# Main Script
# ----------------------------

# Image dimensions
width, height = 528, 528

# Number of images to generate
num_images = 5000

# Output directory
dist_output_dir = 'testing_data'
base_output_dir = 'base_data'
os.makedirs(dist_output_dir, exist_ok=True)
os.makedirs(base_output_dir, exist_ok=True)

# Transformation types to apply, including the new types
transformation_types = ['pincushion', 'rotation', 'wave', 'barrel']

# CSV file to save transformation parameters
csv_file = 'transformation_parameters.csv'

# Prepare the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['filename', 'transformations', 'parameters'])

# Generate coordinate grids
x = np.arange(width)
y = np.arange(height)
X, Y = np.meshgrid(x, y)
coords_base = np.array([X.flatten(), Y.flatten()])

# Counter for the number of generated images
generated_images = 0
attempts = 0

# Progress bar setup
with tqdm(total=num_images, desc='Generating images') as pbar:
    while generated_images < num_images:
        attempts += 1

        # For every batch of 10 images, generate the same transformations
        if generated_images % 10 == 0:
            # Randomly decide the number of transformations to apply (e.g., 1 to 3)
            num_transformations = random.randint(1, 3)

            # Randomly select transformations to apply
            selected_transformations = random.sample(transformation_types, num_transformations)

            # Generate parameters for each transformation
            transformation_parameters_list = []
            for types in selected_transformations:
                parameters = distortionParameter(types, width, height)
                transformation_parameters_list.append((types, parameters))

        # Create a white background image
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        # Random starting point along the top edge
        start_x = random.randint(0, width - 1)
        start_point = (start_x, 0)

        # Randomly choose to end on the left or right edge
        end_edge = random.choice(['left', 'right'])
        if end_edge == 'left':
            end_point = (0, random.randint(0, height - 1))
        else:
            end_point = (width - 1, random.randint(0, height - 1))

        # Draw a black line from the starting point to the ending point
        draw.line([start_point, end_point], fill='black', width=1)


        # Save the image with a zero-padded filename
        filename = f'{generated_images:06d}_base.jpg'
        img.save(os.path.join(base_output_dir, filename), format='JPEG', quality=95)

        # Convert image to NumPy array
        img_np = np.array(img)

        # Copy the base coordinates
        coords = coords_base.copy()

        # Apply the transformations sequentially
        for types, parameters in transformation_parameters_list:
            # Apply the distortion model
            coords = distortionModel(types, coords, width, height, parameters)

        # Reshape coordinates for mapping
        xu = coords[0].reshape((height, width))
        yu = coords[1].reshape((height, width))

        # Use map_coordinates for interpolation
        transformed_img_np = np.zeros_like(img_np)
        for channel in range(3):  # For RGB channels
            transformed_img_np[..., channel] = scipy.ndimage.map_coordinates(
                img_np[..., channel], [yu, xu], order=1, mode='constant', cval=255
            )

        # Convert transformed image back to PIL Image
        transformed_img = Image.fromarray(transformed_img_np.astype('uint8'), 'RGB')

        # Check if the image is blank (all white)
        extrema = transformed_img.getextrema()
        if all(extent == (255, 255) for extent in extrema):
            # Skip saving if the image is blank
            continue

        # Save the image with a zero-padded filename
        filename = f'{generated_images:06d}.jpg'
        transformed_img.save(os.path.join(dist_output_dir, filename), format='JPEG', quality=95)

        # Save the transformation parameters to the CSV file
        # Flatten the parameters list for CSV storage
        flat_parameters = []
        for t_type, params in transformation_parameters_list:
            flat_parameters.append(t_type)
            flat_parameters.extend(params)

        # Write to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename, ';'.join([t[0] for t in transformation_parameters_list]), flat_parameters])

        generated_images += 1
        pbar.update(1)

print(f"\nGenerated {generated_images} images (after {attempts} attempts) in the directories.")