import os
import subprocess
from PIL import Image

def ensure_even_dimensions(input_path, output_path):
    """
    Ensure the image has even width and height by cropping if necessary
    """
    with Image.open(input_path) as img:
        width, height = img.size
        
        # Ensure width is even
        new_width = width if width % 2 == 0 else width - 1
        
        # Ensure height is even
        new_height = height if height % 2 == 0 else height - 1
        
        # Calculate crop coordinates to center the crop
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))
        
        # Save the cropped image
        cropped_img.save(output_path)

def create_seam_carving_video(image_directory, output_video, framerate=30, crf=23):
    """
    Create a video from a sequence of seam carving images
    
    Parameters:
    - image_directory: Path to the directory containing images
    - output_video: Path for the output MP4 file
    - framerate: Frames per second (default 30)
    - crf: Constant Rate Factor for video compression (0-51, lower = higher quality, 23 is a good default)
    """
    # Create a temporary directory for processed images
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Process all images to ensure even dimensions
        processed_images = []
        for filename in sorted(os.listdir(image_directory), key=lambda x: int(os.path.splitext(x)[0])):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(image_directory, filename)
                output_path = os.path.join(tmpdir, filename)
                ensure_even_dimensions(input_path, output_path)
                processed_images.append(output_path)
        
        # Ensure processed images are sorted correctly
        processed_images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        # Create a file with the list of processed images for FFmpeg
        image_list_path = os.path.join(tmpdir, 'image_list.txt')
        with open(image_list_path, 'w') as f:
            for img in processed_images:
                f.write(f"file '{img}'\n")
        
        # FFmpeg command to convert images to video
        ffmpeg_command = [
            'ffmpeg',
            '-framerate', str(framerate),
            '-f', 'concat',
            '-safe', '0',
            '-i', image_list_path,
            '-c:v', 'libx264',  # H.264 codec
            '-crf', str(crf),   # Compression level
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            output_video
        ]
        
        try:
            # Run the FFmpeg command
            subprocess.run(ffmpeg_command, check=True)
            print(f"Video created successfully: {output_video}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e}")
        except FileNotFoundError:
            print("FFmpeg not found. Please install FFmpeg first.")

# Example usage
image_directory = 'D:\Sofia\SeamCarving\meow'
output_video = 'seam_carving_progression.mp4'

create_seam_carving_video(image_directory, output_video)