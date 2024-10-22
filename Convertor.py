import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import time

# ASCII chars to use
ASCII_CHARS = "@%#*+=-:. "

# Resize and convert image to grayscale
def resize_image(image, new_width=100):
    height, width = image.shape
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.55)  # Adjust for font size
    return cv2.resize(image, (new_width, new_height))

# Convert pixels to ASCII characters
def pixel_to_ascii(image):
    ascii_str = ""
    for pixel_value in image:
        ascii_str += ASCII_CHARS[pixel_value // 32]  # 256 // len(ASCII_CHARS)
    return ascii_str

# Convert image to ASCII
def image_to_ascii(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = resize_image(gray_image)
    ascii_image = [pixel_to_ascii(row) for row in resized_image]
    return "\n".join(ascii_image)

# Create a video with ASCII art frames and show a progress bar
def video_to_ascii(video_path, output_path, new_width=100):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    font = ImageFont.load_default()

    start_time = time.time()

    # Progress bar for frame processing
    for i in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        ascii_art = image_to_ascii(frame)

        # Create an image from the ASCII art
        ascii_image = Image.new("RGB", (new_width * 10, new_width * 5), color=(0, 0, 0))
        draw = ImageDraw.Draw(ascii_image)
        draw.text((0, 0), ascii_art, fill=(255, 255, 255), font=font)

        # Convert to a format OpenCV can work with
        cv2_frame = np.array(ascii_image)
        cv2_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)

        frames.append(cv2_frame)

        # Calculate time elapsed and estimated time remaining
        elapsed_time = time.time() - start_time
        time_per_frame = elapsed_time / (i + 1)
        remaining_time = time_per_frame * (total_frames - (i + 1))
        tqdm.write(f"Estimated time remaining: {remaining_time:.2f} seconds")

    # Release the video capture object
    cap.release()

    # Write frames to a new video
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    print("Video processing complete!")

# Convert video to ASCII art video with progress
video_to_ascii('input.mov', 'output.mov', new_width=100)