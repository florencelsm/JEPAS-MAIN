import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

# --- Configuration for Sanity Check ---
SANITY_CHECK_INTERVAL = 10
FONT_SIZE = 20
FONT_PATH = "arial.ttf"

def download_video(video_id, output_path):
    """Downloads a YouTube video using yt-dlp."""
    try:
        # Use --no-warnings to suppress minor warnings
        # Use --quiet to suppress most output during download
        # -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 attempts to get best mp4 streams
        # -o specifies the output file path
        subprocess.run(
            ['yt-dlp', '--no-warnings', '--quiet', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', '-o', output_path, f'https://www.youtube.com/watch?v={video_id}'],
            check=True,
            capture_output=True, # Capture stdout/stderr for error reporting
            text=True            # Decode output as text
        )
        print(f"Successfully downloaded video {video_id}")
        return True
    except subprocess.CalledProcessError as e:
        if "ERROR: [youtube]" in e.stderr:
            print(f"Video {video_id} might be unavailable or private. Error: {e.stderr.strip()}")
        else:
            print(f"Error downloading video {video_id}: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("Error: 'yt-dlp' not found. Please install it (e.g., `pip install yt-dlp`) and ensure it's in your system's PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during video download for {video_id}: {e}")
        return False

def extract_audio_segment(video_path, start_time_seconds, duration_seconds, output_audio_path):
    """Extracts a 6-second audio segment using ffmpeg."""
    try:
        # -ss specifies the start time
        # -t specifies the duration
        # -vn disables video recording (only audio)
        # -acodec aac ensures AAC audio codec, widely compatible
        subprocess.run(
            ['ffmpeg', '-i', video_path, '-ss', str(start_time_seconds), '-t', str(duration_seconds), '-vn', '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', output_audio_path],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {os.path.basename(video_path)}: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("Error: 'ffmpeg' not found. Please install it and ensure it's in your system's PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during audio extraction from {os.path.basename(video_path)}: {e}")
        return False

def extract_image_frame(video_path, timestamp_seconds, output_image_path):
    """Extracts an image frame using ffmpeg."""
    try:
        # -ss specifies the timestamp
        # -vframes 1 extracts only one frame
        # -q:v 2 sets the quality (lower number means higher quality for JPG)
        subprocess.run(
            ['ffmpeg', '-i', video_path, '-ss', str(timestamp_seconds), '-vframes', '1', '-q:v', '2', output_image_path],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting image from {os.path.basename(video_path)}: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("Error: 'ffmpeg' not found. Please install it and ensure it's in your system's PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during image extraction from {os.path.basename(video_path)}: {e}")
        return False

def get_image_size(image_path):
    """Gets the original size of an image."""
    try:
        with Image.open(image_path) as img:
            return list(img.size)  # [width, height]
    except Exception as e:
        print(f"Error getting image size for {image_path}: {e}")
        return None

def perform_sanity_check(image_path, bbox_norm, category_text, output_dir, file_name_for_output):
    """
    Loads an image, plots the bounding box and category, and saves it to the sanity_check folder.

    Args:
        image_path (str): Path to the downloaded image.
        bbox_norm (list): Normalized bounding box coordinates [[x_min, y_min, x_max, y_max]].
        category_text (str): The class category string.
        output_dir (str): Directory to save the sanity check image.
        file_name_for_output (str): Base filename (e.g., zpWuikVorYg_000032) for the output image.
    """
    try:
        img = Image.open(image_path).convert("RGB") # Ensure RGB mode for consistent drawing
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        # The dataset's 'bbox' field is a list of bounding boxes, even if usually one.
        if not bbox_norm or not isinstance(bbox_norm, list) or not bbox_norm[0]:
            print(f"Warning: No valid bounding box data found for {file_name_for_output}. Skipping sanity check plot.")
            return

        # Use the first bounding box for plotting
        x_min, y_min, x_max, y_max = bbox_norm[0]

        # Convert normalized coordinates (0-1) to pixel coordinates
        x_min_px = int(x_min * img_width)
        y_min_px = int(y_min * img_height)
        x_max_px = int(x_max * img_width)
        y_max_px = int(y_max * img_height)

        # Draw the bounding box (red outline, 3 pixels wide)
        draw.rectangle([(x_min_px, y_min_px), (x_max_px, y_max_px)], outline="red", width=3)

        # Load font for text. Fallback to default if specified font not found.
        try:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except IOError:
            # print(f"Warning: Font '{FONT_PATH}' not found. Using default PIL font for sanity check.")
            font = ImageFont.load_default()

        # Add the category text (red text)
        # Position text slightly above the bounding box. If too high, place it inside.
        text_x = x_min_px
        text_y = y_min_px - FONT_SIZE - 5 # 5 pixels padding above box
        if text_y < 0: # If text goes off top of image, place it inside the box
            text_y = y_min_px + 5

        draw.text((text_x, text_y), category_text, fill="red", font=font)

        # Save the annotated image to the sanity_check directory
        sanity_check_output_path = os.path.join(output_dir, f"{file_name_for_output}_bbox.jpg")
        img.save(sanity_check_output_path)
        print(f"Sanity check image saved: {sanity_check_output_path}")

    except FileNotFoundError:
        print(f"Error: Image file not found for sanity check at {image_path}")
    except Exception as e:
        print(f"An error occurred during sanity check for {file_name_for_output}: {e}")


def process_entry(entry, base_output_dir, sanity_check_output_dir, item_index):
    """
    Processes a single dataset entry: downloads video, extracts audio and image,
    and conditionally performs a sanity check.
    
    Returns:
        dict or None: Dictionary with processing results if successful, None otherwise
    """
    file_name = entry['file']
    video_id, timestamp_str = file_name.split('_')
    timestamp_seconds = int(timestamp_str)

    video_output_path = os.path.join(base_output_dir, "videos", f"{video_id}.mp4")
    audio_output_path = os.path.join(base_output_dir, "audio", f"{file_name}.wav")
    image_output_path = os.path.join(base_output_dir, "images", f"{file_name}.jpg")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
    os.makedirs(sanity_check_output_dir, exist_ok=True)

    print(f"Processing {file_name} (Item {item_index + 1})...")

    video_downloaded = False
    # Check if video already exists to avoid re-downloading
    if os.path.exists(video_output_path):
        print(f"Video {video_id} already exists. Skipping download.")
        video_downloaded = True
    else:
        video_downloaded = download_video(video_id, video_output_path)

    if not video_downloaded:
        print(f"Skipping further processing for {file_name} due to video download/availability issue.")
        return None

    # Extract audio
    # Audio segment starts 3 seconds before timestamp, ensuring it doesn't go negative
    audio_start_time = max(0, timestamp_seconds - 3)
    audio_duration = 6 # Total 6 seconds (3 before + 3 after)
    audio_extracted = False
    if not os.path.exists(audio_output_path): # Avoid re-extracting audio
        if extract_audio_segment(video_output_path, audio_start_time, audio_duration, audio_output_path):
            audio_extracted = True
        else:
            print(f"Failed to extract audio for {file_name}.")
            return None
    else:
        print(f"Audio for {file_name} already exists. Skipping audio extraction.")
        audio_extracted = True

    # Extract image frame
    image_extracted = False
    if not os.path.exists(image_output_path): # Avoid re-extracting image
        if extract_image_frame(video_output_path, timestamp_seconds, image_output_path):
            image_extracted = True
        else:
            print(f"Failed to extract image for {file_name}.")
            return None
    else:
        print(f"Image for {file_name} already exists. Skipping image extraction.")
        image_extracted = True # Consider it extracted if it exists

    # Get image size
    image_size = get_image_size(image_output_path)
    if image_size is None:
        print(f"Failed to get image size for {file_name}.")
        return None

    # Perform sanity check if conditions met
    if image_extracted and (item_index + 1) % SANITY_CHECK_INTERVAL == 0:
        perform_sanity_check(
            image_output_path,
            entry.get('bbox', []),      # Get 'bbox' or empty list if not present
            entry.get('class', 'N/A'),  # Get 'class' or 'N/A' if not present
            sanity_check_output_dir,
            file_name
        )

    print(f"Finished processing {file_name}.")
    
    # Return the data in the format [waveform path, image path, bounding box, original size]
    return {
        "file_name": file_name,
        "data": [
            audio_output_path,      # waveform path
            image_output_path,      # image path  
            entry.get('bbox', []),  # bounding box
            image_size              # original size [width, height]
        ]
    }

def main(json_file_path, output_directory="vgg_ss_extracted_data"):
    """
    Main function to automate the extraction process for the VGG-SS dataset.

    Args:
        json_file_path (str): Path to the VGG-SS dataset JSON file.
        output_directory (str): Base directory to save extracted videos, audio, and images.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    try:
        with open(json_file_path, 'r') as f:
            dataset = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}. Please check file integrity.")
        return
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return

    # Create main output and sanity check directories
    os.makedirs(output_directory, exist_ok=True)
    sanity_check_output_dir = os.path.join(output_directory, "sanity_check")
    os.makedirs(sanity_check_output_dir, exist_ok=True)

    print(f"Starting VGG-SS dataset extraction process.")
    print(f"Output will be saved to: {os.path.abspath(output_directory)}")
    print(f"Sanity check images will be saved to: {os.path.abspath(sanity_check_output_dir)}")
    print(f"Sanity check performed every {SANITY_CHECK_INTERVAL} items.")
    print(f"Processing {len(dataset)} entries.")

    # Dictionary to store the final results
    extracted_data = {}

    # Use ThreadPoolExecutor for concurrent processing to speed up downloads/extractions.
    # Adjust max_workers based on your system's CPU cores and internet bandwidth.
    # Too many workers can overload your network or CPU.
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each entry, passing the current index for sanity check logic
        futures = {executor.submit(process_entry, entry, output_directory, sanity_check_output_dir, i): i
                   for i, entry in enumerate(dataset)}

        # Wait for tasks to complete and retrieve results (or exceptions)
        for future in as_completed(futures):
            try:
                result = future.result() # This will re-raise any exception caught during the execution of the task
                if result is not None:
                    # Store the result with file_name as key
                    extracted_data[result["file_name"]] = result["data"]
            except Exception as exc:
                print(f'An item processing task generated an exception: {exc}')

    # Save the extracted data to a JSON file
    output_json_path = os.path.join(output_directory, "extracted_data.json")
    try:
        with open(output_json_path, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        print(f"\nExtracted data saved to: {output_json_path}")
        print(f"Successfully processed {len(extracted_data)} entries.")
    except Exception as e:
        print(f"Error saving extracted data to JSON: {e}")

    print("\nExtraction process completed.")
    print("Please check the 'sanity_check' folder for annotated images.")
    print(f"Final JSON data format: {{\"file_name\": [waveform_path, image_path, bounding_box, original_size]}}")

if __name__ == "__main__":
    print("--- VGG-SS Dataset Extraction Script ---")
    print("Please ensure the following prerequisites are met:")
    print("1. **yt-dlp:** Installed and in your system's PATH. (`pip install yt-dlp`)")
    print("2. **ffmpeg:** Installed and in your system's PATH. (Download from ffmpeg.org)")
    print("3. **Pillow:** Installed. (`pip install Pillow`)")
    print("4. **vggss.json:** Downloaded from https://www.robots.ox.ac.uk/~vgg/research/lvs/data/vggss.json")
    print("   and placed in the same directory as this script, or specify its full path.\n")

    # Define the path to your vggss.json file
    json_dataset_file = "C:/Users/ali97/Desktop/Repos/Florence/JEPAS-MAIN/vggss_download/vggss_dataset.json"

    # You can change the output directory name here
    output_base_directory = "C:/Users/ali97/Desktop/Repos/Florence/JEPAS-MAIN/vggss_data"

    if not os.path.exists(json_dataset_file):
        print(f"ERROR: The dataset file '{json_dataset_file}' was not found.")
        print("Please download it and place it in the same directory as this script, or modify 'json_dataset_file'.")
    else:
        main(json_dataset_file, output_base_directory)