#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

# Input and output folder paths from arguments
INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

# Create the output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Iterate through all files in the input folder
for input_file in "$INPUT_FOLDER"/*; do
    # Get the file extension and name
    filename=$(basename -- "$input_file")
    extension="${filename##*.}"

    # Check if the file is a video (by extension)
    if [[ "$extension" =~ ^(mp4|mkv|avi|mov|flv|wmv)$ ]]; then
        echo "Processing: $input_file"
        # Define the output file path
        output_file="$OUTPUT_FOLDER/$filename"

        # Run the ffmpeg command
        ffmpeg -i "$input_file" -vf scale=720:-1 "$output_file"
        echo "Saved scaled video to: $output_file"
    else
        echo "Skipped non-video file: $filename"
    fi
done
