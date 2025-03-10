#!/bin/bash

# Loop through all .pth files in the current directory
for file in $(ls *.pth); do
    # Check if the file name starts with {self.parameterization}
    if [[ $file == "{self.parameterization}"* ]]; then
        # Extract the file extension
        extension="${file##*.}"
        
        # Form the new file name with "mlp"
        new_file="mlp_${file#${self.parameterization}}"
        
        # Rename the file
        mv "$file" "$new_file"
        
        echo "Renamed $file to $new_file"
    fi
done