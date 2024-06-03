import os
from datetime import datetime

# Get the current datetime and format it for the filename
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the filename for the output .txt file
output_filename = f"{current_datetime}.txt"

# Open the output file
with open(output_filename, 'w') as output_file:
    # Get the list of files in the current directory
    for file in os.listdir('.'):
        if file.endswith('.py') or file.endswith('.yaml'):
            # Write the filename as a header in the output file
            output_file.write(f"### {file} ###\n")
            # Write the contents of the file to the output file
            with open(file, 'r') as input_file:
                output_file.write(input_file.read())
            # Write a separator
            output_file.write("\n\n")

print(f"Contents of .py and .yaml files written to {output_filename}")
