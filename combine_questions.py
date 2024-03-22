import os

# Base directory for searching Python files
base_dir = '/Users/navpreetdevpuri/personal_workspace/leetcode/difficulty_based_questions/hard'

# Output file path
output_file_path = os.path.join(base_dir, 'merged_solutions_with_links.py')

def find_text_file_for_python_file(python_file_path):
    """Finds the corresponding text file for a given Python file."""
    dir_path = os.path.dirname(python_file_path)
    for file in os.listdir(dir_path):
        if file.endswith('.txt'):
            return os.path.join(dir_path, file)
    return None

# Initialize an empty list to store file paths
python_files = []

# Walk through the directory structure
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate through the list of Python file paths
    for file_path in python_files:
        # Find the corresponding text file
        text_file_path = find_text_file_for_python_file(file_path)
        if text_file_path:
            # Read the link from the text file
            with open(text_file_path, 'r') as text_file:
                link = text_file.read().strip()
                # Write the link as a comment in the output file
                output_file.write(f"# Link: {link}\n")
                question_text = os.path.split(os.path.dirname(file_path))[1]
                output_file.write(f"# Question: {question_text}\n")

        # Open and read the Python file, then write its content to the output file
        with open(file_path, 'r') as file:
            content = file.read()
            output_file.write(content)
            output_file.write("\n\n")  # Add some space between files for readability

print(f"Successfully merged {len(python_files)} Python files into '{output_file_path}'")
