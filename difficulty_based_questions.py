import os
import shutil
import json

# Determine the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths relative to the current script
file_path = os.path.join(current_dir, "data/leetcode_questions/questions.json")
solutions_base_dir = os.path.join(current_dir, "LeetCode/solutions")
target_base_dir = os.path.join(current_dir, "difficulty_based_questions")

# Ensure target directories for each difficulty level exist
for difficulty in ['hard', 'medium', 'easy']:
    os.makedirs(os.path.join(target_base_dir, difficulty), exist_ok=True)

# Load the questions data
with open(file_path, 'r') as file:
    data = json.load(file)

# Map question IDs to their full folder names
question_folder_mapping = {}
for folder_name in os.listdir(solutions_base_dir):
    if folder_name.split('. ')[0].isdigit():  # Check if the folder name starts with a question ID
        question_id = folder_name.split('. ')[0]
        question_folder_mapping[question_id] = folder_name

# Organize and copy solution folders, and add a .txt file
for question in data['data']['problemsetQuestionList']['questions']:
    difficulty = question['difficulty'].lower()
    question_id = question['frontendQuestionId']
    title_slug = question['titleSlug']
    leetcode_url = f"https://leetcode.com/problems/{title_slug}/description/"
    
    if question_id in question_folder_mapping:
        source_dir = os.path.join(solutions_base_dir, question_folder_mapping[question_id])
        target_dir = os.path.join(target_base_dir, difficulty, question_folder_mapping[question_id])
        
        # Create a .txt file with the LeetCode URL in the source directory
        txt_file_path = os.path.join(source_dir, f"{title_slug}.txt")
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(leetcode_url)
        
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    else:
        print(f"Source directory does not exist for Question ID: {question_id}")
