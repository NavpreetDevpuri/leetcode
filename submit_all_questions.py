import json
import os
import requests
from time import sleep

from session_secrets import cookies_header, csrftoken_header
from scrape_questions import scrape_questions_and_save
# scrape_questions_and_save()


# Assuming the same headers as before
# Placeholder for headers and the submit_leetcode_question function
headers = {
  'cookie': cookies_header,
  'referer': '',
  'x-csrftoken': csrftoken_header,
  'Content-Type': 'application/json'
}

def submit_leetcode_question(question_id, slug_text, source_code):
    print(f"Submitting solution for Question ID: {question_id}, Slug: {slug_text}")
    url = f"https://leetcode.com/problems/{slug_text}/submit/"
    headers['referer'] = f"https://leetcode.com/problems/{slug_text}/"
    
    payload = json.dumps({
        "lang": "python3",
        "question_id": question_id,
        "typed_code": source_code
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    
    return json.loads(response.text)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load and create the mapping from the questions.json file
questions_file_path = os.path.join(current_dir, "data/leetcode_questions/questions.json")
with open(questions_file_path, 'r') as file:
    questions_data = json.load(file)

question_id_map = {question["frontendQuestionId"]: question["questionId"] for question in questions_data["data"]["problemsetQuestionList"]["questions"]}



# The rest of your solution submission logic follows
solutions_dir = os.path.join(current_dir, "LeetCode/solutions")
solutions = os.listdir(solutions_dir)
solutions.sort()
for folder_name in solutions:
    folder_path = os.path.join(solutions_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    py_file_found = False
    slug_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.py'):
            py_file_found = True
            py_file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.txt'):
            slug_text = file_name[:-4]

    if py_file_found and slug_text:
        with open(py_file_path, 'r') as py_file:
            source_code = py_file.read()

        frontend_question_id = folder_name.split('. ')[0]
        question_id = question_id_map.get(frontend_question_id)
        if question_id:
            response = submit_leetcode_question(question_id, slug_text, source_code)
            print(response)
        else:
            print(f"Question ID mapping not found for Frontend Question ID: {frontend_question_id}")
    else:
        print(f"Python solution file or slug text file is missing in {folder_name}")

    sleep(5)  # Be mindful of rate limiting
