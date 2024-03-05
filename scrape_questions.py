import os
import requests
import json

base_url = "https://leetcode.com"
def get_leetcode_questions(category_slug="all", skip=0, limit=100, filters={}):
    """
    Fetches questions from LeetCode's GraphQL API.

    Args:
    - category_slug (str): The category of the questions to fetch. Default is 'all'.
    - skip (int): The number of questions to skip. Useful for pagination. Default is 0.
    - limit (int): The maximum number of questions to fetch. Default is 100.
    - filters (dict): Filters to apply to the question query.

    Returns:
    - dict: A dictionary containing the fetched question data.
    """
    url = f"{base_url}/graphql/"
    payload_json = {
        "query": """
        query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
          problemsetQuestionList: questionList(
            categorySlug: $categorySlug
            limit: $limit
            skip: $skip
            filters: $filters
          ) {
            total: totalNum
            questions: data {
              acRate
              difficulty
              freqBar
              frontendQuestionId: questionFrontendId
              isFavor
              paidOnly: isPaidOnly
              status
              title
              titleSlug
              topicTags {
                name
                id
                slug
              }
              hasSolution
              hasVideoSolution
            }
          }
        }
        """,
        "variables": {
            "categorySlug": category_slug,
            "skip": skip,
            "limit": limit,
            "filters": filters,
        },
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload_json))
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return {"error": "Failed to fetch data", "status_code": response.status_code}


response_json = get_leetcode_questions(category_slug="all-code-essentials", skip=0, limit=4000)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data/leetcode_questions")
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "questions.json")
with open(file_path, 'w') as file:
    json.dump(response_json, file, indent=4)
print(f"File saved to {file_path}")