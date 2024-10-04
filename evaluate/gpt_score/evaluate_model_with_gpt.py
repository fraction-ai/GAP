import json
import os
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

def validate_api_key():
    try:
        openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say this is a test",
                }
            ],
            model="gpt-3.5-turbo",
        )
        return True
    except Exception as e:
        print(f"Unexpected error when validating API key: {e}")
        return False
    
def get_gpt_response(messages):
    try:
        api_params = {
            "model": "gpt-4o",
            "messages": messages,
        }

        completion = openai.chat.completions.create(**api_params)
        return completion.choices[0].message.content
    
    except Exception as e:
        print(f'Error while generating GPT response, Error: {e}')

    return None

def get_ai_score(correct_answer, model_answer):
    prompt = f"""
Please evaluate the model's answer based on the following criteria compared to the correct answer:

1. **Correct Answer**: "{correct_answer}"
2. **Model Answer**: "{model_answer}"

Criteria for evaluation:
- **Existence**: Does the model's answer correctly identify the existence or non-existence of objects or elements described in the correct answer?
- **Position**: Does the model's answer accurately describe the position or location of objects or elements as stated in the correct answer?
- **Count**: Does the model's answer correctly state the number of objects or elements mentioned in the correct answer?
- **Color**: Does the model's answer accurately describe the color of objects or elements as indicated in the correct answer?

Assign a score from 0 to 1 based on how well the model's answer meets these criteria:
- A score of "1" means the model's answer fully meets all criteria, accurately reflecting existence, position, count, and color as described in the correct answer.
- A score of "0" means the model's answer fails to meet any of the criteria, showing no alignment with the correct answer.
- Scores between "0" and "1" should reflect partial correctness, where the model's answer meets some criteria but not all, or has minor inaccuracies.

Carefully consider each criterion before deciding. What is the appropriate score (between 0 and 1) that best represents the factual correctness of the model's answer? Just return the score as a single number.
"""
    messages = [{"role": "user", "content": prompt}]
    response = get_gpt_response(messages)
    try:
        score = float(response.strip())
        return score
    except (ValueError, TypeError):
        return None

def process_data_entry(entry):
    try:
        imageLink = entry['imageLink']
        question = entry['question']
        modelAnswer = entry['modelAnswer']
        correctAnswer = entry['correctAnswer']

        score = get_ai_score(correctAnswer, modelAnswer)

        return {
            'imageLink': imageLink,
            'question': question,
            'modelAnswer': modelAnswer,
            'correctAnswer': correctAnswer,
            'score': score
        }
    except Exception as e:
        print(f"Error processing data entry: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--input_filename', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output_filename', type=str, required=True, help='Output JSON file')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker threads')

    args = parser.parse_args()

    input_filename = args.input_filename
    output_filename = args.output_filename
    max_workers = args.max_workers
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if (api_key is None):
        print("OPENAI_API_KEY variable not found in .env file.")
        return

    openai.api_key = api_key

    if not validate_api_key():
        print("Invalid OpenAI API key provided.")
        return

    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            processed_data = json.load(f)
        processed_images = {entry['imageLink'] for entry in processed_data}
    else:
        processed_data = []
        processed_images = set()

    with open(input_filename, 'r') as f:
        data = json.load(f)

    result = processed_data
    save_interval = 100
    total_processed = len(processed_data)
    total_score_sum = sum(entry['score'] for entry in processed_data if entry['score'] is not None)
    valid_score_count = sum(1 for entry in processed_data if entry['score'] is not None)

    entries_to_process = [entry for entry in data if entry['imageLink'] not in processed_images]

    with tqdm(total=len(entries_to_process), desc="Processing Entries", unit="entry") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_data_entry, entry): entry for entry in entries_to_process}
            for future in as_completed(futures):
                data_entry = future.result()
                if data_entry:
                    result.append(data_entry)
                    total_processed += 1
                    pbar.update(1)

                    score = data_entry['score']
                    if score is not None:
                        total_score_sum += score
                        valid_score_count += 1

                    if total_processed % save_interval == 0:
                        with open(output_filename, 'w') as f:
                            json.dump(result, f, indent=4)
                        print(f"Saved {total_processed} entries to file.")

    # Final save to ensure all results are stored
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=4)

    # Calculate and print average score
    if valid_score_count > 0:
        average_score = total_score_sum / valid_score_count
        print(f"Average score: {average_score:.2f}")
    else:
        print("No valid scores found.")

    print(f"Final scores saved in {output_filename}.")

if __name__ == "__main__":
    main()
