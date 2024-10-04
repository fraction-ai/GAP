
# GPT Score Evaluation Script

This script is designed to evaluate the factual accuracy of a model's answers to questions about images, using GPT models as the evaluator. It scores answers based on predefined criteria and saves the results into a specified JSON file.

## Usage

### Prerequisites

1. Ensure the `.env` file contains the `OPENAI_API_KEY` in the following format:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
2. Install necessary libraries if not already installed:
   ```bash
   pip install openai tqdm python-dotenv
   ```

### Command-line Arguments

The script takes the following command-line arguments:

- `--input_filename`: Path to the input JSON file containing entries to evaluate.
- `--output_filename`: Path to the output JSON file where processed data and scores will be saved.
- `--max_workers`: (Optional) Maximum number of worker threads to use for parallel processing.

### Example Usage

```bash
python evaluate_model_with_gpt.py --input_filename=input.json --output_filename=output.json --max_workers=4
```

In this example:
- The script will read `input.json`, process each entry, and save the evaluated results to `output.json`.
- The evaluation is done using up to 4 threads.

## Output

The output JSON file will contain a list of dictionaries with the following structure for each entry:

```json
{
    "imageLink": "link_to_image",
    "question": "question_text",
    "modelAnswer": "model_answer_text",
    "correctAnswer": "correct_answer_text",
    "score": 0.85
}
```

Additionally, the script will display the average score based on valid entries at the end of execution.

---

This script provides a systematic approach to measure the accuracy of model-generated answers to image-related questions, leveraging GPT models for scoring.

