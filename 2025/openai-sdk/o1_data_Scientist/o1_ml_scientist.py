import os  # noqa: E402
import re
import shutil
import subprocess

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from termcolor import colored

load_dotenv()

# Number of iterations
iterations = 20

print(colored("Phase 1: Data Loading and Preparation", "cyan"))
# Read the first 10 rows of train.csv
train_data = pd.read_csv("train.csv", nrows=10)
# print(train_data.to_string(index=False))

# Read additional info
with open("additional_info.txt", "r") as file:
    additional_info = file.read()

print(colored("Phase 2: OpenAI API Interaction and Code Execution", "cyan"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# To use openrouter
# client = OpenAI(api_key=os.getenv('OPENROUTER_API_KEY'), base_url='https://openrouter.ai/api/v1') and use model "openai/o1-mini"

previous_code = ""
output_file = "execution_outputs.txt"
result = None

for i in range(iterations):
    print(colored(f"Iteration {i + 1}/{iterations}", "yellow"))

    # Move old solution and progress report to older_solutions folder
    if i > 0:
        older_solutions_dir = "older_solutions"
        os.makedirs(older_solutions_dir, exist_ok=True)

        if os.path.exists("solution.py"):
            shutil.move("solution.py", f"{older_solutions_dir}/solution_{i}.py")

        if os.path.exists("progress_report.json"):
            shutil.move(
                "progress_report.json",
                f"{older_solutions_dir}/progress_report_{i}.json",
            )

    initial_prompt = f"""You are an expert ML scientist. Your task is to write excellent Python code to solve a difficult data challenge. Here's the information about the challenge:

{additional_info}

Here are the first 10 rows of the training data:

{train_data.to_string(index=False)}

The data files available are train.csv and test.csv. Pay specific attention to column names.

Based on this information, please write error free Python code to:
1. Load and preprocess both train.csv and test.csv
2. Perform exploratory data analysis without using any plots
3. Engineer relevant features
4. Select and train an appropriate machine learning model using the training data
5. Evaluate the model's performance using cross-validation. clearly print the accuracy every so often
6. Make predictions on the test set and print the accuracy
- save the progress report and the final accuracy to a file called progress_report.json
7. Prepare the submission file in the required format
- do not use any plots
- you can print any analysis or information necessary as this will be passed back to you for the next iteration to improve the code
- make sure you use utf-8 encoding when writing to files
- do not use logging instead print the information necessary

files are train.csv and test.csv

Please provide the complete code solution, including necessary imports and explanations as comments."""

    if i == 0:
        messages = [{"role": "user", "content": initial_prompt}]
    else:
        messages = [
            {"role": "user", "content": initial_prompt},
            {
                "role": "assistant",
                "content": f"Here's the previous code I generated:\n\n```python\n{previous_code}\n```",
            },
            {
                "role": "user",
                "content": f"Please improve the code and accuracy based on the following execution results and fix any errors. if there are any errors, please fix them first before trying to improve the code and the accuracy:\n\nExecution output:\n{result.stdout}\n\nExecution errors:\n{result.stderr}",
            },
        ]

    response = client.chat.completions.create(model="o1-mini", messages=messages)

    print(colored("ML Scientist's Solution:", "cyan"))
    solution_content = response.choices[0].message.content
    print(solution_content)

    code_blocks = re.findall(r"```python(.*?)```", solution_content, re.DOTALL)
    if code_blocks:
        with open("solution.py", "w") as file:
            for block in code_blocks:
                file.write(block.strip() + "\n\n")
        print(colored("Solution saved to solution.py", "green"))

        # Execute the generated code
        print(colored("Executing generated code:", "cyan"))
        result = subprocess.run(
            [".venv\\Scripts\\python.exe", "solution.py"],
            capture_output=True,
            text=True,
        )

        print(colored("Execution output:", "green"))
        print(result.stdout)

        if result.stderr:
            print(colored("Execution errors:", "red"))
            print(result.stderr)

        # Save execution output to file
        with open(output_file, "a") as f:
            f.write(f"Iteration {i + 1}/{iterations}\n")
            f.write("Execution output:\n")
            f.write(result.stdout)
            f.write("\nExecution errors:\n")
            f.write(result.stderr)
            f.write("\n" + "=" * 50 + "\n\n")

        # Update previous_code for the next iteration
        previous_code = "\n".join(code_blocks)

        # Prepare feedback for the next iteration
        execution_feedback = f"previous code:\n{previous_code}\n\nExecution output:\n{result.stdout}\n\nExecution errors:\n{result.stderr}"
        messages.append({"role": "user", "content": execution_feedback})
    else:
        print(colored("No Python code blocks found in the solution.", "red"))
        break

print(colored("ML Scientist process completed.", "cyan"))
print(f"All execution outputs have been saved to {output_file}")
