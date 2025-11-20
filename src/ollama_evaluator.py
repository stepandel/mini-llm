import json
import urllib.request
from tqdm import tqdm
import psutil
from gpt_fine_tune_instructions import format_input

def check_if_running(process_name):
	running = False
	for proc in psutil.process_iter(['name']):
		if proc.info['name'].lower() == process_name.lower():
			running = True
			break
	
	return running


def query_model(
	prompt,
	model="llama3",
	url="http://localhost:11434/api/chat"
):
	data = {
		"model": model,
		"messages": [
			{
				"role": "user",
				"content": prompt
			}
		],
		"options": {
			"seed": 123,
			"temperature": 0,
			"num_ctx": 2048
		}
	}

	payload = json.dumps(data).encode("utf-8")
	request = urllib.request.Request(url, data=payload, method="POST")

	request.add_header("Content-Type", "application/json")

	response_data = ""

	with urllib.request.urlopen(request) as response:
		while True:
			line = response.readline().decode("utf-8")
			if not line:
				break
			response_json = json.loads(line)
			response_data += response_json["message"]["content"]

	return response_data


def generate_model_scores(json_data, json_key, model="llama3"):
	scores = []
	for entry in tqdm(json_data, desc="Generating scores"):
		prompt = (
			f"Given the input: `{format_input(entry)}` "
			f"and correct response: `{entry['output']}` "
			f"score the model response: `{entry['model_response']}` "
			f"on a scale of 0 to 100, where 0 is the worst and 100 is the best."
			f"Respond with integer number only."
		)

		score = query_model(prompt, model=model)
		try:
			scores.append(int(score))
		except ValueError:
			print(f"Could not covert score: {score}")
			continue

	return scores