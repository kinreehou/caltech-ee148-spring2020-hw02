import json

with open("preds.json", "r") as read_file:
	data = json.load(read_file)
	
print(list(data.keys()))