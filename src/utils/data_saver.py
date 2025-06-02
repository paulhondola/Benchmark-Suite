import os
import json
from datetime import datetime
from utils import RESULTS_DIR

def save_results(results, filename):
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	out_path = os.path.join(RESULTS_DIR, f"{filename}_{timestamp}.json")
	with open(out_path, "w") as f:
		json.dump(results, f, indent=2)
	print(f"Results saved to: {out_path}")
