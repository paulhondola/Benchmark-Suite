import hwinfo.hwinfo as hwi
import json

if __name__ == "__main__":
	hw_info = hwi.collect_hw_info()
	print(json.dumps(hw_info, indent=2))
	print("Hardware information collection complete.")
