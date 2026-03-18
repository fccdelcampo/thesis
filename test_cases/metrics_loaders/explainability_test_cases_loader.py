import os, sys, json
# Add project root to sys.path to allow importing from parallel directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
TEST_CASES_DIR = os.path.join(project_root, "test_cases")

CORRECT_IDS = ["5", "14", "17", "24", "26", "28", "31", "33", "41", "59", "63", "66", "81", "85", "89", "94", "96", "100", "108", "112", "113", "115", "124", "125", "126", "139", "147", "149", "151", "159", "168", "169", "178", "183", "192", "196", "199", "207", "208", "221", "224", "227", "229", "275", "276", "277", "278", "279", "288", "289"]
INCORRECT_IDS = ["1", "3", "4", "7", "8", "10", "15", "16", "20", "22", "23", "27", "34", "36", "40", "44", "46", "47", "48", "49", "50", "51", "53", "60", "68", "70", "74", "77", "78", "91", "99", "114", "135", "137", "138", "148", "163", "166", "172", "180", "209", "222", "225", "233", "239", "243", "248", "252", "267", "269"]

IDS = ["1", "2", "3", "4", "5", "7", "8", "9", "10", "12", "14", "15", "16", "17", "19", "20", "21", "22", "23", "25", "26", "27", "28", "31", "32", "33", "34", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "58", "59", "60", "63", "64", "65", "66", "67", "68", "69", "70", "72", "74", "75", "76", "77", "78", "79", "83", "85", "87", "88", "89", "91", "92", "94", "96", "97", "98", "99", "100", "103", "104", "106", "107", "108", "109", "110", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "125", "126", "127", "129", "134", "135", "137", "138", "139", "142", "144", "147", "148", "149", "150", "151", "155", "157", "158", "159", "160", "161", "163", "164", "165", "166", "168", "171", "173", "174", "176", "178", "179", "183", "184", "185", "186", "188", "189", "190", "192", "195", "196", "197", "199", "200", "203", "204", "206", "207", "208", "210", "211", "214", "218", "219", "221", "222", "223", "225", "229", "232", "233", "237", "239", "243", "245", "248", "251", "252", "253", "255", "256", "259", "260", "262", "266", "267", "268", "269", "270", "272", "274", "275", "276", "277", "278", "279", "280", "281", "284", "285", "286", "288", "289", "290", "292", "293", "294", "295", "296", "297", "298", "299", "200"]

def load_test_cases() -> list[dict]:
    """
    Reads .jsonl files in test_cases/ and turns each JSON line into a dict.
    Specifically targets faithfulness.jsonl (or creates synthetic ones if empty).
    For now, we will use mmlu.jsonl as a proxy if faithfulness.jsonl is empty, 
    or just return an empty list if strict. The plan says use faithfulness.jsonl.
    """
    input_file_path = os.path.join(TEST_CASES_DIR, "metrics_test_cases.jsonl")
    output_file_path = os.path.join(TEST_CASES_DIR, "explainability_test_cases.jsonl")
    metrics_test_cases = []
    explainability_test_cases = []
    if os.path.exists(input_file_path) and os.path.exists(output_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metrics_test_cases.append(json.loads(line))
        
        print(f"{len(metrics_test_cases)} metrics test cases initially loaded.")
                    
        for id in IDS:
            print(id)
            if metrics_test_cases[int(id) - 1]["answer"].lower().replace(".", "") not in ["yes", "no"]:
                explainability_test_cases.append(metrics_test_cases[int(id) - 1])
                
        print(f"{len(explainability_test_cases)} explainability test cases finally loaded.")
            
        with open(output_file_path, 'a', encoding='utf-8') as file:
            for test_case in explainability_test_cases:
                file.write(json.dumps(test_case) + "\n")
        
        print("Done writing to explainability_test_cases.jsonl")
                    
load_test_cases()
                    
    
    