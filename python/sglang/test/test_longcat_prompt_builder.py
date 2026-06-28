import json
from sglang.srt.openai_api.longcat_prompt_builder import PromptBuilder


def read_input_data(in_path):
    input_data = []
    if in_path.endswith("jsonl"):
        with open(in_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                input_data.append(data)
    elif in_path.endswith("json"):
        with open(in_path, 'r', encoding='utf-8') as file:
            samples = json.load(file)
            for data in samples:
                input_data.append(data)
    return input_data

if __name__ == '__main__':
    prompt_builder = PromptBuilder()
    # input_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/konglingbin/tooluse/multi-tool/benchmark/origin/talkToUserBenchmark/talkToUserBenchmark_samples.json'
    # output_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/konglingbin/tooluse/multi-tool/benchmark/origin/talkToUserBenchmark/talkToUserBenchmark_samples_tmp.jsonl'
    # input_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/shixiaowei02/projects/tool_learning/origin_data/search_assistant/archive_v2/assistant_8k_20240103.json'
    input_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liwei293/general_agent/label_data/train_data/0419/train_for_lb_new_format_split_json_mode.jsonl'
    output_path = 'tmp.jsonl'
    fout = open(output_path, 'w')
    invalid_count = 0
    count = 0
    benchmark = read_input_data(input_path)
    # with open(input_path, 'r', encoding='utf-8') as f:
    #     benchmark = json.load(f)
    for data in benchmark:
        count += 1
        try:
            input_text = prompt_builder.build_input(data['messages'][:-1], data['tools'], data['tool_choice'])
            target_text = prompt_builder.build_target(data['messages'][-1], data['tools'])
            print(input_text)
            print(target_text)
            
            if input_text is None or target_text is None:
                invalid_count += 1
                continue
            sample = {
                'input': input_text,
                'target': target_text
            }
            line = json.dumps(sample, ensure_ascii=False)
            fout.write(line + '\n')
        except Exception as e:
            print(data["sample_id"], " ## ", e)
    print(count)
    print(invalid_count)
