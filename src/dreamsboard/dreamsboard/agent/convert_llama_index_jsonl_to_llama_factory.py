import json


# 解析愿jsonl文件
def parse_jsonl_file(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    #     获取jsonl文件
    jsonl_file = "gemini_q_glm_a_finetuning_events_1152q_1710191088.258371.jsonl"
    jsonl_data = parse_jsonl_file(jsonl_file)

    # 转换为json格式
    json_data = []
    for line in jsonl_data:
        messages = line["messages"]
        # 获取系统消息, 作为system字段，并从messages中删除
        for message in messages:
            if message["role"] == "system":
                messages.remove(message)
                line["system"] = message["content"]
                break
        json_data.append(line)

    # 写入json文件
    with open(
        "gemini_q_glm_a_finetuning_events_1152q_1710191088.258371.json", "w"
    ) as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


def print_json():
    #     获取jsonl文件
    jsonl_file = (
        "/home/dmeck/Documents/content/finetuning_events_10q_2023-11-30_170732.jsonl"
    )
    jsonl_data = parse_jsonl_file(jsonl_file)

    # 转换为json格式
    json_data = []
    for line in jsonl_data:
        messages = line["messages"]
        for message in messages:
            if message["role"] == "system":
                print("system" + message["content"])
            if message["role"] == "assistant":
                print("assistant" + message["content"])


if __name__ == "__main__":
    main()
