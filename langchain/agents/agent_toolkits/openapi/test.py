import json


def parse_json(data: str):
    data = data.split("```")

    # 找到JSON部分并删除多余的换行
    json_str = ""
    if data.startswith("json"):
        data = data.split("json", 1)[-1]
    # 去除多余的空白和换行
    json_str = json_str.replace('\n', '').replace(' ', '')

    # 解析JSON
    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("解析失败:", e)
        return None

    return json_data


print(parse_json('{"json": 123}'))