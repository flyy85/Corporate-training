from argparse import ArgumentParser
import json
import re
from typing import Union
import gradio as gr
import warnings
from openai import OpenAI
import os


# 忽视所有警告
warnings.filterwarnings("ignore")

API_KEY = "function-call/watt-tool-8B"
BASE_URL = "http://192.168.0.79:7864/v1"

coder_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)
KNOWLEDGE_BASE = "knowledge_base"


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=7864, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="localhost", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def read_md_files_from_knowledge_base():
    result = []
    knowledge_base_path = KNOWLEDGE_BASE

    # Check if the KNOWLEDGE_BASE directory exists
    if not os.path.exists(knowledge_base_path):
        print(f"Error: The directory {knowledge_base_path} does not exist.")
        return result

    # Iterate through all files in the KNOWLEDGE_BASE directory
    for filename in os.listdir(knowledge_base_path):
        if filename.endswith(".md"):
            file_path = os.path.join(knowledge_base_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    result.append(
                        {
                            "filename": os.path.basename(filename)[:-3],
                            "content": content,
                        }
                    )
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")

    return result


def read_py_files_from_knowledge_base():
    result = []
    knowledge_base_path = KNOWLEDGE_BASE

    # Check if the KNOWLEDGE_BASE directory exists
    if not os.path.exists(knowledge_base_path):
        print(f"Error: The directory {knowledge_base_path} does not exist.")
        return result

    # Iterate through all files in the KNOWLEDGE_BASE directory
    for filename in os.listdir(knowledge_base_path):
        if filename.endswith(".py"):
            file_path = os.path.join(knowledge_base_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    result.append(
                        {
                            "filename": os.path.basename(filename)[:-3],
                            "content": content,
                        }
                    )
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")

    return result


def read_json_files_from_knowledge_base():
    result = []
    knowledge_base_path = KNOWLEDGE_BASE

    # Check if the KNOWLEDGE_BASE directory exists
    if not os.path.exists(knowledge_base_path):
        print(f"Error: The directory {knowledge_base_path} does not exist.")
        return result

    # Iterate through all files in the KNOWLEDGE_BASE directory
    for filename in os.listdir(knowledge_base_path):
        if filename.endswith(".json"):
            file_path = os.path.join(knowledge_base_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    content_json = json.loads(content)
                    result.append(
                        {
                            "filename": os.path.basename(filename)[:-5],
                            "content": content,
                            "name": content_json.get("name", ""),
                        }
                    )
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
    return result


def replace_latex_delimiters(text: str):
    text = text.replace("<think>", "").replace("</think>", "").replace("&", "")
    # text = re.sub(r"<think>.*</think>", "", text, flags=re.DOTALL)

    patterns = [
        r"\\begin\{equation\}(.*?)\\end\{equation\}",  # \begin{equation} ... \end{equation}
        r"\\begin\{aligned\}(.*?)\\end\{aligned\}",  # \begin{aligned} ... \end{aligned}
        r"\\begin\{alignat\}(.*?)\\end\{alignat\}",  # \begin{alignat} ... \end{alignat}
        r"\\begin\{align\}(.*?)\\end\{align\}",  # \begin{align} ... \end{align}
        r"\\begin\{gather\}(.*?)\\end\{gather\}",  # \begin{gather} ... \end{gather}
        r"\\begin\{CD\}(.*?)\\end\{CD\}",  # \begin{CD} ... \end{CD}
    ]
    # 替换所有匹配的模式
    for pattern in patterns:
        text = re.sub(pattern, r" $$ \1 $$ ", text, flags=re.DOTALL)
    # 定义正则表达式模式
    patterns = [
        r"\\\[\n(.*?)\n\\\]",  # \[ ... \]
        r"\\\(\n(.*?)\n\\\)",  # \( ... \)
    ]
    # 替换所有匹配的模式
    for pattern in patterns:
        text = re.sub(pattern, r" $$ \1 $$ ", text, flags=re.DOTALL)
    # 定义正则表达式模式
    patterns = [
        r"\\\[(.*?)\\\]",  # \[ ... \]
        r"\\\((.*?)\\\)",  # \( ... \)
    ]
    # 替换所有匹配的模式
    for pattern in patterns:
        text = re.sub(pattern, r" $ \1 $ ", text, flags=re.DOTALL)
    return text


def parse_function_parameters(response: str) -> dict:
    def extract_parameters(input_string):
        # 定义正则表达式模式
        pattern = r'(\w+)\s*=\s*(\[[\d,\s\.]+\]|[^\'^"^,]+|\'[^\'^"^,]+\'|"[^\'^"^,]+")'
        # 使用 findall 提取所有匹配的参数和对应的值
        matches = re.findall(pattern, input_string)
        # 转换为字典，去掉引号
        parameters = {key: value.strip("'\"") for key, value in matches}
        return parameters

    match = re.search(r"(\[.*\])", response)
    if match:
        response = match.group(1)
    else:
        return {}
    # 匹配函数名
    function_pattern = r"\[([a-zA-Z0-9_]+)\("
    function_match = re.search(function_pattern, response)
    # 匹配参数
    params_pattern = r"\((.*?)\)"
    params_match = re.search(params_pattern, response)
    if function_match and params_match:
        function_name = function_match.group(1)
        params_string = params_match.group(1)
        # 输出函数名和参数对
        print(f"Function Name: {function_name}")
        print(f"Parameters: {params_string}")
        # 提取参数对
        params = extract_parameters(params_string)
        return {"name": function_name, "arguments": params}
    else:
        print("未找到匹配的函数。")
        return {}


def decorate_response(
    file_name: str, func_name: str, arguments: dict, docs: dict, codes: dict
) -> str:
    def try_convert_string(s):
        try:
            return eval(s)
        except Exception as e:
            print(f"{e}")
            return s

    def get_doc() -> None | object:
        for doc in docs:
            if doc["filename"] == file_name:
                return doc["content"]

    def get_code() -> Union[None | object, None | str]:
        for code in codes:
            if code["filename"] == file_name:
                python_str = code["content"]
                func = {}
                try:
                    exec(python_str, None, func)
                except Exception as e:
                    print("exec error:{}".format(e))
                return func.get(func_name), python_str

    func_param = {x: try_convert_string(arguments.get(x)) for x in arguments}
    func_arguments = {x: func_param.get(x) for x in func_param}

    result = ""
    doc = get_doc()
    if doc is not None:
        formula_doc = replace_latex_delimiters(doc)
        result += f"""
# **公式描述**
{formula_doc}
"""

        func_arguments = json.dumps(func_arguments, indent=2, ensure_ascii=False)
        result += f"""
# **公式参数**
```json
{func_arguments}
```
"""

        func_object, func_str = get_code()
        result += f"""
# **公式算子**
```python
{func_str}
```
"""
        if func_object is not None:
            try:
                func_result = func_object(**func_param)
            except Exception as e:
                func_result = f"计算错误： {e}"
        else:
            func_result = "这个公式没有算子实现"
        result += f"""
# **公式结果**
{func_result}
"""
    else:
        print("知识库中没有相关公式！")
    return result


def get_funcation_call_response(query):
    docs = read_md_files_from_knowledge_base()
    codes = read_py_files_from_knowledge_base()
    tools = read_json_files_from_knowledge_base()

    # 调用模型处理
    messages = [
        {
            "role": "system",
            "content": system_prompt.format(
                functions=[tool["content"] for tool in tools]
            ),
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    response = coder_client.chat.completions.create(
        model=API_KEY,
        messages=messages,
        temperature=0,
    )

    ans = response.choices[0].message.content
    print(f"Funcation Call Response: {ans}")
    name_and_arguments = parse_function_parameters(ans)
    func_name = name_and_arguments["name"]
    for tool in tools:
        if tool["name"] == func_name:
            file_name = tool["filename"]
            if name_and_arguments:
                return decorate_response(
                    file_name, func_name, name_and_arguments["arguments"], docs, codes
                )
    return ans


system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""

css = """
#qwen-md .katex-display { display: inline; }
#qwen-md .katex-display>.katex { display: inline; }
#qwen-md .katex-display>.katex>.katex-html { display: inline; }
"""

# 创建Gradio接口
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """\
<p align="center"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png" style="height: 60px"/><p>"""
        """<center><font size=8>📖 Qwen2.5-Coder Demo</center>"""
        """\
<center><font size=3>This WebUI is based on Qwen2.5-Coder-7B-Instruct for formula operator.</center>"""
    )
    state = gr.State({"tab_index": 0})
    with gr.Row():
        with gr.Column():
            formula_text = gr.Textbox(
                label="Input Formula Description (Edit directly)",
                value="""
计算空气相对湿度的数值？

假设已知以下参数：
空气干球温度为25∘C
空气湿球温度24∘C
大气压力为1013kPa
空气温度等于空气干球温度时的饱和水蒸气分压力为6.92kPa
空气温度等于空气湿球温度时的饱和水蒸气分压力为6.83kPa
                """,
            )

            with gr.Row():
                with gr.Column():
                    clear_btn = gr.ClearButton([formula_text])
                with gr.Column():
                    submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            # output_code = gr.Code(
            #     label="Generated Result",
            #     language="markdown",
            #     elem_id="qwen-code",
            # )
            output_md = gr.Markdown(
                label="answer",
                line_breaks=True,
                latex_delimiters=[
                    {
                        "left": "$$",
                        "right": "$$",
                        "display": True,
                    },
                    {
                        "left": "$",
                        "right": "$",
                        "display": True,
                    },
                ],
                elem_id="qwen-md",
            )
    submit_btn.click(
        fn=get_funcation_call_response,
        inputs=[formula_text],
        outputs=output_md,
    )

args = get_args()
demo.launch(
    share=args.share,
    inbrowser=args.inbrowser,
    server_port=args.server_port,
    server_name=args.server_name,
)
