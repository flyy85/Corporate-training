from argparse import ArgumentParser
import re
import time
import gradio as gr
import os
import warnings
from openai import OpenAI

# import pdfplumber
# from openai.types.chat import ChatCompletionChunk

# 忽视所有警告
warnings.filterwarnings("ignore")

API_KEY = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_URL = "http://192.168.0.79:7862/v1"
MODEL_NAME = API_KEY

coder_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)
OUTPUT_DIR = "knowledge_base"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


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
        "--server-port", type=int, default=7862, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="localhost", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def extract_python_code(text):
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_coder_response(formula_text, prompt):
    # 调用模型处理
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"""                
## Description 
{formula_text}

## Python Function
        """,
        },
    ]
    response = coder_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
    )

    ans = response.choices[0].message.content
    return extract_python_code(ans)


def process_md(file_path, prompt):
    name = os.path.basename(file_path)
    name = name.split(".")[0]
    print(f"---- processing {name} ----")
    try:
        with open(file_path, encoding="utf-8") as f:
            formula_text = f.read()
        python_code = get_coder_response(formula_text, prompt)
        # 写入本地文件
        print(f"{OUTPUT_DIR}/{name}.py")
        with open(f"{OUTPUT_DIR}/{name}.py", "w") as f:
            f.write(python_code)
        yield python_code

    except Exception as e:
        print(f"---- error ----\n{e}")
        yield str(e)


def code_chat(path, prompt):
    if not os.path.exists(path):
        return "input image folder path not exits."

    if os.path.isfile(path) and path.endswith(".md"):
        # 如果是文件直接处理
        yield from process_md(path, prompt)
    elif os.path.isdir(path):
        md_files = [
            f for f in os.listdir(path) if f.lower().endswith((".md", ".markdown"))
        ]
        total_files = len(md_files)

        for index, file in enumerate(md_files):
            file_path = os.path.join(path, file)
            image_description = next(process_md(file_path, prompt))

            progress = (index + 1) / total_files * 100
            yield f"# Progress: {progress:.2f}% ({index + 1}/{total_files})\n\n" + image_description

        time.sleep(1)
        yield "All md processed."


system_prompt = r'''
Objective: Refer to the following example to generate one python function based on description.
Requirements:
- All calculations must be implemented in only one python function, not two or more.
- Only generate one function written in python language, do not provide unnecessary explanations.
- All input parameters in the python function default to None, and the default values for constant parameters are provided according to the Description.
- Adds a null check for the input parameters and returns an error if any of it is null.


######################
-Example1-
######################

## Description
闪络时导线上产生的过电压的公式：

\[ U = 30 \times k \times \left( \frac{h}{d} \right) \times I \]

其中：
- \( U \) - 导线上产生的过电压
- \( I \) - 雷电流
- \( h \) - 导线离地高度
- \( k \) - 系数，取决于雷电流反击的速率
- \( d \) - 发生闪络点到导线距离


## Python Function
```python
def calculate_overvoltage(I=None, h=None, d=None, k=None):
    """
    闪络时计算导线上产生的过电压

    参数：
    I : 雷电流 (单位：安培)，默认值为 None
    h : 导线离地高度 (单位：米)，默认值为 None
    d : 闪络点到导线的距离 (单位：米)，默认值为 None
    k : 系数，取决于雷电流反击的速率，默认值为 None

    返回：导线上产生的过电压 (单位：伏特)，如果任何输入为 None，则返回 None
    """
    if I is None:
        return ValueError("雷电流,不可以为空值。")
    elif h is None:
        return ValueError("导线离地高度,不可以为空值。")
    elif d is None:
        return ValueError("闪络点到导线的距离,不可以为空值。")
    elif k is None:
        return ValueError("系数(取决于雷电流反击的速率),不可以为空值。")
    # 计算过电压
    U = 30 * k * (h / d) * I
    return U
```
'''


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
<center><font size=3>This WebUI is based on Qwen2.5-Coder for formula operator implementation.</center>"""
    )
    state = gr.State({"tab_index": 0})
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="prompt", value=system_prompt)
            formula_path = gr.Textbox(
                label="Input Formulas Dir Path(Edit directly)",
                value=os.path.abspath(OUTPUT_DIR),
            )
            with gr.Row():
                with gr.Column():
                    clear_btn = gr.ClearButton([formula_path, prompt])
                with gr.Column():
                    submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            output_code = gr.Code(
                label="Generated Python Function",
                language="python",
                elem_id="qwen-code",
            )
    submit_btn.click(
        fn=code_chat,
        inputs=[formula_path, prompt],
        outputs=output_code,
    )

args = get_args()
demo.launch(
    share=args.share,
    inbrowser=args.inbrowser,
    server_port=args.server_port,
    server_name=args.server_name,
)
