from argparse import ArgumentParser
import re
import gradio as gr
import warnings
from openai import OpenAI

# import pdfplumber
# from openai.types.chat import ChatCompletionChunk

# 忽视所有警告
warnings.filterwarnings("ignore")

API_KEY = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_URL = "http://192.168.0.79:7862/v1"

coder_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


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
    print("formula_text: {}".format(formula_text))
    response = coder_client.chat.completions.create(
        model=API_KEY,
        messages=messages,
        temperature=0,
    )

    ans = response.choices[0].message.content
    print(f"answer: {ans}")
    return extract_python_code(ans)


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
<center><font size=3>This WebUI is based on Qwen2.5-Coder-7B-Instruct for formula operator.</center>"""
    )
    state = gr.State({"tab_index": 0})
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="prompt", value=system_prompt)
            formula_text = gr.Textbox(
                label="Input Formula Description (Edit directly)",
                value="""空气相对湿度宜按下式计算: 

$$ \varphi = \frac{p^{\prime\prime}_{\theta} - 0.000662 p (\theta - \tau)}{p^{\prime\prime}_{\tau}} $$ 

式中: 
- $\varphi$ —— 空气相对湿度
- $\theta$ —— 空气干球温度(℃)
- $\tau$ —— 空气湿球温度(℃)
- $p$ —— 大气压力(kPa)
- $p^{\prime\prime}_{\theta}$ —— 空气温度等于 $\theta$ ℃ 时的饱和水蒸气分压力(kPa)
- $p^{\prime\prime}_{\tau}$ —— 空气温度等于 $\tau$ ℃ 时的饱和水蒸气分压力(kPa)""",
            )

            with gr.Row():
                with gr.Column():
                    clear_btn = gr.ClearButton([formula_text, prompt])
                with gr.Column():
                    submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            output_code = gr.Code(
                label="Generated Python Function",
                language="python",
                elem_id="qwen-code",
            )
    submit_btn.click(
        fn=get_coder_response,
        inputs=[formula_text, prompt],
        outputs=output_code,
    )

args = get_args()
demo.launch(
    share=args.share,
    inbrowser=args.inbrowser,
    server_port=args.server_port,
    server_name=args.server_name,
)
