import re
import httpx
import base64
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from utils.log import logger
from utils.compat import config
from utils.builder import METRICS
from config.prompt import prompt_dict

GPT_PROXY_URL = config.metrics.mllm.gpt_proxy_url
TEMPERATURE = config.metrics.mllm.temperature
MAX_TOKENS = config.metrics.mllm.max_tokens
TOP_P = config.metrics.mllm.top_p
FREQUENCY_PENALTY = config.metrics.mllm.frequency_penalty
PRESENCE_PENALTY = config.metrics.mllm.presence_penalty


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@METRICS.register_module()
class GPT(object):
    def __init__(self,
                 model="gpt-40-mini",
                 user_prompt="",
                 type=""):
        self.model = model
        self.user_prompt = user_prompt

        # ------------------------------------------------------------------------------------------------------------------
        logger.info("gpt client initialization!!!")
        self.client = OpenAI(http_client=httpx.Client(proxy=GPT_PROXY_URL))

    def predict(self, user_prompt, base64_image):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_dict["gpt4v_complex"]
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY,
            response_format={
                "type": "text"
            }
        )
        return response.choices[0].message.content

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        gpt_scores_list = []
        if category == "imageid":
            for data in tqdm(data_list):
                imagewithid = data['imagewithid']
                imagewithid = encode_image(imagewithid)
                prompt = data['prompt']
                user_prompt = self.user_prompt + prompt
                # import pdb;pdb.set_trace()
                reponse = self.predict(user_prompt, imagewithid)
                pattern = r'"score": (\d+),'
                score_strings = re.findall(pattern, reponse)
                gpt_scores_list.append(int(score_strings))

        cur_avg = np.mean(gpt_scores_list)
        return cur_avg
