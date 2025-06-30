import os
import sys
import httpx
from openai import OpenAI

# Ensure the OpenAI API key is set in the environment
if "OPENAI_API_KEY" not in os.environ:
    print("Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

proxy_url = 'http://10.231.139.3:7890'
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=httpx.Client(proxies={"http://": proxy_url, "https://": proxy_url}))

def video_caption(user_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "**Objective**: **Give a highly descriptive 6-second video caption based on input image and user input. **. As an expert, delve deep into the image with a discerning eye, leveraging rich creativity, meticulous thought. When describing the details of an image, include appropriate dynamic information to ensure that the video caption contains reasonable actions and plots. If user input is not empty, then the caption should be expanded according to the user's input.\n\n**Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image. User input is optional and can be empty.\n\n**Note**: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!!\n\n**Answering Style**: Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like \"The image/video showcases\" \"The photo captures\" and more. For example, say \"A woman is on a beach\", instead of \"A woman is depicted in the image\".\n\n**Output Format**: \"[highly descriptive image caption here]\""
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.02,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_prompt = input("Enter your prompt (or 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
        print(video_caption(user_prompt))