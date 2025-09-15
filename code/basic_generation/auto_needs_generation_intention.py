import json
from openai import OpenAI

client = OpenAI(
    api_key="sk-089c25df7bec40f2a83a377aa243af74",
    base_url="https://api.deepseek.com",
)

system_prompt = """
The user will give you an instruction to output several pieces of text. Please output all text pieces in JSON format.

EXAMPLE INPUT:
Give me eight pieces of recommendations in the context of being a new student in school.

EXAMPLE JSON OUTPUT:
{
    "piece 1": "......",
    "piece 2": "......",
    "piece 3": "......",
    ......
    "piece 8": "......",
}
"""
product="backpack"
#user_prompt = "Please give me ten uncommon scene descriptions for the product of comb."
#user_prompt = "Please give me five real-life scene descriptions where the product of comb is unlikely to appear. In each real-life scene description, you should not mention the product and you should include some plot or story in the description."
#user_prompt = "Please give me five real-life intention descriptions which are unlikely to be the intentions of the product of comb. In each real-life intention description, you should not mention the product and you should include some plot or story in the description."
user_prompt = "Please give me ten real-life intention descriptions which are unlikely to be the intentions of the product of "+product+"."

print(user_prompt)


messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={
        'type': 'json_object'
    },
    temperature=1.5
)

print(json.loads(response.choices[0].message.content))