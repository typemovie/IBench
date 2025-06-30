gpt4v_complex = "You are my assistant to evaluate the correspondence of the image to a given text prompt. \
                focus on the objects in the image and their attributes (such as color, shape, texture), spatial layout and action relationships. \
                According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{prompt_name}\"  \
                        Give a score from 0 to 100, according the criteria: \n\
                        5: the image perfectly matches the content of the text prompt, with no discrepancies. \
                        4: the image portrayed most of the actions, events and relationships but with minor discrepancies. \
                        3: the image depicted some elements in the text prompt, but ignored some key parts or details. \
                        2: the image did not depict any actions or events that match the text. \
                        1: the image failed to convey the full scope in the text prompt. \
                        Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."

prompt_dict = {
    "gpt4v_complex": gpt4v_complex
}
