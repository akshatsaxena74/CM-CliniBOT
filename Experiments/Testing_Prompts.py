import pandas as pd

df = pd.read_csv("...")

def build_prompt(row):
    return f"""You are an expert fact-checker.

Question: {row['Question']}

ChatGPT Response: {row['Model_Answer']}

Please analyze the above response and point out any hallucinated (factually incorrect or misleading) parts. Justify your assessment."""

df["Prompt"] = df.apply(build_prompt, axis=1)

for i in range(3):
    print(df.loc[i, "Prompt"])
    print("-" * 80)
