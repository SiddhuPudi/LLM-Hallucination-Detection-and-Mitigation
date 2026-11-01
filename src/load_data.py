import pandas as pd
df = pd.read_json("data/qa_data.json", lines=True)
print(df.shape)
print(df.columns.tolist())   # knowledge, question, right_answer, hallucinated_answer
for i in range(3):
    print(df.iloc[i].to_dict(), "\n")