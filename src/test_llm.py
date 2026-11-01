from llm import ask_llm

out = ask_llm("Who directed the movie Titanic? Answer in a few words.", with_meta=True)
print(out)