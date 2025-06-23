import pandas as pd


faq_bank = pd.read_csv("data/FAQ_Bank.csv")
faq_eval = pd.read_csv("data/FAQ_Bank_eval.csv")
user_queries = pd.read_csv("data/User_Query_Bank.csv")
annotated_relevance = pd.read_csv("data/Annotated_Relevance_Set.csv")


print("FAQ Bank Columns:", faq_bank.columns)
print(faq_bank.head())

print("\nFAQ Eval Columns:", faq_eval.columns)
print(faq_eval.head())

print("\nUser Queries Columns:", user_queries.columns)
print(user_queries.head())

print("\nAnnotated Relevance Columns:", annotated_relevance.columns)
print(annotated_relevance.head())

