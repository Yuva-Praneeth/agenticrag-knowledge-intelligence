
from agents import ask_agentic_rag

# question = "What should a customer do if their product arrives damaged?"
question = "What is NovaMart's employee relocation policy?"
result = ask_agentic_rag(question)

print("\nANSWER:")
print(result["answer"])

print("\nVERIFICATION:")
print(result["verification"])

print("\nRETRY COUNT:")
print(result["retry_count"])

print("\nSOURCES:")
for source in result["sources"]:
    print("-", source["source"])
