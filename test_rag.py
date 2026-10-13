
from rag_pipeline import answer_question

question = "What should a customer do if their product arrives damaged?"

result = answer_question(question)

print("\nANSWER")
print(result["answer"])

print("\nSOURCES")
for source in result["sources"]:
    print(f"\nFile: {source['filename']}")
    print(source["content"])
