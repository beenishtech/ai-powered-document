import os
import time
from dotenv import load_dotenv
from groq import Groq

# 1. Environment variables load karein
load_dotenv()

# Groq client initialize karein
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Assignment ke mutabiq 5 questions
questions = [
    "What is an intelligent agent in AI?",
    "What are Python's primary design principles?",
    "What is the difference between supervised and unsupervised learning?",
    "How do public clouds differ from private clouds?",
    "What is the main purpose of Retrieval-Augmented Generation (RAG)?",
]


# ==========================================
# 1. ZERO-SHOT PROMPTING
# ==========================================
def run_zero_shot(question):
  completion = client.chat.completions.create(
      model="openai/gpt-oss-safeguard-20b",
      messages=[{"role": "user", "content": question}],
      temperature=0.3,
  )
  return completion.choices[0].message.content


# ==========================================
# 2. FEW-SHOT PROMPTING (Sirf 1 Example)
# ==========================================
def run_few_shot(question):
  prompt_with_examples = f"""
    Example:
    Question: What is Artificial Intelligence?
    Answer: Artificial Intelligence is a branch of computer science focused on creating smart machines that perform human-like tasks.

    Now, answer the following question in a similar clear, concise manner:
    Question: {question}
    Answer:
    """
  completion = client.chat.completions.create(
      model="openai/gpt-oss-safeguard-20b",
      messages=[{"role": "user", "content": prompt_with_examples}],
      temperature=0.3,
  )
  return completion.choices[0].message.content


# ==========================================
# 3. ROLE-BASED PROMPTING
# ==========================================
def run_role_based(question):
  completion = client.chat.completions.create(
      model="openai/gpt-oss-safeguard-20b",
      messages=[
          {
              "role": "system",
              "content": (
                  "You are an expert Technical Document Analyst. Provide"
                  " professional, precise, and well-structured answers based"
                  " strictly on technical facts."
              ),
          },
          {"role": "user", "content": question},
      ],
      temperature=0.3,
  )
  return completion.choices[0].message.content


# ==========================================
# TESTING AND COMPARING (With Delay)
# ==========================================
if __name__ == "__main__":
  print("=" * 60)
  print("STARTING PROMPT ENGINEERING EVALUATION (PART 1)")
  print("=" * 60)

  for i, q in enumerate(questions, 1):
    print(f"\n\n--- QUESTION {i}: {q} ---")

    print("\n[1] Zero-Shot Response:")
    print(run_zero_shot(q))
    time.sleep(2)  # Rate limit se bachne ke liye 2 second ka gap

    print("\n[2] Few-Shot Response:")
    print(run_few_shot(q))
    time.sleep(2)

    print("\n[3] Role-Based Response:")
    print(run_role_based(q))
    time.sleep(2)
    print("-" * 60)