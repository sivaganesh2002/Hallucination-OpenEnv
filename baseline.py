import os
from env import HallucinationEnv, Action
from langchain_openai import ChatOpenAI

# Task-specific system prompts so the agent actually tries to do the right thing
TASK1_PROMPT = (
    "You are a factual question answering assistant. "
    "Answer the question using ONLY information from the provided context. "
    "Be precise and concise. Do not add facts not present in the context."
)

TASK2_PROMPT = (
    "You are a scientific research assistant. "
    "Answer the question based strictly on the provided arxiv abstract. "
    "If the abstract does not contain enough information to answer, say so clearly. "
    "Do not speculate or add outside knowledge."
)

TASK3_PROMPT = (
    "You are a source conflict analyst. "
    "You will be given two sources that may contradict each other. "
    "Identify the conflict, decide which source is more reliable, and give a final answer. "
    "Cite which source you relied on (Source A or Source B)."
)

def run_baseline():
    print("\n========================================")
    print("   INITIALIZING HALLUCINATION EVALUATOR ")
    print("========================================")

    env       = HallucinationEnv()
    agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    print("\n[>] Fetching live data (Wikipedia + Arxiv + Conflict Injection)...")
    obs = env.reset()

    # ---------------------------------------------------------
    # Agent answers each task with a task-specific prompt
    # ---------------------------------------------------------
    print("\n[>] Agent answering Task 1 (Medium — Factual Wikipedia)...")
    ans1 = agent_llm.invoke(
        f"{TASK1_PROMPT}\n\nContext:\n{obs.ctx_task1}\n\nQuestion:\n{obs.q_task1}"
    ).content
    print(f"    Answer: {ans1[:120]}...")

    print("\n[>] Agent answering Task 2 (Hard — Arxiv QA)...")
    ans2 = agent_llm.invoke(
        f"{TASK2_PROMPT}\n\nContext:\n{obs.ctx_task2}\n\nQuestion:\n{obs.q_task2}"
    ).content
    print(f"    Answer: {ans2[:120]}...")

    print("\n[>] Agent answering Task 3 (Expert — Conflicting Sources)...")
    ans3 = agent_llm.invoke(
        f"{TASK3_PROMPT}\n\nContext:\n{obs.ctx_task3}\n\nQuestion:\n{obs.q_task3}"
    ).content
    print(f"    Answer: {ans3[:120]}...")

    # ---------------------------------------------------------
    # Build and submit the action
    # ---------------------------------------------------------
    action = Action(
        ans_task1=ans1,
        citations_1=["Wikipedia context provided in prompt"],
        ans_task2=ans2,
        citations_2=["Arxiv abstract provided in prompt"],
        ans_task3=ans3,
        citations_3=["Source A and Source B provided in prompt"],
    )

    print("\n[>] Submitting answers to LangGraph evaluator (3 metrics running in parallel)...")
    next_obs, reward, done, info = env.step(action)

    # ---------------------------------------------------------
    # Print the report card
    # ---------------------------------------------------------
    print("\n========================================")
    print("         EVALUATION REPORT CARD         ")
    print("========================================")

    print(f"\n[ Level 1: Medium  (Factual)  ]  Score: {reward.task1_score:.2f} / 1.0")
    print("  Breakdown ->", reward.metrics_breakdown["Level_1_Metrics"])

    print(f"\n[ Level 2: Hard    (Arxiv QA) ]  Score: {reward.task2_score:.2f} / 1.0")
    print("  Breakdown ->", reward.metrics_breakdown["Level_2_Metrics"])

    print(f"\n[ Level 3: Expert  (Conflict) ]  Score: {reward.task3_score:.2f} / 1.0")
    print("  Breakdown ->", reward.metrics_breakdown["Level_3_Metrics"])

    print("\n========================================")
    print(f"  OVERALL AGGREGATE REWARD: {reward.value:.2f} / 1.0")
    print("========================================\n")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: Set your OPENAI_API_KEY environment variable first.")
        print("  Windows PowerShell : $env:OPENAI_API_KEY = 'sk-...'")
        print("  Linux / Mac        : export OPENAI_API_KEY=sk-...")
    else:
        run_baseline()
