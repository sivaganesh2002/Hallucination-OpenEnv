import os
import random
import json
from typing import TypedDict, Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# ==========================================
# 1. OPENENV PYDANTIC MODELS
# ==========================================
class Action(BaseModel):
    ans_task1: str = Field(description="Answer for Level 1 (Medium Wikipedia)")
    ans_task2: str = Field(description="Answer for Level 2 (Hard Arxiv)")
    ans_task3: str = Field(description="Answer for Level 3 (Expert Conflicting Docs)")
    citations_1: list[str] = Field(default=[], description="Quotes used for task 1")
    citations_2: list[str] = Field(default=[], description="Quotes used for task 2")
    citations_3: list[str] = Field(default=[], description="Quotes used for task 3")

class Observation(BaseModel):
    # FIX: field names now match what HallucinationEnv.reset() actually passes in
    ctx_task1: str = Field(description="Context for task 1")
    q_task1:   str = Field(description="Question for task 1")
    ctx_task2: str = Field(description="Context for task 2")
    q_task2:   str = Field(description="Question for task 2")
    ctx_task3: str = Field(description="Context for task 3")
    q_task3:   str = Field(description="Question for task 3")
    feedback:  str = Field(default="New evaluation battery loaded.")

class Reward(BaseModel):
    value:      float = Field(description="Combined continuous final reward (0.0 to 1.0)")
    task1_score: float = Field(description="Score for Level 1: Factual Check")
    task2_score: float = Field(description="Score for Level 2: Arxiv QA")
    task3_score: float = Field(description="Score for Level 3: Conflict Detection")
    metrics_breakdown: Dict[str, Any] = Field(description="Raw metric JSON per task")

# ==========================================
# 2. LANGGRAPH STATE & TOOLS
# ==========================================
class EnvState(TypedDict):
    phase:  str
    # Task contexts and questions
    ctx1: str;  q1: str
    ctx2: str;  q2: str
    ctx3: str;  q3: str
    # Agent answers and citations (have defaults so the graph can run without them on reset)
    ans1: str;  cite1: list
    ans2: str;  cite2: list
    ans3: str;  cite3: list
    # Evaluation results
    eval1: dict; eval2: dict; eval3: dict
    # Scores
    score1: float; score2: float; score3: float
    final_reward: float
    is_done: bool

llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
wiki_tool = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1200)
arxiv_tool = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1200)

WIKI_TOPICS  = ["Quantum computing", "Roman Empire", "CRISPR", "Black Holes"]
ARXIV_TOPICS = ["Large Language Models", "Reinforcement Learning", "Superconductors"]

# ==========================================
# 3. GENERATION NODES (Parallel)
# ==========================================
def gen_task1(state: EnvState) -> dict:
    topic = random.choice(WIKI_TOPICS)
    ctx   = f"Wiki [{topic}]:\n" + wiki_tool.run(topic)
    q     = llm.invoke(f"Based on this extract, generate a purely factual question: {ctx}").content
    return {"ctx1": ctx, "q1": q}

def gen_task2(state: EnvState) -> dict:
    topic = random.choice(ARXIV_TOPICS)
    ctx   = f"Arxiv [{topic}]:\n" + arxiv_tool.run(topic)
    q     = llm.invoke(
        f"Based on this abstract, generate a complex question about limitations/findings: {ctx}"
    ).content
    return {"ctx2": ctx, "q2": q}

def gen_task3(state: EnvState) -> dict:
    base_ctx    = wiki_tool.run(random.choice(WIKI_TOPICS))
    mutated_ctx = llm.invoke(
        f"Rewrite this slightly, changing one critical date/number to make it conflicting:\n{base_ctx}"
    ).content
    ctx = f"SOURCE A (Original):\n{base_ctx}\n\nSOURCE B (Conflicting):\n{mutated_ctx}"
    q   = llm.invoke(
        f"Generate a trick question that asks about the conflicting detail: {ctx}"
    ).content
    return {"ctx3": ctx, "q3": q}

def sync_generations(state: EnvState) -> dict:
    # Just marks that all three generation branches finished
    return {"phase": "waiting_for_agent"}

# ==========================================
# 4. EVALUATION NODES (Parallel)
# ==========================================
def evaluate_metrics(ctx: str, ans: str, citations: list) -> dict:
    """
    Asks gpt-4o-mini to score the answer on 5 dimensions.
    Falls back to zeros if the JSON parse fails for any reason.
    """
    prompt = ChatPromptTemplate.from_template(
        "Compare the Agent's Answer to the Source Context. "
        "Provide 5 float scores between 0.0 and 1.0.\n\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Scoring criteria:\n"
        "  InfoNCE    — Is the answer semantically anchored to the context?\n"
        "  NLI        — Does the answer align with the context? (1.0 = aligns, 0.0 = contradicts)\n"
        "  Entity     — Are all named entities (people, places, orgs) correct?\n"
        "  Numbers    — Are all statistics, dates, and figures accurate?\n"
        "  Uncertainty — Did the model appropriately admit uncertainty when the context was unclear?\n\n"
        "Output ONLY a raw JSON dict with exactly these keys: "
        "InfoNCE, NLI, Entity, Numbers, Uncertainty. No extra text."
    )
    try:
        raw    = (prompt | llm).invoke({"context": ctx, "answer": ans}).content
        # Strip markdown code fences if the model wraps the JSON anyway
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)
        # Make sure all expected keys exist — fill missing ones with 0
        for key in ["InfoNCE", "NLI", "Entity", "Numbers", "Uncertainty"]:
            scores.setdefault(key, 0.0)
        # Clamp everything to [0, 1] just in case
        scores = {k: max(0.0, min(1.0, float(v))) for k, v in scores.items()}
    except Exception as e:
        print(f"    [!] Metric parsing failed ({e}), defaulting to zeros.")
        scores = {"InfoNCE": 0.0, "NLI": 0.0, "Entity": 0.0, "Numbers": 0.0, "Uncertainty": 0.0}

    scores["Citations"] = 1.0 if len(citations) > 0 else 0.0
    return scores

def eval_task1(state: EnvState) -> dict:
    return {"eval1": evaluate_metrics(state["ctx1"], state["ans1"], state["cite1"])}

def eval_task2(state: EnvState) -> dict:
    return {"eval2": evaluate_metrics(state["ctx2"], state["ans2"], state["cite2"])}

def eval_task3(state: EnvState) -> dict:
    return {"eval3": evaluate_metrics(state["ctx3"], state["ans3"], state["cite3"])}

def aggregate_rewards(state: EnvState) -> dict:
    e1, e2, e3 = state["eval1"], state["eval2"], state["eval3"]

    # Task 1 (Factual): entity accuracy + number accuracy matter most
    s1 = round((e1["Entity"] * 0.4) + (e1["Numbers"] * 0.4) + (e1["NLI"] * 0.2), 3)

    # Task 2 (Arxiv QA): grounding + calibration matter most
    s2 = round((e2["InfoNCE"] * 0.5) + (e2["Uncertainty"] * 0.4) + (e2["Citations"] * 0.1), 3)

    # Task 3 (Conflict): contradiction avoidance + grounding + citations
    s3 = round((e3["NLI"] * 0.6) + (e3["InfoNCE"] * 0.2) + (e3["Citations"] * 0.2), 3)

    return {
        "score1":       s1,
        "score2":       s2,
        "score3":       s3,
        "final_reward": round((s1 + s2 + s3) / 3.0, 3),
        "is_done":      True,
        "phase":        "complete",
    }

# ==========================================
# 5. GRAPH ROUTING
# ==========================================
workflow = StateGraph(EnvState)

for node_name, node_fn in [
    ("gen_t1",    gen_task1),
    ("gen_t2",    gen_task2),
    ("gen_t3",    gen_task3),
    ("sync_gen",  sync_generations),
    ("eval_t1",   eval_task1),
    ("eval_t2",   eval_task2),
    ("eval_t3",   eval_task3),
    ("aggregate", aggregate_rewards),
]:
    workflow.add_node(node_name, node_fn)

def route_phase(state: EnvState) -> list:
    if state["phase"] == "generate":
        return ["gen_t1", "gen_t2", "gen_t3"]
    else:
        return ["eval_t1", "eval_t2", "eval_t3"]

workflow.add_conditional_edges(START, route_phase)
workflow.add_edge(["gen_t1", "gen_t2", "gen_t3"], "sync_gen")
workflow.add_edge("sync_gen", END)
workflow.add_edge(["eval_t1", "eval_t2", "eval_t3"], "aggregate")
workflow.add_edge("aggregate", END)

graph = workflow.compile()

# ==========================================
# 6. OPENENV WRAPPER
# ==========================================
class HallucinationEnv:
    def __init__(self):
        self.state_data: dict = {}

    def reset(self) -> Observation:
        # Provide empty defaults for fields the generate phase doesn't touch
        # so LangGraph's TypedDict validation doesn't complain about missing keys.
        initial_state = {
            "phase":        "generate",
            "ctx1": "", "q1": "",
            "ctx2": "", "q2": "",
            "ctx3": "", "q3": "",
            "ans1": "", "cite1": [],
            "ans2": "", "cite2": [],
            "ans3": "", "cite3": [],
            "eval1": {}, "eval2": {}, "eval3": {},
            "score1": 0.0, "score2": 0.0, "score3": 0.0,
            "final_reward": 0.0,
            "is_done": False,
        }
        result = graph.invoke(initial_state)
        self.state_data = result

        # FIX: map state dict keys (ctx1/q1...) to Observation field names (ctx_task1/q_task1...)
        return Observation(
            ctx_task1=result["ctx1"], q_task1=result["q1"],
            ctx_task2=result["ctx2"], q_task2=result["q2"],
            ctx_task3=result["ctx3"], q_task3=result["q3"],
            feedback="Generated fresh tasks from live Wikipedia + Arxiv.",
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.state_data.update({
            "phase": "evaluate",
            "ans1":  action.ans_task1,  "cite1": action.citations_1,
            "ans2":  action.ans_task2,  "cite2": action.citations_2,
            "ans3":  action.ans_task3,  "cite3": action.citations_3,
        })

        eval_result = graph.invoke(self.state_data)
        self.state_data.update(eval_result)

        obs = Observation(
            ctx_task1=self.state_data["ctx1"], q_task1=self.state_data["q1"],
            ctx_task2=self.state_data["ctx2"], q_task2=self.state_data["q2"],
            ctx_task3=self.state_data["ctx3"], q_task3=self.state_data["q3"],
            feedback="All 3 tasks evaluated.",
        )

        reward = Reward(
            value=       self.state_data["final_reward"],
            task1_score= self.state_data["score1"],
            task2_score= self.state_data["score2"],
            task3_score= self.state_data["score3"],
            metrics_breakdown={
                "Level_1_Metrics": self.state_data["eval1"],
                "Level_2_Metrics": self.state_data["eval2"],
                "Level_3_Metrics": self.state_data["eval3"],
            },
        )

        return obs, reward, self.state_data["is_done"], {"msg": "Evaluated successfully."}

    def state(self) -> dict:
        return self.state_data
