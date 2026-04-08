from fastapi import FastAPI
from env import HallucinationEnv, Action

app = FastAPI()
env = HallucinationEnv()

@app.post("/reset")
def reset():
    """Auto-grader calls this to start the environment"""
    obs = env.reset()
    return obs

@app.post("/step")
def step(action: Action):
    """Auto-grader calls this to submit answers"""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.post("/state")
def state():
    """Auto-grader calls this to check internal variables"""
    return env.state()

# Added GET method for basic health check so Hugging Face knows it's running
@app.get("/")
def health_check():
    return {"status": "Environment is running and ready for evaluation."}