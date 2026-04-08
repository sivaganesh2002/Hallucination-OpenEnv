import uvicorn
from fastapi import FastAPI
from env import HallucinationEnv, Action

app = FastAPI()
env_instance = HallucinationEnv()

@app.post("/reset")
def reset():
    return env_instance.reset()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env_instance.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.post("/state")
def state():
    return env_instance.state()

@app.get("/")
def health_check():
    return {"status": "Environment is running and ready for evaluation."}

def main():
    """This is the entry point the OpenEnv grader is looking for"""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
