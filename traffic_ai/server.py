"""
Flask API server for the Traffic AI Dashboard.
Serves the React frontend and exposes real-time simulation endpoints.
"""

import sys, os, threading, time, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, Response, send_from_directory
import numpy as np

from environment.traffic_env import TrafficEnv, NUM_INTERSECTIONS, NUM_DIRECTIONS, MIN_GREEN_STEPS
from agent.q_agent import QLearningAgent
from agent.fixed_time import FixedTimeController
from training.train import train as run_training

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")

# ── Global State ──────────────────────────────────────────────────────────────
state = {
    "agent":           None,
    "training_done":   False,
    "training_log":    [],       # list of {ep, reward, wait, queue, eps}
    "sim_running":     False,
    "sim_thread":      None,
    "sim_data":        None,     # latest simulation snapshot
    "mode":            "rl",     # "rl" | "fixed"
    "arrival_rate":    0.8,
    "sim_env":         None,
    "sim_controller":  None,
    "sim_history":     collections.deque(maxlen=120),  # last 120 steps
    "step_count":      0,
    "rl_cumulative":   {"wait": 0, "throughput": 0, "steps": 0},
    "fx_cumulative":   {"wait": 0, "throughput": 0, "steps": 0},
}

LOCK = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_snapshot(env: TrafficEnv, reward: float):
    intersections = []
    for i, inter in enumerate(env.intersections):
        queues = [inter.queue_len(d) for d in range(NUM_DIRECTIONS)]
        waits  = [inter.avg_wait(d) * 100 for d in range(NUM_DIRECTIONS)]  # un-normalize
        intersections.append({
            "id":           i,
            "phase":        inter.phase,
            "phase_locked": inter.steps_in_phase < MIN_GREEN_STEPS,
            "queues":       queues,       # raw counts
            "waits":        [round(w, 1) for w in waits],
            "throughput":   inter.total_throughput,
        })
    m = env.get_metrics()
    return {
        "step":         env.step_count,
        "reward":       round(reward, 4),
        "intersections": intersections,
        "total_queue":  m["total_queue"],
        "total_throughput": m["total_throughput"],
        "total_wait":   m["total_wait"],
    }

# ── Simulation loop (background thread) ───────────────────────────────────────
def sim_loop():
    with LOCK:
        arrival_rate = state["arrival_rate"]
        mode         = state["mode"]
        agent        = state["agent"]

    env = TrafficEnv(arrival_rate=arrival_rate, seed=99)
    env.reset()

    if mode == "rl" and agent is not None:
        controller = agent
        greedy = True
    else:
        controller = FixedTimeController(cycle_half=10)

    obs = env.reset()
    reward = 0.0

    while True:
        with LOCK:
            if not state["sim_running"]:
                break
            cur_mode  = state["mode"]
            cur_rate  = state["arrival_rate"]

        # If mode/rate changed, restart env
        if cur_mode != mode or cur_rate != arrival_rate:
            mode = cur_mode
            arrival_rate = cur_rate
            env = TrafficEnv(arrival_rate=arrival_rate, seed=99)
            obs = env.reset()
            if mode == "rl" and state["agent"] is not None:
                controller = state["agent"]
                greedy = True
            else:
                controller = FixedTimeController(cycle_half=10)
                if hasattr(controller, "reset"):
                    controller.reset()
            reward = 0.0

        if isinstance(controller, QLearningAgent):
            action = controller.select_action(obs, greedy=True)
        else:
            action = controller.select_action(obs)

        obs, reward, _ = env.step(action)
        snap = make_snapshot(env, reward)

        with LOCK:
            state["sim_data"] = snap
            state["sim_history"].append({
                "step":       snap["step"],
                "reward":     snap["reward"],
                "total_queue": snap["total_queue"],
                "total_wait": snap["total_wait"],
                "throughput": snap["total_throughput"],
            })
            state["step_count"] += 1

        time.sleep(0.25)   # 4 steps/sec — watchable

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    with LOCK:
        return jsonify({
            "training_done": state["training_done"],
            "sim_running":   state["sim_running"],
            "mode":          state["mode"],
            "arrival_rate":  state["arrival_rate"],
            "agent_ready":   state["agent"] is not None,
            "train_episodes": len(state["training_log"]),
        })

@app.route("/api/train", methods=["POST"])
def api_train():
    """Start training in background thread."""
    def do_train():
        log = []
        from environment.traffic_env import TrafficEnv
        from agent.q_agent import QLearningAgent as QA
        import numpy as np

        EPISODES  = 300
        EP_STEPS  = 200
        SEED      = 42
        np.random.seed(SEED)
        env   = TrafficEnv(arrival_rate=state["arrival_rate"], seed=SEED)
        agent = QA(seed=SEED)

        for ep in range(EPISODES):
            obs = env.reset()
            agent.update_lr(ep, EPISODES)
            ep_reward = 0.0
            for _ in range(EP_STEPS):
                action = agent.select_action(obs)
                next_obs, reward, _ = env.step(action)
                agent.update(obs, action, reward, next_obs)
                obs = next_obs
                ep_reward += reward
            agent.decay_epsilon()
            m = env.get_metrics()
            entry = {
                "ep":    ep + 1,
                "reward": round(ep_reward, 3),
                "wait":   round(m["total_wait"] / EP_STEPS, 2),
                "queue":  round(m["avg_queue_per_intersection"], 2),
                "eps":    round(agent.eps, 4),
            }
            log.append(entry)
            with LOCK:
                state["training_log"] = log[:]

        with LOCK:
            state["agent"]         = agent
            state["training_done"] = True

    t = threading.Thread(target=do_train, daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "Training started"})

@app.route("/api/training_log")
def api_training_log():
    with LOCK:
        return jsonify({
            "log":  state["training_log"],
            "done": state["training_done"],
        })

@app.route("/api/sim/start", methods=["POST"])
def api_sim_start():
    body = request.get_json(silent=True) or {}
    with LOCK:
        state["mode"]         = body.get("mode", state["mode"])
        state["arrival_rate"] = float(body.get("arrival_rate", state["arrival_rate"]))
        if state["sim_running"]:
            return jsonify({"ok": False, "message": "Already running"})
        state["sim_running"]  = True
        state["sim_history"]  = collections.deque(maxlen=120)

    t = threading.Thread(target=sim_loop, daemon=True)
    t.start()
    with LOCK:
        state["sim_thread"] = t
    return jsonify({"ok": True})

@app.route("/api/sim/stop", methods=["POST"])
def api_sim_stop():
    with LOCK:
        state["sim_running"] = False
    return jsonify({"ok": True})

@app.route("/api/sim/snapshot")
def api_sim_snapshot():
    with LOCK:
        snap    = state["sim_data"]
        history = list(state["sim_history"])
        mode    = state["mode"]
    if snap is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "mode": mode, "snapshot": snap, "history": history})

@app.route("/api/sim/config", methods=["POST"])
def api_sim_config():
    body = request.get_json(silent=True) or {}
    with LOCK:
        if "mode" in body:
            state["mode"] = body["mode"]
        if "arrival_rate" in body:
            state["arrival_rate"] = float(body["arrival_rate"])
    return jsonify({"ok": True})

# ── Serve React SPA ───────────────────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    dist = os.path.join(app.root_path, "frontend", "dist")
    full = os.path.join(dist, path)
    if path and os.path.exists(full):
        return send_from_directory(dist, path)
    return send_from_directory(dist, "index.html")

if __name__ == "__main__":
    port = 5000  # Change this to 5001 if port 5000 is in use
    print(f"[Traffic AI] Dashboard running at http://localhost:{port}")
    app.run(debug=False, port=port, threaded=True)
