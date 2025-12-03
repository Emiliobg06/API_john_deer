# tractor_api_flask_qcomm.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import agentpy as ap
import numpy as np
import random
from collections import defaultdict


class QAgent(ap.Agent):

    def setup(self):
        self.size = self.p.size
        self.width = self.p.tractor_width
        self.length = self.p.tractor_length

        self.alpha = self.p.q_alpha
        self.gamma = self.p.q_gamma
        self.epsilon = self.p.q_epsilon
        self.epsilon_min = self.p.q_epsilon_min
        self.epsilon_decay = self.p.q_epsilon_decay

        self.actions = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        self.q = defaultdict(lambda: np.zeros(len(self.actions)))

        self.reached = False
        self.total_distance = 0.0
        self.visited = defaultdict(int)

    def setup_pos(self, space):
        self.space = space
        self.pos = space.positions[self]  # float position
        self.cell = (int(round(self.pos[0])), int(round(self.pos[1])))
        self.target = (int(self.p.size - 2), int(self.p.size - 2))

    def state(self):
        return (int(round(self.pos[0])), int(round(self.pos[1])))

    def choose_action(self, s, occupied_cells):
        if random.random() < self.epsilon:
            random.shuffle(self.actions)
            best = None
            best_score = -99999
            for i, a in enumerate(self.actions):
                cand = (s[0] + a[0], s[1] + a[1])
                if 0 <= cand[0] < self.size and 0 <= cand[1] < self.size:
                    score = -self.p.shared_memory.get(cand, 0) * 30  # increased weight to avoid overlap
                    if cand not in occupied_cells:
                        score += 20
                    if score > best_score:
                        best_score = score
                        best = i
            if best is not None:
                return best
            
            return random.choice(range(len(self.actions)))
        else:
            # En explotación, también considerar shared_memory para evitar overlap
            qvals = self.q[s].copy()
            # Penalizar acciones que llevan a celdas con alta memoria
            for i, a in enumerate(self.actions):
                cand = (s[0] + a[0], s[1] + a[1])
                if 0 <= cand[0] < self.size and 0 <= cand[1] < self.size:
                    memory_penalty = self.p.shared_memory.get(cand, 0) * 30
                    qvals[i] -= memory_penalty
            maxv = np.max(qvals)
            best_actions = [i for i, q in enumerate(qvals) if q == maxv]
            return random.choice(best_actions)

    def update_q(self, s, a_idx, r, s2):
        old = self.q[s][a_idx]
        max_next = np.max(self.q[s2]) if s2 in self.q else 0.0
        self.q[s][a_idx] = old + self.alpha * (r + self.gamma * max_next - old)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

class QTractorModel(ap.Model):

    def setup(self):

        self.space = ap.Space(self, shape=[self.p.size, self.p.size])
        self.agents = ap.AgentList(self, self.p.n_agents, QAgent)

#        start_positions = []
#        for i in range(self.p.n_agents):
#            x = (i + 0.5) * (self.p.size / self.p.n_agents)
#            y = 1.0
#            start_positions.append(np.array([float(x), float(y)]))
        start_positions = []
        cols = self.p.n_agents
        spacing_x = self.p.size / cols
        for i in range(self.p.n_agents):
            x = (i + 0.5) * spacing_x
            y = random.uniform(0, 3)   # pequeña variación pero no aleatoria completa
            start_positions.append(np.array([float(x), float(y)]))

        self.space.add_agents(self.agents, positions=start_positions)
        self.agents.setup_pos(self.space)

        self.shared_visited = defaultdict(int)

        self.shared_memory = defaultdict(int)
        self.p.shared_memory = self.shared_memory  # para acceso dentro del agente
   # (x,y) -> count (all agents)
        # register initial visits
        for a in self.agents:
            c = a.state()
            self.shared_visited[c] += 1
            a.visited[c] += 1

    def step(self):
        """One simulation step: each agent chooses action, then resolve moves to avoid collision/overlap."""
        # Build current occupied cells by agents (their current cells)
        occupied_now = {a.state(): a for a in self.agents}

        # Each agent selects an intended action based on shared visited map
        intents = {}  # agent -> (next_cell, action_idx)
        occupied_cells = set(self.shared_visited.keys())  # cells recently visited by any agent (discourage overlap)
        for a in self.agents:
            if a.reached:
                intents[a] = (a.state(), None)
                continue

            s = a.state()
            action_idx = a.choose_action(s, occupied_cells)
            dx, dy = a.actions[action_idx]
            cand = (s[0] + dx, s[1] + dy)
            # clip to bounds
            cand = (max(0, min(self.p.size - 1, cand[0])), max(0, min(self.p.size - 1, cand[1])))
            intents[a] = (cand, action_idx)

        # Resolve conflicts: if multiple agents intend same cell => none moves (they wait) and get penalty for collision attempt
        # Also avoid moving into currently occupied cell by another agent (simultaneous swap prevented).
        target_counts = defaultdict(list)
        for a, (cand, ai) in intents.items():
            target_counts[cand].append(a)

        # Prepare rewards and apply moves
        rewards = {}
        for a in self.agents:
            s = a.state()
            cand, aidx = intents[a]
            # default reward
            r = -1.0  # small step penalty to encourage coverage efficiency

            if a.reached:
                # already at goal
                rewards[a] = 0.0
                continue

            # collision if candidate is occupied now by another agent (no simultaneous swap allowed)
            occupied_now_by_other = (cand in occupied_now) and (occupied_now[cand] is not a)
            multi_intent = len(target_counts[cand]) > 1

            if occupied_now_by_other or multi_intent:
                # collision/overlap attempt — penalize and do not move
                r += -150.0  # increased penalty to discourage collisions
                # update Q with next state = same state
                a_next_state = s
                a.update_q(s, aidx, r, a_next_state)
                rewards[a] = r
                # do NOT move
                continue

            shared_v = self.shared_visited.get(cand, 0)
            heat = self.shared_memory.get(cand, 0)
            # Penalización exponencial: más visitas = penalización mucho mayor
            r -= shared_v * 25.0  # base penalty
            r -= (shared_v ** 1.5) * 10.0  # penalty adicional exponencial para múltiples visitas
            r -= heat * 20.0  # increased heat penalty
            if shared_v == 0:
                r += 60.0  # increased reward for unvisited cells
            
            # Penalización acumulativa por proximidad (más agresiva)
            proximity_penalty = 0
            for other in self.agents:
                if other is not a:
                    ox, oy = other.state()
                    dist = abs(ox - cand[0]) + abs(oy - cand[1])
                    if dist <= 4:  # aumentar distancia de detección
                        # Penalización inversamente proporcional a la distancia
                        proximity_penalty += 30.0 / (dist + 1)
            r -= proximity_penalty


            # goal check
            if cand == a.target:
                r += 200.0  # big reward for reaching goal

            # apply move: update position
            new_pos = np.array([float(cand[0]), float(cand[1])])
            dist = np.linalg.norm(new_pos - a.pos)
            a.total_distance += dist
            # physically move in the space
            self.space.move_to(a, new_pos)
            a.visited[cand] += 1
            self.shared_visited[cand] += 1
            self.shared_memory[cand] += 1
            
            # Penalización adicional si la celda fue visitada recientemente por otros agentes
            if shared_v > 1:  # si fue visitada más de una vez
                r -= (shared_v - 1) * 30.0  # penalización extra por cada visita adicional

            # Q update: s -> cand
            a.update_q(s, aidx, r, cand)
            rewards[a] = r

            # reached?
            if cand == a.target:
                a.reached = True

        # return rewards optionally (not used)
        for a in self.agents:
            if hasattr(a, "last_cell") and a.last_cell == a.state():
                rewards[a] -= 15.0  # increased penalty for staying in same cell
            a.last_cell = a.state()
        
        # Decay epsilon for ALL agents in every step (even if they didn't move or reached goal)
        # This ensures consistent epsilon decay across all agents
        for a in self.agents:
            a.decay_epsilon()
        
        # Decay shared_memory to prevent infinite accumulation and reduce overlap over time
        # More aggressive decay: 0.995 means faster decay (0.5% per step)
        # After 500 steps: 0.995^500 ≈ 0.082 (92% reduction)
        decay_factor = 0.995  # more aggressive decay
        for cell in list(self.shared_memory.keys()):
            self.shared_memory[cell] *= decay_factor
            if self.shared_memory[cell] < 0.1:
                del self.shared_memory[cell]
            
        return rewards

    def reset(self):
        # reset positions and q-table if desired (we keep q-table to allow learning across runs unless user resets)
        start_positions = []
        for i in range(self.p.n_agents):
            x = (i + 0.5) * (self.p.size / self.p.n_agents)
            y = 1.0
            start_positions.append(np.array([float(x), float(y)]))
        # reset agent positions and flags
        for i, a in enumerate(self.agents):
            self.space.move_to(a, start_positions[i])
            a.reached = False
            a.total_distance = 0.0
            a.visited = defaultdict(int)
            # optionally reset epsilon to initial
            a.epsilon = self.p.q_epsilon

        # clear shared_visited and re-register
        self.shared_visited.clear()
        for a in self.agents:
            self.shared_visited[a.state()] += 1
            a.visited[a.state()] += 1

# -------------------------
# Flask API
# -------------------------
app = Flask(__name__)
CORS(app)

# parameters
parameters = {
    'size': 40,           # grid size (cells)
    'n_agents': 6,
    'tractor_width': 1,
    'tractor_length': 1,
    # Q hyperparams
    'q_alpha': 0.5, #0.6
    'q_gamma': 0.98, #0.95
    'q_epsilon': 1.0, #0.8
    'q_epsilon_min': 0.01, #0.05
    'q_epsilon_decay': 0.997 #0.995
}

# instantiate model
model = QTractorModel(parameters)
model.run(steps=1)  # initial setup

# helper to serialize some Q entries (sample)
def serialize_q(agent, max_entries=200):
    out = []
    for i, (s, vals) in enumerate(agent.q.items()):
        if i >= max_entries:
            break
        out.append({'state': s, 'q': vals.tolist()})
    return out

@app.route("/state")
def state():
    agents_data = []
    for idx, a in enumerate(model.agents):
        agents_data.append({
            'id': idx,
            'x': float(a.pos[0]),
            'y': float(a.pos[1]),
            'cell': a.state(),
            'reached': bool(a.reached),
            'epsilon': float(a.epsilon),
            'total_distance': float(a.total_distance)
        })
    data = {
        'map_size': model.p.size,
        'agents': agents_data,
        'shared_visited': [{'cell': list(k), 'count': v} for k,v in model.shared_visited.items()]
    }
    return jsonify(data)

@app.route("/agents")
def agents_info():
    res = []
    for i, a in enumerate(model.agents):
        res.append({
            'id': i,
            'cell': a.state(),
            'visited_counts': [{'cell': list(k), 'count': v} for k,v in a.visited.items()],
            'epsilon': float(a.epsilon)
        })
    return jsonify({'agents': res})

@app.route("/qtable")
def qtable():
    # returns small sample of each agent's Q-table
    out = {}
    for i, a in enumerate(model.agents):
        out[f'agent_{i}'] = serialize_q(a, max_entries=200)
    return jsonify({'qtables': out})

@app.route("/step", methods=['POST', 'GET'])
def step():
    # run one sim step
    rewards = model.step()
    # option: return rewards per agent (converted)
    rewards_out = {str(i): float(rewards.get(a, 0.0)) for i, a in enumerate(model.agents)}
    return jsonify({'message': f'step {model.t} executed', 'rewards': rewards_out})

@app.route("/reset", methods=['POST', 'GET'])
def reset():
    model.reset()
    return jsonify({'message': 'model reset'})

@app.route("/report")
def report():
    size = model.p.size
    total_cells = size * size

    # Coverage: cuántas celdas fueron visitadas al menos una vez
    visited_cells = len(model.shared_visited)

    # Overlap: celdas con más de 1 visita
    overlapped_cells = sum(1 for c in model.shared_visited.values() if c > 1)

    # Distancias por agente
    distances = [float(a.total_distance) for a in model.agents]
    total_distance = sum(distances)
    avg_distance = total_distance / len(distances)

    # Epsilons
    epsilons = [float(a.epsilon) for a in model.agents]

    report_data = {
        "coverage_percentage": (visited_cells / total_cells) * 100,
        "visited_cells": visited_cells,
        "total_cells": total_cells,

        "overlap_cells": overlapped_cells,
        "overlap_percentage": (overlapped_cells / visited_cells) * 100 if visited_cells > 0 else 0,

        "total_distance_all_agents": total_distance,
        "average_distance_per_agent": avg_distance,

        "epsilon_values": epsilons,

        "agents": [
            {
                "id": i,
                "epsilon": float(a.epsilon),
                "distance": float(a.total_distance),
                "unique_cells_visited": len(a.visited)
            }
            for i, a in enumerate(model.agents)
        ]
    }

    return jsonify(report_data)

if __name__ == "__main__":
    app.run(debug=True)