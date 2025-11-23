from flask import Flask, jsonify
from flask_cors import CORS
import agentpy as ap
import numpy as np
import heapq

# ==============================================================
#                 🚜   MODELO COMPLETO DE TRACTORES
# ==============================================================

class ObstacleTractor(ap.Agent):
    """Static or slowly moving obstacle tractor"""

    def setup(self):
        self.width = self.p.obstacle_width
        self.length = self.p.obstacle_length
        self.velocity = np.array([0.0, 0.0])

        # 50% move slowly
        if self.model.nprandom.random() < 0.5:
            angle = self.model.nprandom.random() * 2 * np.pi
            speed = self.p.obstacle_speed
            self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed

    def setup_pos(self, space):
        self.space = space
        self.pos = space.positions[self]

    def update_position(self):
        if np.linalg.norm(self.velocity) > 0:
            new_pos = self.pos + self.velocity

            # Bounce off boundaries
            for i in range(2):
                if new_pos[i] < 0 or new_pos[i] > self.space.shape[i]:
                    self.velocity[i] *= -1
                    new_pos[i] = np.clip(new_pos[i], 0, self.space.shape[i])

            self.space.move_to(self, new_pos)


class MainTractor(ap.Agent):
    """Main tractor that navigates with A*"""

    def setup(self):
        self.width = self.p.tractor_width
        self.length = self.p.tractor_length
        self.path = []
        self.path_index = 0
        self.reached_target = False
        self.total_distance = 0
        self.replan_counter = 0

    def setup_pos(self, space):
        self.space = space
        self.pos = space.positions[self]
        self.start = np.array(self.pos)
        self.target = np.array([self.p.size - 5, self.p.size - 5])

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_collision(self, pos):
        for obstacle in self.model.obstacles:
            obs_pos = obstacle.pos
            dist = np.linalg.norm(pos - obs_pos)
            if dist < (self.width + obstacle.width) / 2 + self.p.safety_margin:
                return True
        return False

    def get_neighbors(self, pos):
        neighbors = []
        dirs = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        for dx, dy in dirs:
            new = (pos[0] + dx, pos[1] + dy)
            if 0 <= new[0] < self.p.size and 0 <= new[1] < self.p.size:
                new_arr = np.array([float(new[0]), float(new[1])])
                if not self.is_collision(new_arr):
                    neighbors.append(new)

        return neighbors

    def a_star_search(self, start, goal):
        s = (int(start[0]), int(start[1]))
        g = (int(goal[0]), int(goal[1]))

        open_set = []
        heapq.heappush(open_set, (0, s))

        came_from = {}
        g_cost = {s: 0}
        f_cost = {s: self.heuristic(s, g)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if self.heuristic(current, g) < 2:
                path = []
                while current in came_from:
                    path.append(np.array([float(current[0]), float(current[1])]))
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for nb in self.get_neighbors(current):
                tentative = g_cost[current] + 1

                if nb not in g_cost or tentative < g_cost[nb]:
                    came_from[nb] = current
                    g_cost[nb] = tentative
                    f_cost[nb] = tentative + self.heuristic(nb, g)
                    heapq.heappush(open_set, (f_cost[nb], nb))

        return []

    def plan_path(self):
        self.path = self.a_star_search(self.pos, self.target)
        self.path_index = 0
        self.replan_counter += 1

    def update_position(self):
        if self.reached_target:
            return

        if self.path_index < len(self.path):
            next_pos = self.path[self.path_index]
            if self.is_collision(next_pos):
                self.plan_path()
                return

        if not self.path or self.path_index >= len(self.path):
            if np.linalg.norm(self.pos - self.target) < 2:
                self.reached_target = True
                print("Reached target!")
                return
            self.plan_path()
            return

        next_pos = self.path[self.path_index]
        dist = np.linalg.norm(next_pos - self.pos)
        self.total_distance += dist

        self.space.move_to(self, next_pos)
        self.path_index += 1


class TractorModel(ap.Model):
    def setup(self):
        self.space = ap.Space(self, shape=[self.p.size, self.p.size])

        # Obstacles
        self.obstacles = ap.AgentList(self, self.p.n_obstacles, ObstacleTractor)
        self.space.add_agents(self.obstacles, random=True)
        self.obstacles.setup_pos(self.space)

        # Tractor
        self.tractor = MainTractor(self)
        self.space.add_agents([self.tractor], positions=[np.array([5.0, 5.0])])
        self.tractor.setup_pos(self.space)

        # Initial A*
        self.tractor.plan_path()

    def step(self):
        self.obstacles.update_position()
        self.tractor.update_position()


# ==============================================================
#                       🌐  API FLASK
# ==============================================================

app = Flask(__name__)
CORS(app)

# Parameters
parameters = {
    'size': 50,
    'seed': 123,
    'steps': 60,
    'n_obstacles': 15,
    'tractor_width': 2,
    'tractor_length': 2,
    'obstacle_width': 2.5,
    'obstacle_length': 2.5,
    'obstacle_speed': 0.05,
    'safety_margin': 1.5
}

# Init model
model = TractorModel(parameters)
model.run(steps=1)


@app.route("/state")
def get_state():
    t = model.tractor

    data = {
        "tractor": {
            "x": float(t.pos[0]),
            "y": float(t.pos[1]),
            "width": float(t.width),
            "length": float(t.length),
        },
        "obstacles": [
            {
                "x": float(o.pos[0]),
                "y": float(o.pos[1]),
                "width": float(o.width),
                "length": float(o.length)
            }
            for o in model.obstacles
        ],
        "path": [[float(p[0]), float(p[1])] for p in t.path],
        "map_size": model.p.size
    }

    return jsonify(data)


@app.route("/step", methods=["POST"])
def step():
    model.step()
    return jsonify({"message": f"Simulation advanced to t={model.t}"})


@app.route("/reset", methods=["POST"])
def reset():
    global model
    model = TractorModel(parameters)
    model.run(steps=1)
    return jsonify({"message": "Simulation reset."})


if __name__ == "__main__":
    app.run(debug=True)