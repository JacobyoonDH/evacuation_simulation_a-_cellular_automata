import random
from aa import a_star
import random
import numpy as np
import random
import numpy as np
from aa import a_star

def find_optimal_goal(current_position, evacuation_target_cells, dynamic_congestion_grid):
    if not evacuation_target_cells:
        raise ValueError("Evacuation target cells are empty.")
    
    def goal_score(target):
        distance = np.linalg.norm(np.array(current_position) - np.array(target))
        congestion = dynamic_congestion_grid[target[0], target[1]]
        return distance + congestion

    return min(evacuation_target_cells, key=goal_score)

def find_nearest_goal(current_position, evacuation_target_cells):
    if not evacuation_target_cells:
        raise ValueError("Evacuation target cells are empty.")
    
    return min(evacuation_target_cells, key=lambda target: np.linalg.norm(np.array(current_position) - np.array(target)))


class Pedestrian:
    def __init__(self, position, speed, behavior, type, special_area=False, id=None):
        self.position = position
        self.speed = speed
        self.behavior = behavior
        self.type = type
        self.special_area = special_area
        self.path = [position]  # 경로 초기화
        self.knows_target = False
        self.id = id
        self.goal = None
        self.goal_reached = False  # 목표 도달 여부 초기화
        self.total_distance = 0  # 누적 이동 거리 초기화
        self.real_path = [position]  # 실제 이동한 경로 초기화

    def move(self, grid, dynamic_congestion_grid, accumulated_congestion_grid, evacuation_target_set, all_pedestrians):
        time_step = 1
        distance_to_move = self.speed * time_step

        # 목표를 모르는 경우 행동에 따라 목표 설정
        if not self.knows_target:
            if self.behavior == "exploratory":
                path = self.random_walk(grid, dynamic_congestion_grid, all_pedestrians)
                if path:
                    self.path.extend(path[1:])
                    for pedestrian in all_pedestrians:
                        distance_to_pedestrian = np.linalg.norm(np.array(self.position) - np.array(pedestrian.position))
                        if distance_to_pedestrian <= 20 and random.random() < 0.8:
                            if pedestrian.knows_target:
                                self.knows_target = True
                                self.goal = pedestrian.goal
                                print(f"Pedestrian {self.id} knows target {self.goal} now.")
                                self.path = a_star(grid, dynamic_congestion_grid, self.position, self.goal, self)
                                break
            elif self.behavior == "knows_all":
                self.goal = find_optimal_goal(self.position, evacuation_target_set, dynamic_congestion_grid)
                if not self.goal:
                    self.goal = find_nearest_goal(self.position, evacuation_target_set)
                self.knows_target = True
                self.path = a_star(grid, dynamic_congestion_grid, self.position, self.goal, self)

            elif self.behavior == "knows_specific":
                self.goal = random.choice(evacuation_target_set)
                self.knows_target = True
                self.path = a_star(grid, dynamic_congestion_grid, self.position, self.goal, self)

        # 경로가 없으면 랜덤 워크 수행
        if not self.path or len(self.path) <= 1:
            self.path = self.random_walk(grid, dynamic_congestion_grid, all_pedestrians)

        # 목표를 알고 경로가 있을 경우 이동
        if self.knows_target and self.path and len(self.path) > 1:
            next_position = self.path[1]
            distance_to_next = np.linalg.norm(np.array(next_position) - np.array(self.position))

            if distance_to_move >= distance_to_next:
                dynamic_congestion_grid[self.position] -= 1
                self.total_distance += distance_to_move

                # 다음 위치로 이동
                self.position = next_position
                self.path.pop(0)

                dynamic_congestion_grid[self.position] += 1
                accumulated_congestion_grid[self.position] += 1
            else:
                direction = (np.array(next_position) - np.array(self.position)) / distance_to_next
                new_position = tuple(np.round(np.array(self.position) + direction * distance_to_move).astype(int))

                self.total_distance += distance_to_move

                dynamic_congestion_grid[self.position] -= 1
                self.position = new_position
                dynamic_congestion_grid[self.position] += 1
                accumulated_congestion_grid[self.position] += 1

            # 이동한 위치를 real_path에 기록
            self.real_path.append(self.position)

            # 목표에 도달했는지 확인
            if self.position == self.goal:
                self.goal_reached = True
                print(f"Pedestrian {self.id} reached the goal at {self.goal}.")

        # 경로가 다 소진되면 현재 위치를 경로로 설정 (exploratory 보행자에만 적용)
        if self.behavior == "exploratory" and len(self.path) <= 1:
            self.path = [self.position]

    def random_walk(self, grid, dynamic_congestion_grid, all_pedestrians):
        """대피소를 모르는 보행자의 random walk 함수."""
        next_position = random.choice(self.get_neighbors(self.position, grid))

        # 이동한 위치를 real_path에 기록
        self.real_path.append(next_position)
        return [self.position, next_position]

    def get_neighbors(self, position, grid):
        """현재 위치에서 이동 가능한 인접한 셀을 반환"""
        x, y = position
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                # 격자 범위를 벗어나지 않고, 이동 가능한 셀인지 확인
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
                    neighbors.append((nx, ny))
        return neighbors
