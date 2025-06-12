import random
import numpy as np
from aa import *
from shapely.geometry import Point
from shapely.strtree import STRtree
def grid_to_epsg5181(grid_position, x_coords, y_coords):
    """
    격자 좌표를 EPSG:5181 좌표로 변환합니다.

    Parameters:
        grid_position (tuple): 격자 좌표 (row, col)
        x_coords (numpy.ndarray): X 축 좌표 배열
        y_coords (numpy.ndarray): Y 축 좌표 배열

    Returns:
        shapely.geometry.Point: EPSG:5181 좌표계의 포인트 객체
    """
    row, col = grid_position
    x = x_coords[col]
    y = y_coords[row]
    return Point(x, y)

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
    def __init__(self, position, speed, behavior, type, special_area=False, id=None, shelter_knowledge_probability=0.8):
        self.position = position
        self.speed = speed
        self.behavior = behavior
        self.type = type
        self.special_area = special_area
        self.shelter_knowledge_probability = shelter_knowledge_probability  # Include this
        self.position_float = np.array(position, dtype=float)  # 실수 기반 현재 위치
        self.original_speed = speed  # 기본 속도 저장
        self.path = [position]
        self.knows_target = False
        self.id = id
        self.goal = None
        self.goal_reached = False
        self.total_distance = 0
        initial_size = 1000  # 초기 크기
        self.real_path = [position]  # 실제 이동 경로
        self.path_congest = [position]  # 경로의 혼잡도 상태
        self.past_congestion = []  # 지나온 셀의 혼잡도 기록
        self.past_flood_levels = []  # 지나온 셀의 침수 심도 기록
        self.time_steps = []  # 타임스텝 기록    
        # 인덱스 관리
        self.current_index = 1  # 기록 시작 인덱스
        self.time_step_index = 0  # 타임스텝 별도 관리
        self.tried_to_receive_from = set()

        """
        보행자 이동을 처리합니다.
    
        Parameters:
            current_step (int): 현재 시뮬레이션 단계
            shelter_geometries (GeoDataFrame): 대피소의 영역 GeoDataFrame
            x_coords (numpy.ndarray): 격자의 X 좌표
            y_coords (numpy.ndarray): 격자의 Y 좌표
        """
        # 목표를 모르는 경우 경로 계산
    def move(self, grid, dynamic_congestion_grid, accumulated_congestion_grid, evacuation_target_set,
             all_pedestrians, flood_cells, shelter_geometries, x_coords=None, y_coords=None, current_step=None, shelter_knowledge_probability=None):

        if not self.knows_target:
            if self.behavior == "exploratory":
                path = self.random_walk(grid, dynamic_congestion_grid, all_pedestrians)
                if path:
                    self.path.extend(path[1:])
                    
                    for pedestrian in all_pedestrians:
                        if pedestrian.id == self.id:
                            continue
                        # 거리가 8m 이내인 경우만 대상
                        distance = np.linalg.norm(self.position_float - np.array(pedestrian.position_float))
                        if distance <= 8:
                            # 이전에 시도했던 대상이면 건너뜀
                            if pedestrian.id in self.tried_to_receive_from:
                                continue

                            # 전파 시도 기록
                            self.tried_to_receive_from.add(pedestrian.id)

                            # 전파 확률 조건
                            if random.random() < self.shelter_knowledge_probability:
                                if pedestrian.knows_target and pedestrian.goal:
                                    self.knows_target = True
                                    self.goal = pedestrian.goal
                                    self.path = [tuple(map(int, np.round(pos))) for pos in a_star(
                                        grid, dynamic_congestion_grid, self.position_float, self.goal, self, flood_cells
                                    )]
                                    break
            elif self.behavior == "knows_all":
                if evacuation_target_set:
                    self.goal = find_optimal_goal(self.position, evacuation_target_set, dynamic_congestion_grid)
                    if self.goal:
                        self.knows_target = True
                        self.path = [tuple(map(int, np.round(pos))) for pos in a_star(grid, dynamic_congestion_grid, self.position_float, self.goal, self, flood_cells)]
            elif self.behavior == "knows_specific":
                if evacuation_target_set:
                    self.goal = random.choice(evacuation_target_set)
                    self.knows_target = True
                    self.path = [tuple(map(int, np.round(pos))) for pos in a_star(grid, dynamic_congestion_grid, self.position_float, self.goal, self, flood_cells)]
                # 혼잡도 및 침수 상태 확인
        rounded_position = tuple(map(int, np.round(self.position_float)))
        neighbors = [
            (rounded_position[0] + dx, rounded_position[1] + dy)
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if 0 <= rounded_position[0] + dx < flood_cells.shape[0] and 0 <= rounded_position[1] + dy < flood_cells.shape[1]]
        max_flood_depth = max((flood_cells[n] for n in neighbors), default=0)
     #   max_congestion_level = (sum(dynamic_congestion_grid[n] for n in neighbors)/len(neighbors)
     #                               if neighbors else 0)
        max_congestion_level = dynamic_congestion_grid[rounded_position[0],rounded_position[1]]
        # 동적 속도 계산
        adjusted_speed = self.original_speed
        if max_flood_depth > 0:
            adjusted_speed = adjust_speed_based_on_flood(max_flood_depth, adjusted_speed)
        adjusted_speed = adjust_speed_based_on_congestion(max_congestion_level, adjusted_speed)
        self.speed = adjusted_speed
        remaining_distance= adjusted_speed
        if self.goal is not None:
            goal_distance = np.linalg.norm(np.array(self.goal) - np.array(self.position_float))
            if goal_distance <= 10:  # 10m 이내일 경우 속도 복귀
                self.speed = self.original_speed
                adjusted_speed = self.speed
                remaining_distance= adjusted_speed
        # 혼잡도 및 침수 기록
        self.past_congestion.append(max_congestion_level)
        self.past_flood_levels.append(max_flood_depth)
        while remaining_distance > 0 and len(self.path) > 1:
            next_position = np.array(self.path[1], dtype=float)
            distance_to_next = np.linalg.norm(next_position - self.position_float)
    
            if max_congestion_level >= 6.7 and self.goal is not None and current_step % 30 == 0:
                self.path = [tuple(map(int, np.round(pos))) for pos in a_star(grid, dynamic_congestion_grid, self.position_float, self.goal, self, flood_cells)]
            # 이동 처리
            if remaining_distance >= distance_to_next:
                self.position_float = next_position
                self.position = tuple(map(int, np.round(self.position_float)))
                self.total_distance += distance_to_next
                self.path.pop(0)
                remaining_distance -= distance_to_next
            else:
                direction = (next_position - self.position_float) / distance_to_next
                self.position_float += direction * remaining_distance
                self.position = tuple(map(int, np.round(self.position_float)))
                self.total_distance += remaining_distance
                remaining_distance = 0
    
            # 실제 이동 기록
            self.real_path.append(tuple(self.position_float))
            self.path_congest.append(self.position)
            # 대피소 포함 여부 확인
            if shelter_geometries is not None and x_coords is not None and y_coords is not None:
                position_epsg5181 = grid_to_epsg5181(self.position, x_coords, y_coords)
                for shelter in shelter_geometries.geometry:
                    if shelter.contains(position_epsg5181):
                        self.goal_reached = True
                        self.time_steps.append(current_step)
                        return
            elif self.position == self.goal:
                self.goal_reached = True
                self.time_steps.append(current_step)
            self.speed= self.original_speed


    def random_walk(self, grid, dynamic_congestion_grid, all_pedestrians, min_distance=20):
        path = [tuple(self.position)]
        current_position = np.array(self.position, dtype=float)
        total_distance = 0

        while total_distance < min_distance:
            neighbors = self.get_neighbors(current_position, grid)
            if not neighbors:
                break

            next_position = random.choice(neighbors)
            distance_to_next = np.linalg.norm(np.array(next_position) - current_position)
            total_distance += distance_to_next

            path.append(next_position)
            current_position = np.array(next_position, dtype=float)

        self.real_path.extend(path[1:])
        return path

    def get_neighbors(self, position, grid):
        x, y = map(int, position)
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
                    neighbors.append((nx, ny))
        return neighbors