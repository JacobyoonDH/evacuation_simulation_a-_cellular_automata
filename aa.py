import heapq
import numpy as np

# 혼잡도 및 침수 상태 기반 속도 조정 함수

#def adjust_speed_based_on_congestion(congestion_level, base_speed):
#    """혼잡도에 따라 보행자의 속도를 조정합니다 인/M^2."""
#    if congestion_level <= 0.7:
#        return base_speed
#    elif congestion_level <= 1.1:
#        return base_speed * 0.83
##    elif congestion_level <= 2.6:
#        return base_speed * 0.53
#    elif congestion_level > 2.6:
#        return base_speed * 0.3

def adjust_speed_based_on_congestion(congestion_level, base_speed):
    """혼잡도에 따라 보행자의 속도를 조정합니다 (인/m² 기준)."""
    if congestion_level <= 1.1:
        return base_speed
    elif 1.1 < congestion_level <= 2.6:
        return base_speed * 0.83
    elif 2.6 < congestion_level <= 3.8:
        return base_speed * 0.53
    else:  # congestion_level > 3.8
        return base_speed * 0.3

def adjust_speed_based_on_flood(flood_depth, base_speed):
    """침수 깊이에 따라 보행자의 속도를 조정합니다."""
    if flood_depth <= 0:
        return base_speed
    elif flood_depth <= 0.1:
        return base_speed * 0.87
    elif flood_depth <= 0.2:
        return base_speed * 0.75
    elif flood_depth <= 0.3:
        return base_speed * 0.68
    elif flood_depth <= 0.4:
        return base_speed * 0.64
    elif flood_depth <= 0.5:
        return base_speed * 0.59
    else:
        return base_speed * 0.1  # 심각한 침수 지역


def heuristic(a, b):
    """유클리드 거리 기반 휴리스틱 함수"""
    return np.linalg.norm(np.array(a) - np.array(b))
def a_star(grid, congestion_grid, start, goal, pedestrian, flood_cells=None):
    """혼잡도 및 침수를 고려한 소요 시간 기반 A* 알고리즘."""
    start = tuple(map(int, np.round(start)))
    goal = tuple(map(int, np.round(goal)))
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return [tuple(map(int, np.round(pos))) for pos in reconstruct_path(came_from, current)]
        if current in closed_set:
            continue
        closed_set.add(current)

        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + di, current[1] + dj)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if grid[neighbor] == 1:  # 장애물
                continue
            
            # 혼잡도와 침수 심도 기반 속도 계산
            congestion_level = congestion_grid[neighbor]
              # 혼잡도가 6 이상인 셀은 통과 불가
            if congestion_level >= 6:
                continue
            flood_depth = flood_cells[neighbor] if flood_cells is not None else 0
            adjusted_speed = pedestrian.original_speed
            adjusted_speed = adjust_speed_based_on_congestion(congestion_level, adjusted_speed)
            adjusted_speed = adjust_speed_based_on_flood(flood_depth, adjusted_speed)
            
            # 이동 불가 처리
            if adjusted_speed <= 0.2:
                continue
            
            # 소요 시간 계산
            distance = heuristic(current, neighbor)  # 현재 셀에서 이웃 셀까지의 거리
            time_cost = distance / adjusted_speed
            tentative_g_score = g_score[current] + time_cost

            if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def reconstruct_path(came_from, current):
    """최단 경로를 재구성합니다."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]
