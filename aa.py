import heapq
import numpy as np

def heuristic(a, b):
    """유클리드 거리 기반 휴리스틱 함수"""
    return np.linalg.norm(np.array(a) - np.array(b))

def adjust_speed_based_on_congestion(congestion_level, base_speed):
    """혼잡도에 따라 보행자의 속도를 조정합니다."""
    if congestion_level <= 1:
        return base_speed
    elif congestion_level <= 3:
        return base_speed * 0.75
    elif congestion_level <= 5:
        return base_speed * 0.5
    else:
        return base_speed * 0.3
def a_star(grid, congestion_grid, start, goal, pedestrian, congestion_adjustment=True):
    """혼잡도를 고려한 A* 알고리즘."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()  # 닫힌 목록 (검색 속도 최적화)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        # 이미 처리된 노드는 무시
        if current in closed_set:
            continue
        closed_set.add(current)

        # 인접 노드 탐색
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + di, current[1] + dj)

            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if grid[neighbor] == 1:  # 장애물 체크
                continue

            congestion_level = congestion_grid[neighbor]
            adjusted_speed = pedestrian.speed

            if congestion_adjustment:
                adjusted_speed = adjust_speed_based_on_congestion(congestion_level, adjusted_speed)

            tentative_g_score = g_score[current] + (1 / adjusted_speed)

            if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            # 더 나은 경로 발견 시 갱신
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # 경로를 찾지 못한 경우


def reconstruct_path(came_from, current):
    """최단 경로를 재구성합니다."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]
