import numpy as np
from pedestrian import *
import numpy as np
import matplotlib.pyplot as plt
from pedestrian import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_congestion_with_context(grid, x_coords, y_coords, congestion_grid, area_gdf, buildings_gdf, evacuation_target_cells, pedestrians, step, title="Dynamic Congestion Grid"):
    """
    건물, 도보 가능 지역, 대피 목표와 함께 혼잡도를 시각화하는 함수.
    
    Parameters:
        grid (numpy.ndarray): 격자 데이터
        x_coords (numpy.ndarray): X 좌표 배열
        y_coords (numpy.ndarray): Y 좌표 배열
        congestion_grid (numpy.ndarray): 혼잡도 그리드
        area_gdf (GeoDataFrame): 도보 가능 지역 GeoDataFrame
        buildings_gdf (GeoDataFrame): 건물 GeoDataFrame
        evacuation_target_cells (list of tuples): 대피 목표 셀의 인덱스 리스트
        pedestrians (list): 보행자 객체 리스트
        step (int): 현재 스텝
        title (str): 그래프 제목
    """
    plt.figure(figsize=(16, 12))
    ax = plt.gca()

    # 도보 이동 가능 지역 시각화
    area_gdf.plot(ax=ax, color='lightgreen', edgecolor='white', alpha=0.7, label='Walkable Area')

    # 건물 시각화
    buildings_gdf.plot(ax=ax, color='black', alpha=0.9, label='Buildings')

    plt.imshow(congestion_grid, cmap='hot', origin='upper',
               extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]], vmin=0, vmax=5, alpha=0.6)

    # 대피 목표 셀 시각화 (주황색 'X' 마커)
    for i, j in evacuation_target_cells:
        plt.scatter(x_coords[j], y_coords[i], color='orange', s=50, marker='x', label='Evacuation Target' if 'Evacuation Target' not in ax.get_legend_handles_labels()[1] else "")

    # 보행자의 현재 위치 시각화 (작은 빨간 점)
    for pedestrian in pedestrians:
        ped_x = x_coords[pedestrian.position[1]]
        ped_y = y_coords[pedestrian.position[0]]
        plt.scatter(ped_x, ped_y, color='red', s=10, marker='.', label='Pedestrian' if 'Pedestrian' not in ax.get_legend_handles_labels()[1] else "")

    # 컬러바 설정 (혼잡도 레벨 표시)
    plt.colorbar(label='Congestion Level (≥3)', shrink=0.8)

    # 범례 설정
    legend_handles = [
        mpatches.Patch(color='lightgreen', label='Walkable Area'),
        mpatches.Patch(color='black', label='Buildings'),
        mpatches.Patch(color='orange', label='Evacuation Target'),
        mpatches.Patch(color='red', label='Pedestrian')
    ]
    plt.legend(handles=legend_handles, loc='upper right')

    # 그래프 제목 및 라벨
    plt.title(f"{title} at Step {step}")
    plt.xlabel("X Coordinate (meters in EPSG:5181)")
    plt.ylabel("Y Coordinate (meters in EPSG:5181)")

    plt.show()



import numpy as np


import numpy as np
from pedestrian import *
import matplotlib.pyplot as plt
def run_simulation(grid, pedestrians, evacuation_target_cells, x_coords, y_coords, area_gdf, buildings_gdf, steps=100):
    """
    보행자들의 이동 시뮬레이션을 실행하고 대피 동선을 기록합니다.

    Returns:
        dynamic_congestion_grid (numpy.ndarray): 최종 일시적 혼잡도 그리드
        accumulated_congestion_grid (numpy.ndarray): 최종 누적 혼잡도 그리드
        evacuation_details (list): 각 보행자의 대피 경로, behavior, type, 소요시간, 도착 여부, 목표 기록
        congestion_timestamps (list): 혼잡도가 3 이상인 셀이 발생한 타임스텝 리스트
    """
    # 혼잡도 그리드 초기화
    dynamic_congestion_grid = np.zeros_like(grid, dtype=int)
    accumulated_congestion_grid = np.zeros_like(grid, dtype=int)
    evacuation_details = []
    congestion_timestamps = []

    # 초기 보행자 위치에 혼잡도 설정
    for pedestrian in pedestrians:
        dynamic_congestion_grid[pedestrian.position] += 1
        pedestrian.total_distance = 0  # 누적 이동 거리 초기화

    # 누적 혼잡도에 초기 혼잡도를 반영
    accumulated_congestion_grid += dynamic_congestion_grid

    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")

        # 보행자 이동 및 목표 도달 여부 확인
        for idx, pedestrian in enumerate(pedestrians):
            old_position = pedestrian.position

            # 보행자 이동
            pedestrian.move(grid, dynamic_congestion_grid, accumulated_congestion_grid, evacuation_target_cells, pedestrians)
            new_position = pedestrian.position

            # 누적 이동 거리 업데이트
            pedestrian.total_distance += np.linalg.norm(np.array(new_position) - np.array(old_position))
            pedestrian.real_path.append(old_position)
            # 혼잡도 그리드 갱신
            if old_position != new_position:
                dynamic_congestion_grid[old_position] -= 1
                dynamic_congestion_grid[new_position] += 1

        # 혼잡도가 3 이상인 셀이 있는지 확인하고 시각화
#        if np.any(dynamic_congestion_grid >= 3):
#            congestion_timestamps.append(step + 1)
#            plot_congestion_with_context(grid, x_coords, y_coords, dynamic_congestion_grid, area_gdf, buildings_gdf, evacuation_target_cells, pedestrians, step, title="Dynamic Congestion Grid")

        # 누적 혼잡도 갱신
        accumulated_congestion_grid += dynamic_congestion_grid

        # 모든 보행자 대피 완료 시 조기 종료
        if all(p.goal_reached for p in pedestrians):
            print(f"All pedestrians evacuated by step {step + 1}.")
            break

    # 최종 누적 혼잡도 시각화
    plot_congestion_with_context(grid, x_coords, y_coords, accumulated_congestion_grid, area_gdf, buildings_gdf, evacuation_target_cells, pedestrians, step + 1, title="Final Accumulated Congestion Grid")

    return dynamic_congestion_grid, accumulated_congestion_grid, evacuation_details, congestion_timestamps
