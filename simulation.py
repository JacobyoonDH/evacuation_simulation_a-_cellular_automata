from pedestrian import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import time
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_congestion_with_context222_past_ver(
    grid, x_coords, y_coords, congestion_grid, flood_cells,
    area_gdf, buildings_gdf, evacuee_area, pedestrians,
    step, save_base_path, title="Congestion and Flood Visualization"
):
    """
    혼잡도와 보행자 현재 위치를 시각화.

    Parameters:
        grid (numpy.ndarray): 격자 데이터
        x_coords (numpy.ndarray): X 좌표
        y_coords (numpy.ndarray): Y 좌표
        congestion_grid (numpy.ndarray): 혼잡도 데이터
        flood_cells (numpy.ndarray): 침수 데이터
        area_gdf (GeoDataFrame): 이동 가능한 영역
        buildings_gdf (GeoDataFrame): 건물 데이터
        evacuee_area (GeoDataFrame): 대피소 영역
        pedestrians (list): 보행자 객체
        step (int): 현재 스텝
        save_base_path (str): 결과 저장 경로
        title (str): 시각화 제목
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.title(f"{title} (Step {step})")

    # 혼잡도 히트맵 추가
    congestion_map = ax.imshow(
        congestion_grid, cmap="Reds", origin="lower",
        extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
        alpha=0.7
    )
    cbar_congestion = plt.colorbar(congestion_map, ax=ax, fraction=0.03, pad=0.04)
    cbar_congestion.set_label("Congestion Level")

    # 이동 가능한 영역 및 건물 시각화
    area_gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5, label="Walkable Area")
    buildings_gdf.plot(ax=ax, color='black', alpha=0.7, label="Buildings")
    evacuee_area.plot(ax=ax, color='orange', alpha=0.5, label="Evacuation Areas")

    # 보행자 현재 위치 시각화 (goal_reached 제외)
    current_positions = [
        (ped.position[1], ped.position[0]) for ped in pedestrians if not ped.goal_reached
    ]
    ax.scatter(
        [x_coords[j] for j, i in current_positions],
        [y_coords[i] for j, i in current_positions],
        color='blue', s=30, label="Pedestrian Current Positions"
    )

    # 축 및 범례 설정
    plt.xlabel("X Coordinate (meters in EPSG:5181)")
    plt.ylabel("Y Coordinate (meters in EPSG:5181)")
    ax.legend(loc="upper right")

    # 결과 저장
    save_path = f"{save_base_path}/congestion_with_pedestrians_step_{step}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_congestion_with_context(
    grid, x_coords, y_coords, congestion_grid, flood_cells,
    area_gdf, buildings_gdf, evacuee_area, pedestrians,
    step, save_base_path, title="Congestion and Flood Visualization"
):
    """
    혼잡도와 보행자 현재 위치를 시각화.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.title(f"{title} (Step {step})")

    # 혼잡도 히트맵 추가 (컬러맵 및 범위 고정)
    congestion_map = ax.imshow(
        congestion_grid, cmap="coolwarm", origin="lower",
        extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
        alpha=0.8, vmin=0, vmax=10
    )
    cbar_congestion = plt.colorbar(congestion_map, ax=ax, fraction=0.03, pad=0.04)
    cbar_congestion.set_label("Congestion Level")

    # 혼잡도 값 텍스트 표시
    for i in range(congestion_grid.shape[0]):
        for j in range(congestion_grid.shape[1]):
            value = congestion_grid[i, j]
            if value > 0:  # 혼잡도가 0보다 큰 경우만 표시
                ax.text(
                    x_coords[j], y_coords[i], str(value),
                    color="black", fontsize=8, ha="center", va="center"
                )

    # 이동 가능한 영역 및 건물 시각화
    area_gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5, label="Walkable Area")
    buildings_gdf.plot(ax=ax, color='black', alpha=0.7, label="Buildings")
    evacuee_area.plot(ax=ax, color='orange', alpha=0.5, label="Evacuation Areas")

    # 보행자 현재 위치 시각화 (goal_reached 제외)
    current_positions = [
        (ped.position[1], ped.position[0]) for ped in pedestrians if not ped.goal_reached
    ]
    ax.scatter(
        [x_coords[j] for j, i in current_positions],
        [y_coords[i] for j, i in current_positions],
        color='blue', s=30, label="Pedestrian Current Positions"
    )

    # 축 및 범례 설정
    plt.xlabel("X Coordinate (meters in EPSG:5181)")
    plt.ylabel("Y Coordinate (meters in EPSG:5181)")
    ax.legend(loc="upper right")

    # 결과 저장
    save_path = f"{save_base_path}/congestion_with_pedestrians_step_{step}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np
from pedestrian import *
import os
def run_simulation(
    grid, pedestrians, evacuation_target_cells, x_coords, y_coords, 
    area_gdf, buildings_gdf, steps, evacuee_area, save_path, flood_cells
):
    """
    보행자들의 이동 시뮬레이션을 실행하고 대피 동선을 기록합니다.

    Returns:
        dynamic_congestion_grid (numpy.ndarray): 최종 일시적 혼잡도 그리드
        accumulated_congestion_grid (numpy.ndarray): 최종 누적 혼잡도 그리드
        evacuation_details (list): 각 보행자의 대피 경로, behavior, type, 소요시간, 도착 여부, 목표 기록
        max_congestion_per_step (list): 각 타임스텝별 최대 혼잡도 값
    """
    # 혼잡도 그리드 초기화
    dynamic_congestion_grid = np.zeros_like(grid, dtype=int)
    accumulated_congestion_grid = np.zeros_like(grid, dtype=int)
    evacuation_details = []
    max_congestion_per_step = []  # 각 타임스텝별 최대 혼잡도를 저장할 리스트
    saved_dynamic_congestion_grids = []

    # 초기 침수 값 계산
    initial_flood_cells = flood_cells - (flood_cells *2600/ 3600)  # 초기 16.67분 전 침수심 계산
    final_flood_cells = flood_cells

    # 초기 보행자 위치에 혼잡도 설정 및 path_congest 초기화
    for pedestrian in pedestrians:
        rounded_position = tuple(np.round(pedestrian.position).astype(int))
        dynamic_congestion_grid[rounded_position] += 1
        pedestrian.total_distance = 0  # 누적 이동 거리 초기화
        pedestrian.path_congest = [rounded_position]  # path_congest 초기화

    for step in range(steps):
        #if (step + 1) % 50 == 0:
            #print(f"\nStep {step + 1}/{steps}")
    
        # 침수심 업데이트
        flood_cells = initial_flood_cells + (final_flood_cells - initial_flood_cells) * (step / steps)

        # 보행자 이동
        old_positions = []  # 보행자들의 이전 위치 기록
        new_positions = []  # 보행자들의 새로운 위치 기록
        for pedestrian in pedestrians:
            rounded_position = tuple(np.round(pedestrian.position).astype(int))  # 위치 계산
            if pedestrian.goal_reached:
                # 목표 도달한 보행자의 위치에서 혼잡도 감소
                if dynamic_congestion_grid[rounded_position] > 0:
                    dynamic_congestion_grid[rounded_position] -= 1
                continue  # 목표 도달한 보행자는 이동하지 않음
        
            old_positions.append(pedestrian.position)  # 이전 위치 기록

            # 보행자 이동
            pedestrian.move(
                grid=grid,
                dynamic_congestion_grid=dynamic_congestion_grid,
                accumulated_congestion_grid=accumulated_congestion_grid,
                evacuation_target_set=evacuation_target_cells,
                all_pedestrians=pedestrians,
                shelter_geometries=evacuee_area,
                flood_cells=flood_cells,
                current_step=step,
                x_coords=x_coords,
                y_coords=y_coords
            )

            new_positions.append(pedestrian.position)  # 새로운 위치 기록
            if pedestrian.goal_reached != True and pedestrian.goal == pedestrian.position:
                pedestrian.goal_reached = True
                pedestrian.time_steps.append(step)
        # 혼잡도 격자 업데이트
        for old_position, new_position in zip(old_positions, new_positions):
            if old_position != new_position:
                if dynamic_congestion_grid[old_position] > 0:
                    dynamic_congestion_grid[old_position] -= 1  # 이전 위치에서 혼잡도 감소
                dynamic_congestion_grid[new_position] += 1  # 새로운 위치에서 혼잡도 증가
                accumulated_congestion_grid[new_position] += 1  # 누적 혼잡도 갱신

        # 각 타임스텝의 최대 혼잡도 값 저장
        max_congestion_per_step.append(dynamic_congestion_grid.max())
#        if dynamic_congestion_grid.max() > 8 and (step + 1) % 10 == 0:
#            saved_dynamic_congestion_grids.append(step)
#            plot_congestion_with_context(
#                grid, x_coords, y_coords, dynamic_congestion_grid, flood_cells,
#                area_gdf, buildings_gdf, evacuee_area, pedestrians,
#                step, save_path, title="Congestion and Flood Visualization"
#            )

        # 모든 보행자 대피 완료 시 조기 종료
        if all(p.goal_reached for p in pedestrians):
            print(f"All pedestrians evacuated by step {step + 1}.")
            break

    # 결과 반환
    return dynamic_congestion_grid, accumulated_congestion_grid, evacuation_details, max_congestion_per_step
