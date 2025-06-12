import matplotlib.pyplot as plt
def visualize_accumulated_congestion(
    area_gdf, buildings_gdf, evacuation_target_area_gdf, accumulated_congestion_grid, x_coords, y_coords, save_path=None
):
    """
    누적 혼잡도와 도보 이동 가능 지역, 건물, 대피소를 시각화합니다.

    Parameters:
        area_gdf (GeoDataFrame): 도보 이동 가능 지역 GeoDataFrame.
        buildings_gdf (GeoDataFrame): 건물 GeoDataFrame.
        evacuation_target_area_gdf (GeoDataFrame): 대피소 GeoDataFrame.
        accumulated_congestion_grid (numpy.ndarray): 누적 혼잡도 그리드.
        x_coords (numpy.ndarray): X축 좌표 배열.
        y_coords (numpy.ndarray): Y축 좌표 배열.
        save_path (str, optional): 그림을 저장할 파일 경로. 지정되지 않으면 저장하지 않음.
    """
    # Grid Extent 설정
    grid_extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    # 새로운 Figure 및 Axes 생성
    fig, ax = plt.subplots(figsize=(15, 12))

    # 도보 이동 가능 지역 시각화
    area_gdf.plot(ax=ax, color='lightgreen', edgecolor='black', alpha=0.5, label='Walkable Area')
    # 건물 시각화
    buildings_gdf.plot(ax=ax, color='black', alpha=0.7, label='Buildings')
    # 대피소 시각화
    evacuation_target_area_gdf.plot(ax=ax, color='blue', edgecolor='black', alpha=0.7, label='Evacuation Targets')

    # 누적 혼잡도 격자 시각화
    im = ax.imshow(
        accumulated_congestion_grid,
        cmap='hot',  # 파란 계열
        origin='lower',  # 아래에서 위로 Y축 증가
        extent=grid_extent,  # 격자 좌표 설정
        alpha=0.6  # 투명도 설정
    )

    # 색상 바 추가
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label='Congestion')
    cbar.ax.set_ylabel('Congestion', rotation=270, labelpad=15)

    # 제목 및 축 레이블
    ax.set_title("Congestion and Shelter")
    ax.set_xlabel("X (meter)")
    ax.set_ylabel("Y (meter)")

    # 범례 및 격자 추가
    ax.legend()
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # 저장 또는 출력
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 저장
        print(f"Figure saved to {save_path}")
    else:
        plt.show()  # 화면 출력

    # 리소스 정리
    plt.close()


def visualize_congestion(area_gdf, buildings_gdf, dynamic_congestion_grid, accumulated_congestion_grid, x_coords, y_coords):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 도보 이동 가능 지역과 건물 시각화
    for ax in axes:
        area_gdf.plot(ax=ax, color='lightgreen', edgecolor='black', alpha=0.5, label='도보 이동 가능 지역')
        buildings_gdf.plot(ax=ax, color='black', alpha=0.7, label='건물')

    # 일시적 혼잡도 시각화
    axes[0].imshow(dynamic_congestion_grid, cmap='hot', origin='lower',
                   extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]], alpha=0.6)
    axes[0].set_title("Dynamic Congestion (During Simulation)")
    axes[0].set_xlabel("X Coordinate (meters)")
    axes[0].set_ylabel("Y Coordinate (meters)")

    # 누적 혼잡도 시각화
    axes[1].imshow(accumulated_congestion_grid, cmap='hot', origin='lower',
                   extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]], alpha=0.6)
    axes[1].set_title("Accumulated Congestion (Final Result)")
    axes[1].set_xlabel("X Coordinate (meters)")
    axes[1].set_ylabel("Y Coordinate (meters)")

    plt.colorbar(axes[0].images[0], ax=axes, orientation='vertical', label='Congestion Level')
    plt.legend()
    plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def position_to_index(position, x_coords, y_coords):
    """좌표 (x, y)를 격자 인덱스 (i, j)로 변환합니다."""
    x, y = position
    j = np.argmin(np.abs(x_coords - x))
    i = np.argmin(np.abs(y_coords - y))
    
    # 안전 장치: 인덱스가 격자 범위를 벗어나지 않도록 보정
    i = np.clip(i, 0, len(y_coords) - 1)
    j = np.clip(j, 0, len(x_coords) - 1)
    
    return i, j


def visualize_pedestrian_positions(grid, x_coords, y_coords, pedestrians, special_area_gdf, area_gdf, buildings_gdf, evacuation_target_cells):
    """
    격자 위에 보행자의 초기 위치와 대피 영역을 시각화합니다.

    Parameters:
        grid (numpy.ndarray): 격자 (0 = 이동 가능, 1 = 장애물)
        x_coords (numpy.ndarray): X 좌표 배열
        y_coords (numpy.ndarray): Y 좌표 배열
        pedestrians (list): 보행자 객체 리스트
        special_area_gdf (GeoDataFrame): 특별 구역 GeoDataFrame
        area_gdf (GeoDataFrame): 이동 가능한 전체 영역 GeoDataFrame
        buildings_gdf (GeoDataFrame): 건물 GeoDataFrame
        evacuation_target_cells (list): 대피 목표 셀의 인덱스 리스트
    """

    fig, ax = plt.subplots(figsize=(16, 10))

    # 도보 이동 가능 지역 시각화
    area_gdf.plot(ax=ax, color='gray', edgecolor='black', alpha=0.5, label='Walkable Area')

    # 건물 시각화
    buildings_gdf.plot(ax=ax, color='black', alpha=0.7, label='Buildings')

    # 특별 구역 시각화
    special_area_gdf.plot(ax=ax, color='purple', alpha=0.5, label='Special Area')

    # 격자 시각화 (origin='lower' 설정)
    ax.imshow(grid, cmap='gray_r', origin='lower', extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])

    # 대피 목표 셀 시각화 (주황색 X 마커)
    for i, j in evacuation_target_cells:
        ax.scatter(x_coords[j], y_coords[i], color='orange', s=50, marker='x', label='Evacuation Target' if 'Evacuation Target' not in ax.get_legend_handles_labels()[1] else "")

    # 보행자 초기 위치 시각화
    special_area_positions = [(ped.position[1], ped.position[0]) for ped in pedestrians if ped.special_area]
    general_area_positions = [(ped.position[1], ped.position[0]) for ped in pedestrians if not ped.special_area]

    # 특별 구역 보행자 시각화 (빨간색)
    ax.scatter(
        [x_coords[j] for j, i in special_area_positions],
        [y_coords[i] for j, i in special_area_positions],
        color='red',
        s=50,
        alpha=0.7,
        label='Special Area Agents'
    )

    # 일반 구역 보행자 시각화 (초록색)
    ax.scatter(
        [x_coords[j] for j, i in general_area_positions],
        [y_coords[i] for j, i in general_area_positions],
        color='yellow',
        s=50,
        alpha=0.7,
        label='General Area Agents'
    )

    # 범례 설정
    legend_handles = [
        mpatches.Patch(color='gray', label='Walkable Area'),
        mpatches.Patch(color='black', label='Buildings'),
        mpatches.Patch(color='purple', label='Special Area'),
        mpatches.Patch(color='red', label='Special Area Agents'),
        mpatches.Patch(color='yellow', label='General Area Agents'),
        mpatches.Patch(color='orange', label='Evacuation Target'),
    ]

    ax.legend(handles=legend_handles)

    # 타이틀 및 라벨 설정
    plt.title("Pedestrian Initial Positions on Grid with Evacuation Targets")
    plt.xlabel("X Coordinate (meters in EPSG:5181)")
    plt.ylabel("Y Coordinate (meters in EPSG:5181)")

    plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def visualize_by_types_evacuations(grid, x_coords, y_coords, pedestrians, special_area_gdf, area_gdf, buildings_gdf, evacuation_target_cells, save_path):
    """
    보행자들의 대피 경로를 시각화하고 저장합니다.

    Parameters:
        grid (numpy.ndarray): 격자 (0 = 이동 가능, 1 = 장애물)
        x_coords (numpy.ndarray): X 좌표 배열
        y_coords (numpy.ndarray): Y 좌표 배열
        pedestrians (list): 보행자 객체 리스트
        special_area_gdf (GeoDataFrame): 특별 구역 GeoDataFrame
        area_gdf (GeoDataFrame): 이동 가능한 전체 영역 GeoDataFrame
        buildings_gdf (GeoDataFrame): 건물 GeoDataFrame
        evacuation_target_cells (list): 대피 목표 셀의 인덱스 리스트
        save_path (str): 결과 이미지를 저장할 경로
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    # 도보 이동 가능 지역 시각화
    area_gdf.plot(ax=ax, color='gray', edgecolor='black', alpha=0.5, label='Walkable Area')

    # 건물 시각화
    buildings_gdf.plot(ax=ax, color='black', alpha=0.7, label='Buildings')

    # 특별 구역 시각화
    special_area_gdf.plot(ax=ax, color='purple', alpha=0.5, label='Special Area')

    # 격자 시각화
    ax.imshow(grid, cmap='gray_r', origin='lower', extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])

    # 대피 목표 셀 시각화
    for i, j in evacuation_target_cells:
        ax.scatter(x_coords[j], y_coords[i], color='orange', s=50, marker='x', label='Evacuation Target' if 'Evacuation Target' not in ax.get_legend_handles_labels()[1] else "")

    # 보행자 타입별 색상 설정
    type_colors = {
        "노인/어린이": "red",
        "중장년": "blue",
        "청소년/청년": "green",
        "장애인": "magenta"
    }

    # 보행자의 혼잡도 기반 이동 경로 시각화
    for pedestrian in pedestrians:
        path = pedestrian.path_congest  # 정수 좌표 경로 사용
        type_color = type_colors.get(pedestrian.type, "black")

        if len(path) > 1:
            path_x = [x_coords[pos[1]] for pos in path]
            path_y = [y_coords[pos[0]] for pos in path]
            ax.plot(path_x, path_y, color=type_color, linewidth=1, label=f'{pedestrian.type}' if f'{pedestrian.type}' not in ax.get_legend_handles_labels()[1] else "")

            ax.scatter(path_x[0], path_y[0], color=type_color, s=30, marker='o')  # 시작점
            ax.scatter(path_x[-1], path_y[-1], color=type_color, s=30, marker='s')  # 도착점

    # 범례 설정
    ax.legend(loc='upper right')
    plt.title("보행자 대피 동선 결과")
    plt.xlabel("X Coordinate (meters in EPSG:5181)")
    plt.ylabel("Y Coordinate (meters in EPSG:5181)")

    # 결과 저장
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
def visualize_evacuations_by_behavior(grid, x_coords, y_coords, pedestrians, special_area_gdf, area_gdf, buildings_gdf, evacuation_target_cells, save_path):
    """
    보행자들의 대피 경로를 행동 패턴별로 시각화하고 저장합니다.

    Parameters:
        grid (numpy.ndarray): 격자 (0 = 이동 가능, 1 = 장애물)
        x_coords (numpy.ndarray): X 좌표 배열
        y_coords (numpy.ndarray): Y 좌표 배열
        pedestrians (list): 보행자 객체 리스트
        special_area_gdf (GeoDataFrame): 특별 구역 GeoDataFrame
        area_gdf (GeoDataFrame): 이동 가능한 전체 영역 GeoDataFrame
        buildings_gdf (GeoDataFrame): 건물 GeoDataFrame
        evacuation_target_cells (list): 대피 목표 셀의 인덱스 리스트
        save_path (str): 결과 이미지를 저장할 경로
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    # 도보 이동 가능 지역 시각화
    area_gdf.plot(ax=ax, color='gray', edgecolor='black', alpha=0.5, label='Walkable Area')

    # 건물 시각화
    buildings_gdf.plot(ax=ax, color='black', alpha=0.7, label='Buildings')

    # 특별 구역 시각화
    special_area_gdf.plot(ax=ax, color='purple', alpha=0.5, label='Special Area')

    # 격자 시각화
    ax.imshow(grid, cmap='gray_r', origin='lower', extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])

    # 대피 목표 셀 시각화
    for i, j in evacuation_target_cells:
        ax.scatter(x_coords[j], y_coords[i], color='orange', s=50, marker='x', label='Evacuation Target' if 'Evacuation Target' not in ax.get_legend_handles_labels()[1] else "")

    # 보행자 행동 패턴별 색상 설정
    behavior_colors = {
        "knows_specific": "red",
        "knows_all": "blue",
        "exploratory": "green",
    }

    # 보행자의 혼잡도 기반 이동 경로 시각화
    for pedestrian in pedestrians:
        path = pedestrian.path_congest  # 정수 좌표 경로 사용
        behavior_color = behavior_colors.get(pedestrian.behavior, "black")

        if len(path) > 1:
            path_x = [x_coords[pos[1]] for pos in path]
            path_y = [y_coords[pos[0]] for pos in path]
            ax.plot(path_x, path_y, color=behavior_color, linewidth=1, label=f'{pedestrian.behavior}' if f'{pedestrian.behavior}' not in ax.get_legend_handles_labels()[1] else "")

            ax.scatter(path_x[0], path_y[0], color=behavior_color, s=30, marker='o')  # 시작점
            ax.scatter(path_x[-1], path_y[-1], color=behavior_color, s=30, marker='s')  # 도착점

    ax.legend(loc='upper right')
    plt.title("대피소 인지 여부에 따른 대피 결과")
    plt.xlabel("X Coordinate (meters in EPSG:5181)")
    plt.ylabel("Y Coordinate (meters in EPSG:5181)")

    # 결과 저장
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
