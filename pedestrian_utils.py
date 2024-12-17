import random
from pedestrian import Pedestrian
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

# 데이터 경로 설정
area_shp_path = r"C:\Users\doohu\Desktop\대학원\디트\1106_디트_최종대상지_4326\신규대상지_도보이동가능지역.shp"
buildings_shp_path = r"C:\Users\doohu\Desktop\대학원\디트\1106_디트_최종대상지_4326\신규대상지_건물지역_대피소제외.shp"
special_area_shp_path = r"C:\Users\doohu\Desktop\대학원\디트\디트파이썬코드\시장영역.shp"
evacuation_target_area_shp_path = r"C:\Users\doohu\Desktop\대학원\디트\1106_디트_최종대상지_4326\대피대상지_shp_point아님.shp"
creation_area_path = r"C:\Users\doohu\Desktop\대학원\디트\1106_디트_최종대상지_4326\agent_초기위치형성용.shp"
# GeoDataFrame 불러오기 및 좌표계 변환
target_crs = "EPSG:5181"
area_gdf = gpd.read_file(area_shp_path).to_crs(target_crs)
buildings_gdf = gpd.read_file(buildings_shp_path).to_crs(target_crs)
special_area_gdf = gpd.read_file(special_area_shp_path).to_crs(target_crs)
evacuation_target_area_gdf = gpd.read_file(evacuation_target_area_shp_path).to_crs(target_crs)
creation_area_gdf = gpd.read_file(creation_area_path).to_crs(target_crs)


def find_nearest_grid_index(position, x_coords, y_coords):
    """절대 좌표를 격자 인덱스로 변환합니다."""
    x, y = position
    j = np.argmin(np.abs(x_coords - x))
    i = np.argmin(np.abs(y_coords - y))
    return i, j


def create_grid(area_gdf, buildings_gdf, resolution=3):
    """
    격자를 생성하고 이동 불가능한 영역을 표시합니다.

    Parameters:
        area_gdf (GeoDataFrame): 이동 가능한 전체 영역
        buildings_gdf (GeoDataFrame): 건물 영역
        resolution (int): 격자 셀의 크기 (기본값: 3미터)

    Returns:
        tuple: 격자 (numpy.ndarray), X 좌표 (numpy.ndarray), Y 좌표 (numpy.ndarray)
    """
    # 전체 대상지의 경계(bounds)에서 격자의 X, Y 좌표 생성
    minx, miny, maxx, maxy = area_gdf.total_bounds
    x_coords = np.arange(minx, maxx, resolution)
    y_coords = np.arange(miny, maxy, resolution)

    # 격자 초기화 (0 = 이동 가능, 1 = 장애물)
    grid = np.zeros((len(y_coords), len(x_coords)))

    # 각 셀에 대해 건물 또는 영역 밖 여부 확인
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            cell = box(x, y, x + resolution, y + resolution)
            if buildings_gdf.intersects(cell).any() or not area_gdf.contains(cell.centroid).any():
                grid[i, j] = 1

    # 연결되지 않은 이동 가능한 셀 제거
    cleaned_grid = remove_disconnected_cells(grid)

    return cleaned_grid, x_coords, y_coords

def get_available_cells(grid, x_coords, y_coords, area_gdf):
    available_cells = []

    for i, y in enumerate(y_coords[:-1]):
        for j, x in enumerate(x_coords[:-1]):
            cell = box(x, y, x_coords[j + 1], y_coords[i + 1])
            if area_gdf.intersects(cell).any() and grid[i, j] == 0:
                available_cells.append((i, j))
    return available_cells

import random
from shapely.geometry import box
from pedestrian import Pedestrian
import random
from shapely.geometry import box
from pedestrian import Pedestrian

def generate_pedestrians(
    total_agents, agent_type_ratios, agent_types, behaviors, 
    special_area_cells, general_area_cells, evacuation_target_area_cells, 
    x_coords, y_coords, special_area_ratio
):
    """
    보행자를 생성하는 함수.

    Parameters:
        total_agents (int): 전체 보행자 수
        agent_type_ratios (dict): 보행자 유형별 비율
        agent_types (dict): 보행자 유형별 속도
        behaviors (dict): 행동 패턴과 가중치
        special_area_cells (list): 특별 구역 셀의 리스트 (격자 좌표계)
        general_area_cells (list): 일반 구역 셀의 리스트 (격자 좌표계)
        evacuation_target_area_cells (list): 대피 목표 셀의 리스트 (격자 좌표계)
        x_coords (numpy.ndarray): X 좌표 배열 (격자 -> 5181 좌표 변환)
        y_coords (numpy.ndarray): Y 좌표 배열 (격자 -> 5181 좌표 변환)
        special_area_ratio (float): 특별 구역 보행자 비율

    Returns:
        list: 보행자 객체 리스트
    """
    # Step 1: 대피 목표 구역 셀을 제외한 특별 구역과 일반 구역 셀
    special_area_cells = [cell for cell in special_area_cells if cell not in evacuation_target_area_cells]
    general_area_cells = [cell for cell in general_area_cells if cell not in evacuation_target_area_cells]

    # Step 2: 특별 구역과 일반 구역에서 보행자 수 계산
    num_special_area_agents = int(total_agents * special_area_ratio)
    num_general_area_agents = total_agents - num_special_area_agents

    # Step 3: 보행자 위치 선택 및 대피 구역 제외 확인
    selected_special_area_cells = []
    while len(selected_special_area_cells) < num_special_area_agents:
        cell = random.choice(special_area_cells)
        if cell not in evacuation_target_area_cells:
            selected_special_area_cells.append(cell)

    selected_general_area_cells = []
    while len(selected_general_area_cells) < num_general_area_agents:
        cell = random.choice(general_area_cells)
        if cell not in evacuation_target_area_cells:
            selected_general_area_cells.append(cell)

    # 모든 선택된 셀 합치기
    all_selected_cells = selected_special_area_cells + selected_general_area_cells
    random.shuffle(all_selected_cells)

    # Step 4: 보행자 유형별 수 계산
    pedestrians = []
    type_distribution = []

    for agent_type, ratio in agent_type_ratios.items():
        num_type_agents = int(total_agents * ratio)
        type_distribution.extend([agent_type] * num_type_agents)

    random.shuffle(type_distribution)

    # Step 5: 보행자 객체 생성
    for i, cell in enumerate(all_selected_cells):
        if i >= total_agents:
            break

        # 격자 좌표를 보행자의 초기 위치로 사용
        position = cell

        # 보행자 유형과 속도
        agent_type = type_distribution[i]
        speed = agent_types[agent_type]

        # 행동 패턴 무작위 선택
        behavior = random.choices(list(behaviors.keys()), weights=behaviors.values(), k=1)[0]

        # 특별 구역 여부 확인
        special_area = cell in selected_special_area_cells

        # 보행자 객체 생성 시 id 값 부여
        pedestrian = Pedestrian(position, speed, behavior, agent_type, special_area, id=i)
        pedestrians.append(pedestrian)

    return pedestrians


def convert_to_grid_indices(points, x_coords, y_coords):
    """점 좌표 리스트를 격자 인덱스 리스트로 변환합니다."""
    grid_indices = []
    for point in points:
        j = np.argmin(np.abs(x_coords - point[0]))
        i = np.argmin(np.abs(y_coords - point[1]))
        grid_indices.append((i, j))
    return grid_indices