import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

from grid_utils import create_grid
from pedestrian_utils import generate_pedestrians
from simulation import run_simulation
from results_utils import save_agent_paths, save_congestion_grid, save_congestion_image
from visualization import visualize_congestion

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

print("Reprojection complete. All data is now in EPSG:5181.")

# 사용자 입력 받기
total_agents = int(input("총 보행자 수를 입력하세요: "))
special_area_ratio = float(input("시장 영역에서 시작할 보행자 비율을 입력하세요 (0 ~ 1 사이): "))

print("\n보행자 유형 비율을 입력하세요 (0 ~ 1 사이의 값으로 입력):")
agent_type_ratios = {
    '노인/어린이': float(input("노인/어린이 비율: ")),
    '중장년': float(input("중장년 비율: ")),
    '청소년/청년': float(input("청소년/청년 비율: ")),
    '장애인': float(input("장애인 비율: ")),
    '집단 대피': float(input("집단 대피 비율: ")),
}

print("\n보행자 행동 패턴 비율을 입력하세요 (0 ~ 1 사이의 값으로 입력):")
behaviors = {
    'knows_specific': float(input("특정 대피소를 알고 있는 비율 (knows_specific): ")),
    'knows_all': float(input("모든 대피소를 알고 있는 비율 (knows_all): ")),
    'exploratory': float(input("탐색형 비율 (exploratory): ")),
}

# 보행자 유형별 이동 속도
agent_types = {
    '노인/어린이': 1.0,
    '중장년': 1.3,
    '청소년/청년': 1.9,
    '장애인': 0.71,
    '집단 대피': 0.71 * 1.3,
}

# 격자 생성
grid, x_coords, y_coords = create_grid(area_gdf, buildings_gdf, resolution=3)

# 보행자 생성
pedestrians = generate_pedestrians(total_agents, agent_type_ratios, agent_types, behaviors, creation_area_gdf, special_area_gdf, special_area_ratio)

# 보행자 초기 위치 시각화
fig, ax = plt.subplots(figsize=(12, 10))
area_gdf.plot(ax=ax, color='lightgreen', edgecolor='black', label='도보 이동 가능 지역')
buildings_gdf.plot(ax=ax, color='black', alpha=0.7, label='건물')
special_area_gdf.plot(ax=ax, color='purple', alpha=0.5, label='시장 영역')

special_area_positions = [ped.position for ped in pedestrians if ped.special_area]
general_area_positions = [ped.position for ped in pedestrians if not ped.special_area]

# 보행자 초기 위치 시각화
ax.scatter(
    [pos[0] for pos in special_area_positions],
    [pos[1] for pos in special_area_positions],
    color='red',
    s=30,
    label='Special Area Agents'
)

ax.scatter(
    [pos[0] for pos in general_area_positions],
    [pos[1] for pos in general_area_positions],
    color='green',
    s=30,
    label='General Area Agents'
)

ax.legend()
plt.title("Grid 및 보행자 초기 위치 시각화")
plt.xlabel("X Coordinate (meters in EPSG:5181)")
plt.ylabel("Y Coordinate (meters in EPSG:5181)")
plt.show()

# 시뮬레이션 실행
evacuation_target_cells = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == 0]
dynamic_congestion_grid, accumulated_congestion_grid = run_simulation(grid, pedestrians, evacuation_target_cells, steps=100)

# 결과 저장 경로 설정
output_dir = r"C:\Users\doohu\Desktop\대학원\디트\1209_디트시뮬레이션_혼잡도포함"

# 결과 저장
save_agent_paths(pedestrians, output_dir)
save_congestion_grid(dynamic_congestion_grid, output_dir, "dynamic_congestion.csv")
save_congestion_grid(accumulated_congestion_grid, output_dir, "accumulated_congestion.csv")
save_congestion_image(dynamic_congestion_grid, output_dir, "dynamic_congestion.png", "Dynamic Congestion (During Simulation)")
save_congestion_image(accumulated_congestion_grid, output_dir, "accumulated_congestion.png", "Accumulated Congestion (Final Result)")

# 최종 혼잡도 시각화
visualize_congestion(area_gdf, buildings_gdf, dynamic_congestion_grid, accumulated_congestion_grid, x_coords, y_coords)

print("\nSimulation complete. Results saved to:", output_dir)

