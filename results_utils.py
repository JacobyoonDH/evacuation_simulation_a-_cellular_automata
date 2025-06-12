import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import os
import fiona
from shapely.geometry import LineString
import geopandas as gpd
import fiona
def save_agent_paths(pedestrians, output_dir):
    """
    각 보행자의 이동 경로를 CSV 파일로 저장합니다.

    Parameters:
        pedestrians (list): 보행자 객체 리스트
        output_dir (str): 경로를 저장할 디렉토리
    """
    agent_paths_dir = os.path.join(output_dir, "agent_paths")
    os.makedirs(agent_paths_dir, exist_ok=True)

    for idx, pedestrian in enumerate(pedestrians):
        file_path = os.path.join(agent_paths_dir, f"agent_{idx + 1}.csv")
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "X", "Y"])
            for step, position in enumerate(pedestrian.path):
                writer.writerow([step, position[1], position[0]])

def save_congestion_grid(congestion_grid, output_dir, filename):
    """
    혼잡도 그리드를 CSV 파일로 저장합니다.

    Parameters:
        congestion_grid (numpy.ndarray): 혼잡도 그리드
        output_dir (str): 그리드를 저장할 디렉토리
        filename (str): 저장할 파일 이름
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    np.savetxt(file_path, congestion_grid, delimiter=",", fmt="%d")

def save_congestion_image(congestion_grid, output_dir, filename, title):
    """
    혼잡도 그리드를 이미지 파일로 저장합니다.

    Parameters:
        congestion_grid (numpy.ndarray): 혼잡도 그리드
        output_dir (str): 이미지를 저장할 디렉토리
        filename (str): 저장할 파일 이름
        title (str): 이미지 제목
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 8))
    plt.imshow(congestion_grid, cmap='hot', origin='upper')
    plt.colorbar(label="Congestion Level")
    plt.title(title)
    plt.savefig(file_path)
    plt.close()
# Shapefile 저장에서 한글 문제 해결 - 영어로 변경한 보행자 타입

def save_pedestrian_paths_to_shp(pedestrians, x_coords, y_coords, output_path, crs="EPSG:5181"):
    """
    Save pedestrian paths as a Shapefile with type field converted to English.

    Parameters:
        pedestrians (list): List of pedestrian objects
        x_coords (numpy.ndarray): Array of X coordinates
        y_coords (numpy.ndarray): Array of Y coordinates
        output_path (str): Path to save the Shapefile
        crs (str): Coordinate reference system (default: EPSG:5181)
    """
    # Mapping for pedestrian type translation
    type_mapping = {
        "노인/어린이": "Elderly/Children",
        "중장년": "Middle-aged",
        "청소년/청년": "Youth",
        "장애인": "Disabled"
    }

    # Data for Shapefile
    records = []

    for pedestrian in pedestrians:
        if len(pedestrian.path_congest) > 1:  # If the path exists
            # Convert path to LineString
            path_coords = [(x_coords[pos[1]], y_coords[pos[0]]) for pos in pedestrian.path_congest]
            geometry = LineString(path_coords)

            # Append pedestrian data
            records.append({
                "ID": pedestrian.id,
                "Speed": pedestrian.speed,
                "Behavior": pedestrian.behavior,
                "Type": type_mapping.get(pedestrian.type, pedestrian.type),  # Convert type to English
                "Special": pedestrian.special_area,
                "KnowsTgt": pedestrian.knows_target,
                "GoalSucc": pedestrian.goal_reached,
                "Distance": pedestrian.total_distance,
                "geometry": geometry
            })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Save as Shapefile
    gdf.to_file(output_path, driver="ESRI Shapefile")
    print(f"Pedestrian paths saved to Shapefile: {output_path}")


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from openpyxl import Workbook

def save_evacuations_to_excel(pedestrians, evacuation_target_area_gdf, x_coords, y_coords, filename=r"C:\Users\doohu\Desktop\digital_twin\results\evacuation_results.xlsx"):
    """
    보행자 시뮬레이션 결과를 Excel 파일로 저장하며, 다섯 가지 시트를 포함합니다.

    Parameters:
        pedestrians (list): 보행자 객체 리스트
        evacuation_target_area_gdf (GeoDataFrame): 대피소 영역 GeoDataFrame
        x_coords, y_coords (numpy.ndarray): 격자 좌표를 EPSG:5181로 변환하는 배열
        filename (str): 저장할 Excel 파일 이름
    """
    # 결과를 저장할 데이터 리스트
    results = []
    for pedestrian in pedestrians:
        escape_time = round(pedestrian.total_distance / pedestrian.speed, 2) if pedestrian.speed > 0 else None

        result = {
            "ID": pedestrian.id,
            "현재 위치": pedestrian.position,
            "속도": pedestrian.speed,
            "행동 유형": pedestrian.behavior,
            "보행자 타입": pedestrian.type,
            "시장 지역 시작 여부": pedestrian.special_area,
            "대피소 인지 여부": pedestrian.knows_target,
            "대피 목표": pedestrian.goal,
            "대피 성공 여부": pedestrian.goal_reached,
            "실제 경로": pedestrian.real_path,
            "총 이동 거리(미터)": pedestrian.total_distance,
            "소요 시간(초)": escape_time
        }
        results.append(result)

    # DataFrame 생성
    df = pd.DataFrame(results)

    # Sheet 1: 각 보행자별 상세 결과
    sheet1 = df

    # Sheet 2: 행동 유형별 통계
    behavior_stats = df.groupby("행동 유형").agg(
        총_보행자수=("ID", "count"),
        대피_성공자수=("대피 성공 여부", "sum"),
        대피_성공률_퍼센트=("대피 성공 여부", lambda x: round(x.mean() * 100, 2)),
        대피소_인지_보행자수=("대피소 인지 여부", "sum"),
        평균_소요시간_성공=("소요 시간(초)", "mean"),
        평균_이동거리_성공=("총 이동 거리(미터)", "mean")
    )
    behavior_stats.index.name = "행동 유형"

    # Sheet 3: 보행자 타입별 통계
    type_stats = df.groupby("보행자 타입").agg(
        총_보행자수=("ID", "count"),
        대피_성공자수=("대피 성공 여부", "sum"),
        대피_성공률_퍼센트=("대피 성공 여부", lambda x: round(x.mean() * 100, 2)),
        대피소_인지_보행자수=("대피소 인지 여부", "sum"),
        평균_소요시간_성공=("소요 시간(초)", "mean"),
        평균_이동거리_성공=("총 이동 거리(미터)", "mean")
    )
    type_stats.index.name = "보행자 타입"

    # Sheet 4: 대피소별 대피 현황
    evacuation_data = []
    pedestrian_goals = gpd.GeoDataFrame(
        df[df["대피 목표"].notnull()],
        geometry=[
            Point(x_coords[goal[1]], y_coords[goal[0]]) if goal is not None else None
            for goal in df["대피 목표"]
        ],
        crs="EPSG:5181"
    )

    pedestrian_goals.dropna(subset=["geometry"], inplace=True)

    for idx, shelter in evacuation_target_area_gdf.iterrows():
        shelter_name = f"{shelter.get('A24', '')}{shelter.get('A25', '')}"
        shelter_geometry = shelter.geometry
        reached_pedestrians = pedestrian_goals[pedestrian_goals.geometry.within(shelter_geometry)]

        evacuation_data.append({
            "대피소 이름": shelter_name,
            "대피소 좌표": (shelter_geometry.centroid.x, shelter_geometry.centroid.y),
            "도착한 보행자 수": len(reached_pedestrians),
            "평균 소요 시간(초)": reached_pedestrians["소요 시간(초)"].mean() if len(reached_pedestrians) > 0 else None,
            "평균 이동 거리(미터)": reached_pedestrians["총 이동 거리(미터)"].mean() if len(reached_pedestrians) > 0 else None,
            "보행자 타입별 도착 수": reached_pedestrians["보행자 타입"].value_counts().to_dict(),
            "행동 유형별 도착 수": reached_pedestrians["행동 유형"].value_counts().to_dict()
        })

    shelter_stats = pd.DataFrame(evacuation_data)

    # Sheet 5: 대피소 미배정 보행자 통계
    unassigned_pedestrians = df[df["대피 목표"].isnull()]
    unassigned_stats = {
        "총 미배정 보행자 수": len(unassigned_pedestrians),
        "평균 이동 거리(미터)": unassigned_pedestrians["총 이동 거리(미터)"].mean(),
        "보행자 타입별 수": unassigned_pedestrians["보행자 타입"].value_counts().to_dict(),
        "행동 유형별 수": unassigned_pedestrians["행동 유형"].value_counts().to_dict()
    }
    unassigned_df = pd.DataFrame([unassigned_stats])

    # Sheet 6: 전체 요약 통계
    overall_stats = {
        "총 보행자 수": len(pedestrians),
        "대피 성공률 (%)": round(df["대피 성공 여부"].mean() * 100, 2),
        "평균 소요 시간 (초)": df["소요 시간(초)"].mean(),
        "평균 이동 거리 (미터)": df["총 이동 거리(미터)"].mean(),
    }
    overall_df = pd.DataFrame([overall_stats])

    # Excel 파일로 저장
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        sheet1.to_excel(writer, sheet_name="보행자 상세 결과", index=False)
        behavior_stats.to_excel(writer, sheet_name="행동 유형별 통계")
        type_stats.to_excel(writer, sheet_name="보행자 타입별 통계")
        shelter_stats.to_excel(writer, sheet_name="대피소별 대피 현황", index=False)
        unassigned_df.to_excel(writer, sheet_name="대피소 미배정 보행자", index=False)
        overall_df.to_excel(writer, sheet_name="전체 요약 통계", index=False)

    print(f"Evacuation results saved to '{filename}' with 6 sheets.")
