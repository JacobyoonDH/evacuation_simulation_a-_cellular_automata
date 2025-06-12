import numpy as np
from shapely.geometry import Point, box
from scipy.ndimage import label
from shapely.geometry import Point, box

def get_evacuation_target_cells(grid, x_coords, y_coords, evacuation_target_area_gdf, resolution=1):
    """
    격자에서 대피 목표 영역과 교차하는 셀을 추출합니다.

    Parameters:
        grid (np.array): 2D 격자 (0 = 이동 가능, 1 = 장애물)
        x_coords (np.array): X 좌표 배열
        y_coords (np.array): Y 좌표 배열
        evacuation_target_area_gdf (GeoDataFrame): 대피 목표 영역 GeoDataFrame
        resolution (int): 격자 해상도 (기본값 3)

    Returns:
        list: 대피 목표 셀의 (i, j) 인덱스 리스트
    """
    evacuation_target_cells = []

    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # 셀의 경계를 생성
            cell = box(x, y, x + resolution, y + resolution)
            # 격자가 이동 가능한 셀인지 확인하고 대피 영역과 교차하는지 확인
            if grid[i, j] == 0 and evacuation_target_area_gdf.intersects(cell).any():
                evacuation_target_cells.append((i, j))

    return evacuation_target_cells


def remove_disconnected_cells(grid):
    """
    격자에서 가장 큰 연결된 컴포넌트만 유지하고 나머지 이동 가능한 셀을 이동 불가능(1)으로 변경합니다.

    Parameters:
        grid (numpy.ndarray): 이동 가능(0), 이동 불가능(1)으로 이루어진 격자.

    Returns:
        numpy.ndarray: 동떨어진 이동 불가능 셀이 제거된 격자.
    """
    # 연결된 컴포넌트 레이블링
    labeled_grid, num_features = label(grid == 0)

    # 각 컴포넌트의 크기 계산
    component_sizes = np.bincount(labeled_grid.ravel())

    # 가장 큰 컴포넌트의 레이블 (0은 배경이므로 제외)
    largest_component_label = component_sizes[1:].argmax() + 1

    # 가장 큰 컴포넌트만 이동 가능하게 유지 (나머지는 이동 불가능으로 변경)
    cleaned_grid = np.where(labeled_grid == largest_component_label, 0, 1)

    return cleaned_grid



def create_grid(area_gdf, buildings_gdf, resolution=1):
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
