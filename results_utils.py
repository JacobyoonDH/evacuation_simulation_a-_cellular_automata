import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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
