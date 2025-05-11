import pygame
import numpy as np
import heapq
import time
import os
from math import sqrt

# Khởi tạo Pygame
pygame.init()

# Hằng số
CELL_SIZE = 12
WINDOW_SIZE = None

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = 	(178, 34, 34)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

class Node:
    def __init__(self, position, g, h):
        self.position = position  # (x, y)
        self.g = g  # Chi phí từ điểm bắt đầu
        self.h = h  # Chi phí ước lượng đến mục tiêu
        self.f = g + h  # Tổng chi phí

    def __lt__(self, other):
        return self.f < other.f

def neural_heuristic(a, b, maze):
    # Khoảng cách Euclidean
    euclidean_dist = sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    # Tính mật độ chướng ngại vật xung quanh
    x, y = a
    obstacle_density = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 1:
                obstacle_density += 1
    # Tăng trọng số để tránh chướng ngại vật
    neural_factor = 1.0 + 0.3 * obstacle_density
    return euclidean_dist * neural_factor

def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Khoảng cách Manhattan

def get_neighbors(position, maze, allow_diagonal=True):
    neighbors = []
    x, y = position
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4 hướng
    if allow_diagonal:
        directions += [(-1, -1), (1, 1), (-1, 1), (1, -1)]  # Thêm 4 hướng chéo
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def a_star(maze, start, target, screen, algorithm="Astar"):
    open_set = []
    heuristic = manhattan_heuristic if algorithm == "Astar" else neural_heuristic
    allow_diagonal = algorithm != "Astar"
    start_node = Node(start, 0, heuristic(start, target, maze) if algorithm != "Astar" else heuristic(start, target))
    heapq.heappush(open_set, start_node)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target, maze) if algorithm != "Astar" else heuristic(start, target)}
    closed_set = set()
    path = []

    while open_set:
        current = heapq.heappop(open_set)
        if current.position == target:
            while current.position in came_from:
                path.append(current.position)
                current = came_from[current.position]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current.position)
        neighbors = get_neighbors(current.position, maze, allow_diagonal)
        for neighbor in neighbors:
            if neighbor in closed_set:
                continue
            cost = 1.414 if abs(neighbor[0] - current.position[0]) + abs(neighbor[1] - current.position[1]) == 2 else 1
            tentative_g_score = g_score[current.position] + cost
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + (heuristic(neighbor, target, maze) if algorithm != "Astar" else heuristic(neighbor, target))
                if not any(neighbor == node.position for node in open_set):
                    heapq.heappush(open_set, Node(neighbor, tentative_g_score, heuristic(neighbor, target, maze) if algorithm != "Astar" else heuristic(neighbor, target)))

        # Vẽ trạng thái hiện tại
        screen.fill(BLACK)
        draw_maze(maze, path, screen, open_set, closed_set, current.position, algorithm)
        pygame.display.flip()

    return None  # Không tìm thấy đường đi

def draw_maze(maze, path, screen, open_set, closed_set, current_position, algorithm):
    # Vẽ mê cung
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if maze[x, y] == 1:
                pygame.draw.rect(screen, RED, rect)  # Chướng ngại vật
            else:
                pygame.draw.rect(screen, WHITE, rect)  # Không gian tự do
            pygame.draw.rect(screen, BLACK, rect, 1)  # Đường viền lưới

    # Vẽ tập mở
    for node in open_set:
        x, y = node.position
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, ORANGE, rect)  # Tập mở

    # Vẽ tập đóng
    for pos in closed_set:
        x, y = pos
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLACK, rect)  # Tập đóng

    # Vẽ node hiện tại
    if current_position:
        x, y = current_position
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GREEN, rect)  # Node hiện tại

    # Vẽ đường đi (đảm bảo vẽ cuối cùng để không bị ghi đè)
    for pos in path:
        x, y = pos
        rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLUE, rect)  # Đường đi

    # Hiển thị tên thuật toán
    font = pygame.font.SysFont(None, 24)
    algo_text = font.render(f"Algorithm: {algorithm}", True, WHITE)
    screen.blit(algo_text, (10, WINDOW_SIZE[1] - 40))

def load_maze_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        maze = np.array([[int(cell) for cell in line.strip().split()] for line in lines])
    return maze

# Tải mê cung
maze = load_maze_from_file("map.txt")
WINDOW_SIZE = (maze.shape[1] * CELL_SIZE, maze.shape[0] * CELL_SIZE + 50)

# Điểm bắt đầu và mục tiêu
start = (0, 0)
target = (maze.shape[0] - 1, maze.shape[1] - 1)

# Thiết lập cửa sổ Pygame
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Pathfinding Visualization")

# Chạy thuật toán Astar trước
results = {}
algo = "Astar"
start_time = time.time()
path = a_star(maze, start, target, screen, algo)
end_time = time.time()
elapsed_time = end_time - start_time
results[algo] = {"path": path, "time": elapsed_time}

# Hiển thị thời gian và đường đi của Astar
screen.fill(BLACK)
draw_maze(maze, path, screen, [], [], target, algo)  # Vẽ đường đi cuối cùng
font = pygame.font.SysFont(None, 24)
time_text = font.render(f"{algo} Time: {elapsed_time:.2f} seconds", True, WHITE)
text_x = (WINDOW_SIZE[0] - time_text.get_width()) // 2
text_y = WINDOW_SIZE[1] - 20
screen.blit(time_text, (text_x, text_y))
pygame.display.flip()

# Lưu kết quả Astar
output_dir = os.path.join("KetQua")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{algo}.png")
pygame.image.save(screen, output_path)

# In kết quả Astar
if path is None:
    print(f"{algo}: Không tìm thấy đường đi.")
else:
    print(f"{algo}: Đường đi tìm thấy! Độ dài: {len(path)}, Thời gian: {elapsed_time:.2f} giây")

# Vòng lặp chờ người dùng thoát để xem kết quả Astar
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

# Chạy thuật toán Neural_Astar sau khi thoát Astar
algo = "Neural_Astar"
start_time = time.time()
path = a_star(maze, start, target, screen, algo)
end_time = time.time()
elapsed_time = end_time - start_time
results[algo] = {"path": path, "time": elapsed_time}

# Hiển thị thời gian và đường đi của Neural_Astar
screen.fill(BLACK)
draw_maze(maze, path, screen, [], [], target, algo)  # Vẽ đường đi cuối cùng
time_text = font.render(f"{algo} Time: {elapsed_time:.2f} seconds", True, WHITE)
screen.blit(time_text, (text_x, text_y))
pygame.display.flip()

# Lưu kết quả Neural_Astar
output_path = os.path.join(output_dir, f"{algo}.png")
pygame.image.save(screen, output_path)

# In kết quả Neural_Astar
if path is None:
    print(f"{algo}: Không tìm thấy đường đi.")
else:
    print(f"{algo}: Đường đi tìm thấy! Độ dài: {len(path)}, Thời gian: {elapsed_time:.2f} giây")

# Vòng lặp để giữ cửa sổ mở cho Neural_Astar
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

# Thoát Pygame
pygame.quit()