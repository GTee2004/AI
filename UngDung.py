import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import heapq
import torch
import torch.nn as nn
import cv2

# Neural A* Network
class NeuralHeuristic(nn.Module):
    def __init__(self):
        super(NeuralHeuristic, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Giả sử bản đồ 64x64
        self.fc2 = nn.Linear(128, 1)  # Dự đoán heuristic
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hàm chuyển đổi ảnh thành lưới nhị phân ngay trong ứng dụng
def image_to_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # Điều chỉnh ngưỡng
    grid = (binary > 0).astype(np.uint8)  # 0: đi được, 1: chướng ngại
    grid = cv2.resize(grid, (64, 64), interpolation=cv2.INTER_NEAREST)
    return grid

# Thuật toán Neural A* với fallback heuristic
def neural_a_star(grid, start, goal, model=None):
    rows, cols = grid.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    # Fallback heuristic nếu model chưa có
    def euclidean_heuristic(node, goal):
        return ((node[0] - goal[0])**2 + (node[1] - goal[1])**2)**0.5

    # Khởi tạo f_score
    if model is not None:
        grid_tensor = torch.tensor(grid.astype(float), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        f_score = {start: model(grid_tensor).item()}
    else:
        f_score = {start: euclidean_heuristic(start, goal)}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 hướng
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and 
                grid[neighbor] == 0):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    if model is not None:
                        f_score[neighbor] = tentative_g_score + model(grid_tensor).item()
                    else:
                        f_score[neighbor] = tentative_g_score + euclidean_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

# Ứng dụng Tkinter
class PathPlanningApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Neural A* Path Planning")
        self.image_path = image_path

        # Đọc và xử lý ảnh thành grid ngay trong __init__
        self.img_cv = cv2.imread(self.image_path)
        self.grid = image_to_grid(self.img_cv)
        self.model = NeuralHeuristic()  # Giả sử chưa huấn luyện, dùng heuristic mặc định
        self.start = None
        self.goal = None

        # Tải và hiển thị ảnh
        self.img_pil = Image.fromarray(cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB))
        self.img_pil = self.img_pil.resize((512, 512))
        self.photo = ImageTk.PhotoImage(self.img_pil)
        self.canvas = tk.Canvas(root, width=512, height=512)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.pack()

        # Nút chạy thuật toán
        self.run_button = tk.Button(root, text="Find Path", command=self.find_path)
        self.run_button.pack()

        # Sự kiện nhấp chuột
        self.canvas.bind("<Button-1>", self.set_points)

    def set_points(self, event):
        x, y = event.x, event.y
        grid_x, grid_y = int(x / 512 * self.grid.shape[1]), int(y / 512 * self.grid.shape[0])
        if self.grid[grid_y, grid_x] == 0:  # Chỉ chọn vùng đi được
            if self.start is None:
                self.start = (grid_y, grid_x)
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="green")
            elif self.goal is None:
                self.goal = (grid_y, grid_x)
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red")

    def find_path(self):
        if self.start and self.goal:
            path = neural_a_star(self.grid, self.start, self.goal, self.model)
            if path:
                print("Path found:", path)  # Gỡ lỗi
                for i in range(len(path)-1):
                    x1, y1 = path[i][1] * 512 / self.grid.shape[1], path[i][0] * 512 / self.grid.shape[0]
                    x2, y2 = path[i+1][1] * 512 / self.grid.shape[1], path[i+1][0] * 512 / self.grid.shape[0]
                    self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
            else:
                print("No path found")

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = PathPlanningApp(root, "bando1.png")
    root.mainloop()