import random
from PIL import Image, ImageDraw

DIRS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

def is_inside(x, y, width, height):
    return 0 <= x < height and 0 <= y < width

def create_map(width, height):
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    maze = [[1 for _ in range(width)] for _ in range(height)]
    start_x = random.randrange(1, height, 2)
    start_y = random.randrange(1, width, 2)
    maze[start_x][start_y] = 0

    walls = []
    for dx, dy in DIRS:
        nx, ny = start_x + dx, start_y + dy
        if is_inside(nx, ny, width, height):
            walls.append((start_x, start_y, nx, ny))

    while walls:
        x1, y1, x2, y2 = walls.pop(random.randint(0, len(walls) - 1))
        if not is_inside(x2, y2, width, height):
            continue
        if maze[x2][y2] == 0:
            continue

        maze[x2][y2] = 0
        maze[(x1 + x2) // 2][(y1 + y2) // 2] = 0

        for dx, dy in DIRS:
            nx, ny = x2 + dx, y2 + dy
            if is_inside(nx, ny, width, height) and maze[nx][ny] == 1:
                walls.append((x2, y2, nx, ny))

    return maze

def save_map_to_file(maze, filename):
    with open(filename, 'w') as f:
        for row in maze[1:-1]:
            line = ' '.join(str(cell) for cell in row[1:-1])
            f.write(line + '\n')

def save_map_as_image(maze, filename, cell_size=10):
    height = len(maze)
    width = len(maze[0])
    img = Image.new("RGB", (width * cell_size, height * cell_size), "white")
    draw = ImageDraw.Draw(img)

    for i in range(height):
        for j in range(width):
            if maze[i][j] == 1:
                x0, y0 = j * cell_size, i * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                draw.rectangle([x0, y0, x1, y1], fill="red")

    img.save(filename)

if __name__ == "__main__":
    width, height = 40, 40
    maze = create_map(width, height)

    save_map_to_file(maze, "map.txt")
    save_map_as_image(maze, "map.png")

