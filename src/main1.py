import taichi as ti
import math

ti.init(arch=ti.cpu)

cube_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
cube_edges = ti.Vector.field(2, dtype=ti.i32, shape=12)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

# 初始化立方体顶点
cube_vertices[0] = ti.Vector([-1.0, -1.0, -1.0])
cube_vertices[1] = ti.Vector([1.0, -1.0, -1.0])
cube_vertices[2] = ti.Vector([1.0, -1.0, 1.0])
cube_vertices[3] = ti.Vector([-1.0, -1.0, 1.0])
cube_vertices[4] = ti.Vector([-1.0, 1.0, -1.0])
cube_vertices[5] = ti.Vector([1.0, 1.0, -1.0])
cube_vertices[6] = ti.Vector([1.0, 1.0, 1.0])
cube_vertices[7] = ti.Vector([-1.0, 1.0, 1.0])

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]
for i, (a, b) in enumerate(edges):
    cube_edges[i] = ti.Vector([a, b])


@ti.func
def get_model_matrix(angle_z: ti.f32, angle_y: ti.f32):
    rad_z = angle_z * math.pi / 180.0
    rad_y = angle_y * math.pi / 180.0

    cz = ti.cos(rad_z)
    sz = ti.sin(rad_z)
    cy = ti.cos(rad_y)
    sy = ti.sin(rad_y)

    rot_z = ti.Matrix([
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    rot_y = ti.Matrix([
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return rot_y @ rot_z


@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])


@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar

    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o


@ti.kernel
def compute_transform(angle_z: ti.f32, angle_y: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 3.0])  # 拉近相机距离
    model = get_model_matrix(angle_z, angle_y)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(60.0, 1.0, 0.1, 10.0)  # 缩小远平面

    mvp = proj @ view @ model

    for i in range(8):
        v = cube_vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4

        if v_clip[3] != 0.0:
            v_ndc = v_clip / v_clip[3]
        else:
            v_ndc = v_clip

        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0


def main():
    gui = ti.GUI("3D Cube", res=(700, 700))
    angle_z = 0.0
    angle_y = 0.0

    print("Controls:")
    print("  A/D: Rotate around Z-axis")
    print("  W/S: Rotate around Y-axis")
    print("  ESC: Exit")

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle_z -= 5.0
            elif gui.event.key == 'd':
                angle_z += 5.0
            elif gui.event.key == 'w':
                angle_y += 5.0
            elif gui.event.key == 's':
                angle_y -= 5.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        compute_transform(angle_z, angle_y)

        gui.clear(0x000000)

        # 调试：打印第一个点的坐标
        if angle_z == 0:
            p0 = screen_coords[0]
            print(f"Point 0: ({p0[0]:.3f}, {p0[1]:.3f})")

        for i in range(12):
            edge = cube_edges[i]
            idx0 = int(edge[0])
            idx1 = int(edge[1])

            p0 = screen_coords[idx0]
            p1 = screen_coords[idx1]

            # 只绘制在屏幕内的点
            if 0 <= p0[0] <= 1 and 0 <= p0[1] <= 1 and 0 <= p1[0] <= 1 and 0 <= p1[1] <= 1:
                x0 = p0[0] * 700
                y0 = p0[1] * 700
                x1 = p1[0] * 700
                y1 = p1[1] * 700
                gui.line(x0, y0, x1, y1, color=0x00FF00, radius=2)

        gui.show()


if __name__ == '__main__':
    main()