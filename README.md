# 计算机图形学实验报告

## 实验一：三维空间中的坐标变换与三角形线框渲染

**姓名：** 王宇畅
**学号：** 202311030025
**授课教师：** 张鸿文
**助教：** 张怡冉
**日期：** 2026年3月24日

---

## 一、项目架构

采用标准的 src 布局，实现代码与配置的物理隔离：

```
CG-Lab/
├── .venv/                 # 虚拟环境
├── assets/                # 演示资源
│   ├── demo.gif           # 三角形旋转演示
├── src/
│   └── Work1/
│       ├── main.py        # 主程序入口
└── README.md
```

---

## 二、核心代码逻辑

### 2.1 配置参数
```python
# 窗口分辨率
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700

# 相机参数
EYE_POS = (0.0, 0.0, 5.0)      # 相机位置
EYE_FOV = 45.0                   # 视场角
Z_NEAR = 0.1                     # 近截面
Z_FAR = 50.0                     # 远截面
```

### 2.2 模型变换矩阵 (get_model_matrix)
```python
@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
```
**功能**：实现绕Z轴旋转的模型变换，将角度转换为弧度后构建旋转矩阵。

### 2.3 视图变换矩阵 (get_view_matrix)
```python
@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
```
**功能**：将相机从指定位置平移到世界坐标系原点。

### 2.4 投影变换矩阵 (get_projection_matrix)
```python
@ti.func
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([...])      # 透视到正交矩阵
    M_ortho = ti.Matrix([...])     # 正交投影矩阵
    return M_ortho @ M_p2o
```
**功能**：将透视平截头体压缩为正交长方体，再映射到[-1,1]³空间。

### 2.5 顶点变换 (compute_transform)
```python
@ti.kernel
def compute_transform(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model          # 组合MVP矩阵
    
    for i in range(3):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]      # 透视除法
        
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0
```
**功能**：在GPU上并行计算三个顶点的MVP变换，并映射到屏幕空间。

### 2.6 渲染循环 (main)
```python
def main():
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]
    
    gui = ti.GUI("3D Transformation", res=(700, 700))
    angle = 0.0
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        compute_transform(angle)        # 更新变换
        
        # 绘制三角形边框
        a, b, c = screen_coords[0], screen_coords[1], screen_coords[2]
        gui.line(a, b, radius=2, color=0xFF0000)
        gui.line(b, c, radius=2, color=0x00FF00)
        gui.line(c, a, radius=2, color=0x0000FF)
        
        gui.show()
```
**功能**：初始化三角形顶点，处理键盘输入，实时渲染旋转的三角形。

---

## 三、运行效果展示

### 3.1 GPU后端调用
程序成功调用 Taichi 的 CPU/GPU 后端进行并行计算：
```
[Taichi] version 1.7.4, llvm 15.0.1, commit b4b956fd, win, python 3.12.0
[Taichi] Starting on arch=x64
```

### 3.2 三角形旋转效果

**初始状态**：
- 三角形显示在屏幕中央
- 三个顶点颜色：红、绿、蓝

**A/D键交互**：
- **A键**：三角形绕Z轴逆时针旋转（角度+10°）
- **D键**：三角形绕Z轴顺时针旋转（角度-10°）

![运行效果](assets/demo.gif?raw=true)

---
## 四、选做内容：3D立方体线框渲染

### 4.1 实现思路

在基础实验的基础上，将二维三角形升级为三维立方体，主要改动包括：

#### 数据结构扩展：
- **8个顶点**：坐标范围[-1, 1]，中心在原点
- **12条边**：连接顶点构成立方体线框
- 使用 `cube_edges` 存储边的连接关系

#### 双轴旋转矩阵：
- 保留绕Z轴旋转
- 新增绕Y轴旋转矩阵
- 组合矩阵：`model = rot_y @ rot_z`

#### 渲染逻辑修改：
- 从绘制3条边改为绘制12条边
- 使用循环遍历所有边进行绘制

### 4.2 核心代码实现

#### 立方体顶点与边的定义：

```python
# 8个顶点（边长为2，中心在原点）
cube_vertices[0] = [-1.0, -1.0, -1.0]  # 底面后左下
cube_vertices[1] = [ 1.0, -1.0, -1.0]  # 底面后右下
cube_vertices[2] = [ 1.0, -1.0,  1.0]  # 底面前右下
cube_vertices[3] = [-1.0, -1.0,  1.0]  # 底面前左下
cube_vertices[4] = [-1.0,  1.0, -1.0]  # 顶面后左上
cube_vertices[5] = [ 1.0,  1.0, -1.0]  # 顶面后右上
cube_vertices[6] = [ 1.0,  1.0,  1.0]  # 顶面前右上
cube_vertices[7] = [-1.0,  1.0,  1.0]  # 顶面前左上

# 12条边
edges = [
    (0,1), (1,2), (2,3), (3,0),  # 底面
    (4,5), (5,6), (6,7), (7,4),  # 顶面
    (0,4), (1,5), (2,6), (3,7)   # 垂直边
]
```
#### 立方体渲染循环：
```python
for i in range(12):
    edge = cube_edges[i]
    idx0 = int(edge[0])
    idx1 = int(edge[1])
    
    p0 = screen_coords[idx0]
    p1 = screen_coords[idx1]
    
    p0_pos = (p0[0] * 700, p0[1] * 700)
    p1_pos = (p1[0] * 700, p1[1] * 700)
    
    gui.line(p0_pos, p1_pos, radius=2, color=0x00FF00)
```
### 4.3 交互控制

| 按键 | 功能 | 旋转轴 |
|------|------|--------|
| A | 逆时针旋转 | Z轴 |
| D | 顺时针旋转 | Z轴 |
| W | 向上旋转 | Y轴 |
| S | 向下旋转 | Y轴 |
| ESC | 退出程序 | - |

### 4.4 运行效果

#### 视觉特点：
- **绿色线框立方体**，中心在屏幕中央
- **透视投影**产生近大远小效果
- **双轴旋转**展现完整的三维空间感
- 旋转时12条边始终保持正确连接

#### 3D效果验证：
- 绕Y轴旋转时，正面和背面交替可见
- 绕Z轴旋转时，立方体保持直立但水平旋转
- 同时旋转时产生复合运动轨迹
# 计算机图形学实验报告

## 实验一：三维空间中的坐标变换与三角形线框渲染

**姓名：** 王宇畅
**学号：** 202311030025
**授课教师：** 张鸿文
**助教：** 张怡冉
**日期：** 2026年3月24日

---

## 一、项目架构

采用标准的 src 布局，实现代码与配置的物理隔离：

```
CG-Lab/
├── .venv/                 # 虚拟环境
├── assets/                # 演示资源
│   ├── demo.gif           # 三角形旋转演示
├── src/
│   └── Work1/
│       ├── main.py        # 主程序入口
└── README.md
```

---

## 二、核心代码逻辑

### 2.1 配置参数
```python
# 窗口分辨率
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700

# 相机参数
EYE_POS = (0.0, 0.0, 5.0)      # 相机位置
EYE_FOV = 45.0                   # 视场角
Z_NEAR = 0.1                     # 近截面
Z_FAR = 50.0                     # 远截面
```

### 2.2 模型变换矩阵 (get_model_matrix)
```python
@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
```
**功能**：实现绕Z轴旋转的模型变换，将角度转换为弧度后构建旋转矩阵。

### 2.3 视图变换矩阵 (get_view_matrix)
```python
@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
```
**功能**：将相机从指定位置平移到世界坐标系原点。

### 2.4 投影变换矩阵 (get_projection_matrix)
```python
@ti.func
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([...])      # 透视到正交矩阵
    M_ortho = ti.Matrix([...])     # 正交投影矩阵
    return M_ortho @ M_p2o
```
**功能**：将透视平截头体压缩为正交长方体，再映射到[-1,1]³空间。

### 2.5 顶点变换 (compute_transform)
```python
@ti.kernel
def compute_transform(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model          # 组合MVP矩阵
    
    for i in range(3):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]      # 透视除法
        
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0
```
**功能**：在GPU上并行计算三个顶点的MVP变换，并映射到屏幕空间。

### 2.6 渲染循环 (main)
```python
def main():
    vertices[0] = [2.0, 0.0, -2.0]
    vertices[1] = [0.0, 2.0, -2.0]
    vertices[2] = [-2.0, 0.0, -2.0]
    
    gui = ti.GUI("3D Transformation", res=(700, 700))
    angle = 0.0
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        compute_transform(angle)        # 更新变换
        
        # 绘制三角形边框
        a, b, c = screen_coords[0], screen_coords[1], screen_coords[2]
        gui.line(a, b, radius=2, color=0xFF0000)
        gui.line(b, c, radius=2, color=0x00FF00)
        gui.line(c, a, radius=2, color=0x0000FF)
        
        gui.show()
```
**功能**：初始化三角形顶点，处理键盘输入，实时渲染旋转的三角形。

---

## 三、运行效果展示

### 3.1 GPU后端调用
程序成功调用 Taichi 的 CPU/GPU 后端进行并行计算：
```
[Taichi] version 1.7.4, llvm 15.0.1, commit b4b956fd, win, python 3.12.0
[Taichi] Starting on arch=x64
```

### 3.2 三角形旋转效果

**初始状态**：
- 三角形显示在屏幕中央
- 三个顶点颜色：红、绿、蓝

**A/D键交互**：
- **A键**：三角形绕Z轴逆时针旋转（角度+10°）
- **D键**：三角形绕Z轴顺时针旋转（角度-10°）

![运行效果](assets/demo.gif?raw=true)

---
## 四、选做内容：3D立方体线框渲染

### 4.1 实现思路

在基础实验的基础上，将二维三角形升级为三维立方体，主要改动包括：

#### 数据结构扩展：
- **8个顶点**：坐标范围[-1, 1]，中心在原点
- **12条边**：连接顶点构成立方体线框
- 使用 `cube_edges` 存储边的连接关系

#### 双轴旋转矩阵：
- 保留绕Z轴旋转
- 新增绕Y轴旋转矩阵
- 组合矩阵：`model = rot_y @ rot_z`

#### 渲染逻辑修改：
- 从绘制3条边改为绘制12条边
- 使用循环遍历所有边进行绘制

### 4.2 核心代码实现

#### 立方体顶点与边的定义：

```python
# 8个顶点（边长为2，中心在原点）
cube_vertices[0] = [-1.0, -1.0, -1.0]  # 底面后左下
cube_vertices[1] = [ 1.0, -1.0, -1.0]  # 底面后右下
cube_vertices[2] = [ 1.0, -1.0,  1.0]  # 底面前右下
cube_vertices[3] = [-1.0, -1.0,  1.0]  # 底面前左下
cube_vertices[4] = [-1.0,  1.0, -1.0]  # 顶面后左上
cube_vertices[5] = [ 1.0,  1.0, -1.0]  # 顶面后右上
cube_vertices[6] = [ 1.0,  1.0,  1.0]  # 顶面前右上
cube_vertices[7] = [-1.0,  1.0,  1.0]  # 顶面前左上

# 12条边
edges = [
    (0,1), (1,2), (2,3), (3,0),  # 底面
    (4,5), (5,6), (6,7), (7,4),  # 顶面
    (0,4), (1,5), (2,6), (3,7)   # 垂直边
]
```
#### 立方体渲染循环：
```python
for i in range(12):
    edge = cube_edges[i]
    idx0 = int(edge[0])
    idx1 = int(edge[1])
    
    p0 = screen_coords[idx0]
    p1 = screen_coords[idx1]
    
    p0_pos = (p0[0] * 700, p0[1] * 700)
    p1_pos = (p1[0] * 700, p1[1] * 700)
    
    gui.line(p0_pos, p1_pos, radius=2, color=0x00FF00)
```
### 4.3 交互控制

| 按键 | 功能 | 旋转轴 |
|------|------|--------|
| A | 逆时针旋转 | Z轴 |
| D | 顺时针旋转 | Z轴 |
| W | 向上旋转 | Y轴 |
| S | 向下旋转 | Y轴 |
| ESC | 退出程序 | - |

### 4.4 运行效果

#### 视觉特点：
- **绿色线框立方体**，中心在屏幕中央
- **透视投影**产生近大远小效果
- **双轴旋转**展现完整的三维空间感
- 旋转时12条边始终保持正确连接

#### 3D效果验证：
- 绕Y轴旋转时，正面和背面交替可见
- 绕Z轴旋转时，立方体保持直立但水平旋转
- 同时旋转时产生复合运动轨迹

![运行效果](assets/demo1.gif)
---
## 五、实验总结

本次实验成功实现了三维图形渲染管线的核心流程：

1. **理论验证**：通过代码实现了模型、视图、投影矩阵的数学推导
2. **实践应用**：成功在屏幕上渲染出可交互旋转的3D三角形
3. **性能优化**：利用Taichi框架实现GPU并行计算
4. **深入理解**：掌握了齐次坐标、矩阵变换、透视投影等核心概念

实验过程中遇到的坐标系统、矩阵运算、Taichi框架限制等问题，通过查阅文档和调试逐一解决，为后续复杂的3D图形学实验奠定了坚实基础。

---
## 六、Git仓库链接

🔗 **https://github.com/char-math/CG-Lab-Work2**

---

**实验完成日期**：2026年3月24日

---
## 五、实验总结

本次实验成功实现了三维图形渲染管线的核心流程：

1. **理论验证**：通过代码实现了模型、视图、投影矩阵的数学推导
2. **实践应用**：成功在屏幕上渲染出可交互旋转的3D三角形
3. **性能优化**：利用Taichi框架实现GPU并行计算
4. **深入理解**：掌握了齐次坐标、矩阵变换、透视投影等核心概念

实验过程中遇到的坐标系统、矩阵运算、Taichi框架限制等问题，通过查阅文档和调试逐一解决，为后续复杂的3D图形学实验奠定了坚实基础。

---
## 六、Git仓库链接

🔗 **https://github.com/char-math/CG-Lab-Work2**

---

**实验完成日期**：2026年3月24日
