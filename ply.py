import open3d as o3d

# 读取 .ply 文件
pcd = o3d.io.read_point_cloud("./point_cloud.ply")

# 可视化
o3d.visualization.draw_geometries([pcd])

# 读取 .ply 文件
pcd = o3d.io.read_point_cloud("./front_point_cloud.ply")

# 可视化
o3d.visualization.draw_geometries([pcd])

# 读取 .ply 文件
pcd = o3d.io.read_point_cloud("./right_shoulder_point_cloud.ply")

# 可视化
o3d.visualization.draw_geometries([pcd])