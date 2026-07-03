import open3d as o3d


ply_file_path = "output/tracking_3d_bboxes_20250320_164449.ply"
pcd = o3d.io.read_point_cloud(ply_file_path)


#o3d.visualization.draw_geometries([pcd])


vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(pcd)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("output/output_tracking_3d_bboxes_20250320_164449_ply.png")
vis.destroy_window()
