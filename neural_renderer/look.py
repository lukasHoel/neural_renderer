import numpy as np
import torch
import torch.nn.functional as F


def vis(vertices):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    points = vertices[0].detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.visualization.draw_geometries([pcd])

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()


def look(vertices, eye, direction=[0, 1, 0], up=None):
    """
    "Look" transformation of vertices.
    """
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')

    device = vertices.device

    if isinstance(direction, list) or isinstance(direction, tuple):
        direction = torch.tensor(direction, dtype=torch.float32, device=device)
    elif isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction).to(device)
    elif torch.is_tensor(direction):
        direction = direction.to(device)

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
    elif isinstance(eye, np.ndarray):
        eye = torch.from_numpy(eye).to(device)
    elif torch.is_tensor(eye):
        eye = eye.to(device)

    if isinstance(up, list) or isinstance(up, tuple):
        up = torch.tensor(up, dtype=torch.float32, device=device)
    elif isinstance(up, np.ndarray):
        up = torch.from_numpy(up).to(device)
    elif torch.is_tensor(up):
        up.to(device)

    if up is None:
        up = torch.cuda.FloatTensor([0, 1, 0])
    if eye.ndimension() == 1:
        eye = eye[None, :]
    if direction.ndimension() == 1:
        direction = direction[None, :]
    if up.ndimension() == 1:
        up = up[None, :]

    # create new axes
    z_axis = F.normalize(direction, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5)
    z_axis = -z_axis

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :]
    #vis(vertices)
    vertices = vertices - eye
    vertices = torch.matmul(vertices, r.transpose(1,2))
    #vis(vertices)

    return vertices
