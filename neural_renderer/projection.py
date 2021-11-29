from __future__ import division

import torch
import numpy as np
from .look import look


def vis(vertices):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    points = vertices[0, :, :3].detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()


def projection_graphics(vertices, P, R, t, eps=1e-9):
    '''
    Calculate projective transformation of vertices given a graphics projection matrix
    Input parameters:
    P: batch_size * 4 * 4 graphics projection matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # world space to camera space
    eye = t.squeeze()
    at = R[0, :3, 2]
    up = R[0, :3, 1]
    #vertices = look(vertices, eye, at, up)  # for scannet
    vertices = torch.matmul(vertices, R.transpose(2,1)) + t   # for matterport

    vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, :1])], dim=-1)  # add homogenous component
    vertices = torch.bmm(vertices, P.transpose(1, 2))  # view to clip space
    w = vertices[:, :, 3:]
    vertices = vertices / (w + eps)  # perspective division

    return vertices[:, :, :3]


def projection(vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    if K.shape[1] == 4 and K.shape[2] == 4:
        # if we have given the graphics projection matrix
        return projection_graphics(vertices, K, R, t)

    # instead of P*x we compute x'*P'
    vertices = torch.matmul(vertices, R.transpose(2,1)) + t
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_

    vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
    vertices = torch.matmul(vertices, K.transpose(1,2))
    u, v = vertices[:, :, 0], vertices[:, :, 1]

    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size

    vertices = torch.stack([u, v, z], dim=-1)

    return vertices
