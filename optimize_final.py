import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import soft_renderer as sr
import torch.optim as optim
import torch.nn.functional as F
from utils.loss_utils import OffsetNet
from torch.utils.data import DataLoader
from lietorch import SO3, SE3, LieGroupParameter
from utils.data_utils import read_obj, Optimization_data
from utils.train_utils import get_scale_init, forward_kinematic
from ChamferDistance.chamfer3D.dist_chamfer_3D import chamfer_3DDist


class FinalTrainer:
    def __init__(self, config):
        self.config = config
        self.test_animal = config.data.test_animal
        self.retrieval_animal = config.data.retrieval_animal

        self.info_dir = config.data.info_dir
        self.mesh_dir = config.data.mesh_dir

        self.out_path = os.path.join(config.data.out_dir, self.test_animal)
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(os.path.join(self.out_path, 'temparray'), exist_ok=True)

        self.raw_info_dir = os.path.join(self.info_dir, self.test_animal)
        self.retrieve_info_dir = os.path.join(self.info_dir, self.retrieval_animal)
        self.retrieve_mesh_dir = os.path.join(self.mesh_dir, self.retrieval_animal)

    @staticmethod
    def load_skeleton(retrieve_info_dir, retrieval_animal):
        # load skeleton info + align coordinate
        ske = np.load(os.path.join(retrieve_info_dir, 'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()[
            'frame_000001']
        for key in ske.keys():
            head, tail = ske[key]['head'], ske[key]['tail']
            head[[1]] *= -1
            tail[[1]] *= -1
            head, tail = head[[0, 2, 1]], tail[[0, 2, 1]]
            ske[key]['head'] = head
            ske[key]['tail'] = tail
        with open(os.path.join(retrieve_info_dir, 'weight', '{}.json'.format(retrieval_animal)), 'r',
                  encoding='utf8') as fp:
            json_data = json.load(fp)
        return ske, json_data

    def save_utils(self, p1, p2, epoch_id, W1, basic_mesh, x, bones_len_scale, mesh_scale, shifting, ske,
                   json_data, offset_net, zeros_tensor, faces):
        SO3_R = SO3.InitFromVec(p1)
        SE3_T = SE3.InitFromVec(p2)
        temp_out_path = os.path.join(self.out_path, "Epoch_{}".format(epoch_id))
        os.makedirs(temp_out_path, exist_ok=True)

        os.makedirs(os.path.join(temp_out_path, 'temparray'), exist_ok=True)

        W1_numpy = W1.cpu().detach().numpy()
        old_mesh_numpy = basic_mesh.cpu().detach().numpy()
        p1_numpy = p1.cpu().detach().numpy()
        p2_numpy = p2.cpu().detach().numpy()
        x_numpy = x.cpu().detach().numpy()
        bones_len_scale_numpy = bones_len_scale.cpu().detach().numpy()
        mesh_scale_numpy = mesh_scale.cpu().detach().numpy()
        shifting_numpy = shifting.cpu().detach().numpy()

        np.save(os.path.join(temp_out_path, 'temparray', 'W1'), W1_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'old_mesh'), old_mesh_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'p1'), p1_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'p2'), p2_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'x'), x_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'bones_len_scale'), bones_len_scale_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'mesh_scale'), mesh_scale_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'shifting'), shifting_numpy)

        x = basic_mesh.clone()  # for optimization w/o basic shape offset
        offset = offset_net(x[:, :-1].cuda())
        homo_offset = torch.cat([offset, zeros_tensor], dim=1)  # for optimization w/ basic shape offset
        x = x + homo_offset  # for optimization w/ basic shape offset
        offset_numpy = offset.clone().detach().cpu().numpy()
        offset_x_numpy = x.clone().detach().cpu().numpy()
        np.save(os.path.join(temp_out_path, 'temparray', 'offset_x'), offset_x_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'offset'), offset_numpy)
        bb_center = (x.max(0)[0] + x.min(0)[0]) / 2
        x[:, :-1] *= mesh_scale
        bb_new_center = (x.max(0)[0] + x.min(0)[0]) / 2
        x += bb_center - bb_new_center
        x[:, :-1] += shifting
        x_beforetran_numpy = x.cpu().detach().numpy()
        np.save(os.path.join(temp_out_path, 'temparray', 'x_beforetran'), x_beforetran_numpy)
        ske_shift = (bb_center - bb_new_center)[:-1] + shifting[0] + offset.mean(0)

        wbx_results, wbx_no_root = forward_kinematic(x[:, :-1].clone(), W1.clone(), bones_len_scale, SO3_R,
                                                     SE3_T, ske, json_data, mesh_scale,
                                                     ske_shift.detach().cpu().numpy()
                                                     )
        wbx_numpy = wbx_results.cpu().detach().numpy()
        wbx_no_root_numpy = wbx_no_root.cpu().detach().numpy()
        np.save(os.path.join(temp_out_path, 'temparray', 'wbx'), wbx_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'wbx_noroot'), wbx_no_root_numpy)
        np.save(os.path.join(temp_out_path, 'temparray', 'ske'), ske)
        faces_final = faces[None].clone()

        for ind, vertices in enumerate(wbx_results):
            sr.Mesh(vertices, faces_final).save_obj(os.path.join(temp_out_path, 'Frame{:02d}.obj'.format(ind + 1)))

    def calculate_loss(self, mesh1, epoch_id, iter_id, verts, offset_x, SO3_R, SE3_T, mask, flow, faces):

        rendering_mask = self.renderer_soft.render_mesh(mesh1)
        rendering_mask = rendering_mask[:, -1]
        loss_mask = F.mse_loss(mask, rendering_mask)
        if not epoch_id > 60:
            loss = loss_mask * self.w_mask
            if epoch_id % 10 == 1:
                print("Loss for epoch {}, iter {} : mask: {:.2f}".format(epoch_id, iter_id,
                                                                         loss_mask.item() * self.w_mask))
            return loss

        if epoch_id % 5 == 0:
            mask_temp_path = os.path.join(self.out_path, 'Mask', "Epoch_{}".format(epoch_id))
            os.makedirs(mask_temp_path, exist_ok=True)
            for mask_ind in range(mask.shape[0]):
                Image.fromarray((rendering_mask[mask_ind].detach().cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(mask_temp_path, 'rendering_mask{}.jpg'.format(mask_ind)))
                Image.fromarray((mask[mask_ind].detach().cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(mask_temp_path, 'gt_mask{}.jpg'.format(mask_ind)))

        mesh_flow = sr.Mesh(
            verts[:-1], faces.repeat(int(self.config.data.batch_size) - 1, 1, 1), textures=verts[1:],
            texture_type="vertex"
        )

        rendering_uv = self.renderer_softtex.render_mesh(mesh_flow)

        rendering_grid = torch.Tensor(
            np.meshgrid(range(rendering_mask.shape[2]), range(rendering_mask.shape[1])), device="cuda")
        rendering_grid[0] = rendering_grid[0] * 2 / (rendering_mask.shape[2]) - 1
        rendering_grid[1] = 1 - rendering_grid[1] * 2 / (rendering_mask.shape[2])
        rendering_grid = rendering_grid[None].repeat(rendering_uv.shape[0], 1, 1, 1)
        rendering_flow = rendering_uv[:, :2] - rendering_grid
        rendering_flow[:, 1] *= -1
        flow_mask = (rendering_mask[:-1, None].clone().detach().bool() | mask[:-1, None].bool())

        loss_flow = F.mse_loss(rendering_flow * flow_mask, flow[:-1] * flow_mask)

        gt_bone = torch.zeros((SO3_R.shape[0] - 1, SO3_R.shape[1], 4), device="cuda")
        gt_bone[:, :, -1] += 1
        gt_bone2 = torch.zeros((SE3_T.shape[0] - 1, SE3_T.shape[1], 4), device="cuda")
        gt_bone2[:, :, -1] += 1

        loss_bone = F.mse_loss(SO3_R[:-1].inv().mul(SO3_R[1:]).vec(), gt_bone)
        loss_bone3 = F.mse_loss(SE3_T[:-1].inv().mul(SE3_T[1:]).vec()[:, :, 3:], gt_bone2)

        gt_velo = torch.zeros((SO3_R.shape[0] - 2, SO3_R.shape[1], 4), device="cuda")
        gt_velo[:, :, -1] += 1
        gt_velo2 = torch.zeros((SE3_T.shape[0] - 2, SE3_T.shape[1], 4), device="cuda")
        gt_velo2[:, :, -1] += 1

        velo_so3 = SO3_R[:-1].inv().mul(SO3_R[1:])
        velo_se3 = SE3_T[:-1].inv().mul(SE3_T[1:])

        loss_bone_velo = F.mse_loss(velo_so3[:-1].inv().mul(velo_so3[1:]).vec(), gt_velo)
        loss_bone_velo2 = F.mse_loss(velo_se3[:-1].inv().mul(velo_se3[1:]).vec()[:, :, 3:], gt_velo2)

        loss_consist = F.mse_loss(mask[1:], mask[:-1])

        cham_loss = chamfer_3DDist()
        x_reverse = offset_x.clone().detach()
        x_reverse[:, 0] *= -1
        loss_chamfer = cham_loss(offset_x[None, :, :-1], x_reverse[None, :, :-1])[0].mean()

        loss = loss_mask * self.w_mask + loss_flow * self.w_flow + \
               (loss_bone + loss_bone3) * self.w_smooth + loss_chamfer * self.w_symm + \
               (loss_bone_velo + loss_bone_velo2) * self.w_velo + loss_consist * self.w_consist

        if epoch_id % 10 == 1:
            print(
                "Loss for epoch {}, iter {} : mask: {:.2f}, flow: {:.2f},  bone: {:.2f}, chamfer: {:.2f}".format(
                    epoch_id, iter_id, loss_mask.item() * self.w_mask,
                                       loss_flow.item() * self.w_flow, (loss_bone + loss_bone3) * self.w_smooth,
                                       loss_chamfer * self.w_symm))

        return loss

    def diff_render(self, wbx, intrin, extrin, faces):
        ones_tensor = torch.ones_like(wbx[:, :, [0]])
        wbx = torch.cat([wbx, ones_tensor], dim=2)
        wbx0 = wbx[:, :, [0, 2, 1, 3]]
        wbx0[:, :, 1] *= -1

        verts = torch.matmul(intrin, torch.matmul(extrin, wbx0.permute(0, 2, 1))).permute(0, 2, 1)
        depth_z = verts[:, :, [2]]
        verts = verts / depth_z
        verts = (verts - 512) / 512
        verts[:, :, 1] *= -1

        mesh1 = sr.Mesh(verts, faces.repeat(self.config.data.batch_size, 1, 1))

        return verts, mesh1

    def basic_transform(self, basic_mesh, offset_net, zeros_tensor, mesh_scale, shifting):

        x = basic_mesh.clone()
        offset = offset_net(x[:, :-1].cuda())
        homo_offset = torch.cat([offset, zeros_tensor], dim=1)
        x = x + homo_offset
        offset_x = x.clone()
        bb_center = (x.max(0)[0] + x.min(0)[0]) / 2
        x[:, :-1] *= mesh_scale.clamp(0.15, 8).item()
        bb_new_center = (x.max(0)[0] + x.min(0)[0]) / 2
        x += bb_center - bb_new_center
        x[:, :-1] += shifting
        ske_shift = (bb_center - bb_new_center)[:-1] + shifting[0] + offset.mean(0)

        return x, ske_shift, offset_x

    def init_training(self):

        self.w_mask, self.w_flow, self.w_smooth, self.w_symm, self.w_velo, self.w_consist = int(
            self.config.model.w_mask), int(self.config.model.w_flow), \
            int(self.config.model.w_smooth), int(self.config.model.w_symm), int(self.config.model.w_velo), int(
            self.config.model.w_consist)

        self.renderer_soft = sr.SoftRenderer(image_size=1024, sigma_val=1e-5,
                                             camera_mode='look_at', perspective=False, aggr_func_rgb='hard',
                                             light_mode='vertex', light_intensity_ambient=1.,
                                             light_intensity_directionals=0.)
        self.renderer_softtex = sr.SoftRenderer(image_size=1024, sigma_val=1e-4, gamma_val=1e-2,
                                                camera_mode='look_at', perspective=False, aggr_func_rgb='softmax',
                                                light_mode='vertex', light_intensity_ambient=1.,
                                                light_intensity_directionals=0.)

        self.start_idx, self.end_idx = self.config.data.start_idx, self.config.data.end_idx

        points_info, normals_info, face_info = read_obj(os.path.join(self.retrieve_mesh_dir, 'remesh.obj'))

        faces = torch.tensor(face_info, device="cuda")

        print("Mesh point number is ", points_info.shape[0])
        points_info = np.concatenate((points_info, np.ones((points_info.shape[0], 1))), axis=1)

        W = torch.tensor(np.load(os.path.join(self.retrieve_mesh_dir, 'W1.npy')), requires_grad=True, device="cuda")
        N, Frames, B = points_info.shape[0], self.end_idx - self.start_idx, W.shape[1]

        basic_mesh = torch.tensor(points_info, dtype=torch.float32, device="cuda").detach()
        p1 = torch.randn((Frames, B, 4), requires_grad=True, device="cuda")
        p2 = torch.randn((Frames, 1, 7), requires_grad=True, device="cuda")
        p1_init = np.array([0., 0., 0., 1.])
        with torch.no_grad():
            for f in range(Frames):
                for b in range(B):
                    p1[f, b] = torch.tensor(p1_init)

            for f in range(Frames):
                p2[f, -1][:3] = torch.tensor([0, 0, 0.01 * f], dtype=torch.float64)
                p2[f, -1][3:] = torch.tensor(p1_init)

        offset = torch.zeros((basic_mesh.shape[0], 3), requires_grad=True, device="cuda")
        offset_net = OffsetNet().cuda()
        zeros_tensor = torch.zeros_like(offset[:, [-1]], requires_grad=False, device="cuda")

        bones_len_scale = torch.ones((B, 1), requires_grad=True, device="cuda")
        shifting = torch.zeros((1, 3), requires_grad=True, device="cuda")
        mesh_scale = torch.ones((1), requires_grad=True, device="cuda")

        return p1, p2, basic_mesh, W, faces, offset_net, zeros_tensor, bones_len_scale, shifting, mesh_scale

    def optimize(self, ):

        p1, p2, basic_mesh, W, faces, offset_net, zeros_tensor, bones_len_scale, shifting, mesh_scale = self.init_training()

        train_loader = DataLoader(
            Optimization_data(self.config), batch_size=int(self.config.data.batch_size), drop_last=True, shuffle=False,
            num_workers=0
        )
        print("Dataloader Length: ", len(train_loader))

        optimizer = optim.Adam([mesh_scale], lr=1e-3)  # for optimization w/ basic shape offset
        optimizer.add_param_group({"params": shifting, 'lr': 5e-2})

        epoch_num = 202
        flag = 0

        for epoch_id in range(epoch_num):
            if epoch_id <= 60 and epoch_id % 15 == 14:
                for params in optimizer.param_groups:
                    params['lr'] *= 0.5

            if epoch_id > 60 and flag == 0:
                optimizer = optim.Adam([p1, p2], lr=4e-3)
                optimizer.add_param_group({"params": [mesh_scale], 'lr': 1e-4})
                flag = 1

            if epoch_id in [100, 140, 175]:
                for params in optimizer.param_groups:
                    params['lr'] *= 0.85

            if epoch_id % 10 == 1:
                print("MESH SCALE: ", mesh_scale.clamp(0.15, 8).item())

            for iter_id, data in enumerate(train_loader):
                intrin, extrin, mask, flow, index, color = data
                intrin, extrin, mask, flow, color = intrin.cuda(), extrin.cuda(), mask.cuda(), flow.permute(0, 3, 1,
                                                                                                            2).cuda(), color.cuda()

                if epoch_id == 0 and iter_id == 0:
                    with torch.no_grad():
                        init_scale, shift = get_scale_init(
                            basic_mesh, faces, intrin, extrin, mask, self.renderer_soft
                        )
                        mesh_scale *= init_scale

                SO3_R = SO3.InitFromVec(p1)
                SE3_T = SE3.InitFromVec(p2)

                optimizer.zero_grad()

                x, ske_shift, offset_x = self.basic_transform(basic_mesh, offset_net, zeros_tensor, mesh_scale,
                                                              shifting)

                W1 = F.softmax(W * 10)
                W1 = (W1 / (W1.sum(1, keepdim=True).detach()))

                ske, json_data = self.load_skeleton(self.retrieve_info_dir, self.retrieval_animal)

                wbx, _ = forward_kinematic(
                    x[:, :-1].clone(), W1.clone(), bones_len_scale, SO3_R[index[0]:index[-1] + 1],
                    SE3_T[index[0]:index[-1] + 1], ske, json_data, mesh_scale,
                    ske_shift.detach().cpu().numpy()
                )

                verts, mesh1 = self.diff_render(wbx, intrin, extrin, faces)

                loss = self.calculate_loss(mesh1, epoch_id, iter_id, verts, offset_x, SO3_R, SE3_T, mask, flow, faces)

                sys.stdout.flush()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                if epoch_id % 20 == 1:
                    self.save_utils(p1, p2, epoch_id, W1, basic_mesh, x, bones_len_scale, mesh_scale, shifting,
                                    ske, json_data, offset_net, zeros_tensor, faces)
