import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import random
import IPython
import pdb

e = IPython.embed
import cv2
from sklearn.model_selection import KFold

class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, arm_delay_time,
                 use_depth_image, use_robot_base):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids  # 1000
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_depth_image = use_depth_image
        self.arm_delay_time = arm_delay_time
        self.use_robot_base = use_robot_base
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        # 读取数据
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            original_action_shape = root['/action'].shape
            max_action_len = original_action_shape[0]  # max_episode
            if self.use_robot_base:
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            start_ts = np.random.choice(max_action_len)  # 随机抽取一个索引, 0~499
            actions = root['/observations/qpos'][1:] # (499, 14)
            # pdb.set_trace()
            actions = np.append(actions, actions[-1][np.newaxis, :], axis=0) # (500, 14)
            qpos = root['/observations/qpos'][start_ts]
            if self.use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][start_ts]), axis=0)
            image_dict = dict()
            image_depth_dict = dict()
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                    # print(image_dict[cam_name].shape)
                    # exit(-1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_ts]

            start_action = min(start_ts, max_action_len - 1) # 71
            index = max(0, start_action - self.arm_delay_time) # 71
            action = actions[index:]  # hack, to make timesteps more aligned # (429, 14)
            if self.use_robot_base:
                action = np.concatenate((action, root['/base_action'][index:]), axis=1)
            action_len = max_action_len - index  # hack, to make timesteps more aligned (429)

        self.is_sim = is_sim

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        action_is_pad = np.zeros(max_action_len)
        action_is_pad[action_len:] = 1
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        # pdb.set_trace()

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = []
            for cam_name in self.camera_names:
                all_cam_images_depth.append(image_depth_dict[cam_name])
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            # construct observations
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)
            image_depth_data = image_depth_data / 255.0

        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_data = torch.from_numpy(padded_action).float()
        action_is_pad = torch.from_numpy(action_is_pad).bool()
        action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # torch.set_printoptions(precision=10, sci_mode=False)
        # torch.set_printoptions(threshold=float('inf'))
        # print("qpos_data:", qpos_data[7:])
        # print("action_data:", action_data[:, 7:])

        return image_data, image_depth_data, qpos_data, action_data, action_is_pad


import os
import numpy as np
import torch
import h5py
import cv2

class EpisodicDataset_10_250(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, arm_delay_time,
                 use_depth_image, use_robot_base):
        super(EpisodicDataset_10_250).__init__()
        self.episode_ids = episode_ids  # 1000
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_depth_image = use_depth_image
        self.arm_delay_time = arm_delay_time
        self.use_robot_base = use_robot_base
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        # 读取数据
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            original_action_shape = root['/action'].shape
            max_action_len = original_action_shape[0]  # max_episode

            if self.use_robot_base:
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            # 限制start_ts的范围在10到249之间
            start_ts = np.random.choice(range(10, min(max_action_len, 250))) # 10~249

            # 读取11到250帧的数据
            actions = root['/observations/qpos'][11:250] # 11~249
            # pdb.set_trace()
            actions = np.append(actions, actions[-1][np.newaxis, :], axis=0) # 11~250

            qpos = root['/observations/qpos'][start_ts]
            if self.use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][start_ts]), axis=0)

            image_dict = dict()
            image_depth_dict = dict()
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_ts]

            start_action = min(start_ts, 250)
            index = max(10, start_action - self.arm_delay_time)
            action = actions[index : 251]  # 调整索引以适应10到250帧
            if self.use_robot_base:
                action = np.concatenate((action, root['/base_action'][index:]), axis=1)
            # print("index: ", index)
            # action_len = 250 - index + 10
            # pdb.set_trace()
            action_len = 250 - index - 10


        self.is_sim = is_sim

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        action_is_pad = np.zeros(250)
        action_is_pad[action_len:] = 1
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = []
            for cam_name in self.camera_names:
                all_cam_images_depth.append(image_depth_dict[cam_name])
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            # construct observations
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            image_depth_data = image_depth_data / 255.0

        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_data = torch.from_numpy(padded_action).float()
        action_is_pad = torch.from_numpy(action_is_pad).bool()
        action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, image_depth_data, qpos_data, action_data, action_is_pad


def get_norm_stats(dataset_dir, num_episodes, use_robot_base):
    all_qpos_data = []
    all_action_data = []

    for episode_idx in range(num_episodes):
        if episode_idx == 0 or episode_idx == 1 :
            continue
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            if use_robot_base:
                qpos = np.concatenate((qpos, root['/base_action'][()]), axis=1)
                action = np.concatenate((action, root['/base_action'][()]), axis=1)
            # 检查 qpos 和 action 的维度是否一致
            if action.shape[0] != 1000 and action.shape[1] == 14 :
                print(f"Skipping episode {episode_idx} due to inconsistent dimensions.")
                continue
            # if qpos.shape[0] != 140 and qpos.shape[1] == 7:
            #     print(f"Skipping episode {episode_idx} due to inconsistent dimensions.")
            #     continue
        print("qpos shape:", qpos.shape)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, arm_delay_time, use_depth_image,
              use_robot_base, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    # obtain train test split
    train_ratio = 0.92  # 数据集比例
    shuffled_indices = np.random.permutation(num_episodes)  # 打乱

    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action  返回均值和方差
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_robot_base)

    # construct dataset and dataloader 归一化处理  结构化处理数据
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                    use_depth_image, use_robot_base)

    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                  use_depth_image, use_robot_base)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def load_data_kfolder(dataset_dir, num_episodes, arm_delay_time, use_depth_image,
              use_robot_base, camera_names, batch_size_train, batch_size_val, n_folds=5):
    print(f'\nData from: {dataset_dir}\n')
    
    # obtain kFold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_dataloaders = []
    val_dataloaders = []
    norm_stats_list = []
    is_sim_list = []

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(range(num_episodes))):

        # obtain normalization stats for qpos and action  返回均值和方差
        norm_stats = get_norm_stats(dataset_dir, num_episodes, use_robot_base)

        # construct dataset and dataloader 归一化处理  结构化处理数据
        train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                        use_depth_image, use_robot_base)

        val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                    use_depth_image, use_robot_base)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                    num_workers=4, prefetch_factor=1)

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=4,
                                    prefetch_factor=1)
         # 添加到结果列表
        train_dataloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)
        norm_stats_list.append(norm_stats)
        is_sim_list.append(train_dataset.is_sim)

    return train_dataloaders, val_dataloaders, norm_stats_list, is_sim_list



# env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])

    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
