from data_provider.data_loader import Dataset_Craft
from torch.utils.data import DataLoader


data_dict = {
    "CRAFT": Dataset_Craft,
}


def data_provider(args, flag):
    Data = data_dict[args.model]

    if flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        # Important: ordinary shuffling can destroy hierarchical groups.
        shuffle_flag = False
        drop_last = flag == "train" # Drop the batch that is not close to the batch size in training
        batch_size = args.batch_size

    data_set = Data(
        args=args,
        root_path=args.root_path,
        flag=flag,
        size=[
            args.train_window,
            args.predict_window,
        ],
        data_path=args.data_path,
        behavior_data_path=args.behavior_data_path,
    )

    print(
        f"{flag}: {len(data_set)} samples, "
        f"{len(data_set) // batch_size} approximate batches"
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )

    return data_set, data_loader