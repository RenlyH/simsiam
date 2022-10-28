
#def build_opensrh(cf='opensrh_data.yaml'):
#    import yaml 
#    from opensrh.datasets.srh_dataset import SRHClassificationDataset
#    from opensrh.datasets.improc import get_srh_aug_list
#    cf = yaml.load(open(cf), Loader=yaml.FullLoader)
#    train_dset = SRHClassificationDataset(
#        data_root=cf["data"]["db_root"],
#        studies="train",
#        transform=Compose(
#            get_srh_aug_list(cf["data"]["train_augmentation"],
#                             cf["data"]["rand_aug_prob"])),
#        balance_patch_per_class=cf["data"]["balance_patch_per_class"])
#    val_dset = SRHClassificationDataset(
#        data_root=cf["data"]["db_root"],
#        studies="val",
#        transform=Compose(
#            get_srh_aug_list(cf["data"]["valid_augmentation"],
#                             cf["data"]["rand_aug_prob"])),
#        balance_patch_per_class=False)
#    return train_dset, val_dset

def build_opensrh(cf='opensrh_data.yaml'):
    import yaml
    from torchvision.transforms import Compose
    from opensrh.datasets.srh_dataset import SRHClassificationDataset, SRHContrastiveDataset
    from opensrh.datasets.improc import get_srh_aug_list
    cf = yaml.load(open(cf), Loader=yaml.FullLoader)
    train_dset = SRHContrastiveDataset(
        data_root=cf["data"]["db_root"],
        studies="train",
        transform=Compose(
            get_srh_aug_list(cf["data"]["train_augmentation"],
                             cf["data"]["rand_aug_prob"])),
        balance_patch_per_class=cf["data"]["balance_patch_per_class"])
    val_dset = SRHContrastiveDataset(
        data_root=cf["data"]["db_root"],
        studies="val",
        transform=Compose(
            get_srh_aug_list(cf["data"]["valid_augmentation"],
                             cf["data"]["rand_aug_prob"])),
        balance_patch_per_class=False)
    return train_dset, val_dset

if __name__ == "__main__":
    train_dataset, valid_dataset = build_opensrh("../opensrh_data.yaml")
    import pdb
    pdb.set_trace()
