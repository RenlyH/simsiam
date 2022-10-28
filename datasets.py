
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
def build_srh7(config='srh7_data.yaml'):
    import yaml
    from torchsrh.datasets.db_improc import (get_transformations, process_read_png,
                                             get_srh_base_aug)
    from torchsrh.train.patch_data_utils import get_patch_contrastive_datasets
    from torchsrh.datasets import (PatchDataset, PatchContrastiveDataset)
    config = yaml.load(open(config), Loader=yaml.FullLoader)

    train_xform, valid_xform = get_transformations(config, get_srh_base_aug)
    train_dset, valid_dset = get_patch_contrastive_datasets(
        config, train_xform, valid_xform, PatchContrastiveDataset)
    return train_dset, valid_dset
    

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
    train_dataset, valid_dataset = build_srh7()
    import pdb
    pdb.set_trace()
