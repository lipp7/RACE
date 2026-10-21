"""Shared deterministic ten-view datasets."""
from maft.data import CUBImages
from scripts.extract_pets import PetViews
from scripts.extract_flowers import FlowerViews
from scripts.extract_transport import TransportViews

def dataset(name, split, preprocess):
    if name == 'cub':
        return CUBImages('data/CUB_200_2011', split, preprocess, view_grid=3)
    if name == 'pets':
        return PetViews('data', 'trainval' if split == 'train' else 'test', preprocess)
    if name == 'flowers':
        return FlowerViews('data', ('train', 'val') if split == 'train' else ('test',), preprocess)
    return TransportViews('data/stanford-cars-parquet' if name == 'cars' else 'data', name,
                          'trainval' if name == 'aircraft' and split == 'train' else split, preprocess)

if __name__ == '__main__':
    from race.extract import main
    main()
