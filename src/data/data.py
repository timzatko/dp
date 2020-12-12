import os


def get_path_to_saliencies_and_segmentations_factory(root_path):
    def get_path_to_saliencies_and_segmentations(sub_path):
        return os.path.join(root_path, sub_path)

    return get_path_to_saliencies_and_segmentations
