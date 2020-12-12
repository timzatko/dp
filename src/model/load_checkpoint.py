import os


def load_checkpoint(model, checkpoint_dir, model_key, ckpt_name, options=None):
    checkpoint_path = os.path.join(checkpoint_dir, model_key, ckpt_name)
    print(f'loading checkpoint from {checkpoint_path}...')
    model.load_weights(checkpoint_path, options=options)
