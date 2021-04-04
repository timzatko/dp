import os


def load_checkpoint(model, checkpoint_dir, model_key, ckpt_name, options=None, by_name=False, skip_mismatch=False):
    checkpoint_path = os.path.join(checkpoint_dir, model_key, ckpt_name)
    print(f'loading checkpoint from {checkpoint_path}...')
    model.load_weights(checkpoint_path, options=options, by_name=by_name, skip_mismatch=skip_mismatch)
