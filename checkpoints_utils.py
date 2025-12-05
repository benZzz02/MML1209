"""
Utilities to load checkpoints by inspecting what files were actually saved.
Behavior:
- If a file path is given, load it.
- If a directory is given, search for commonly-named checkpoint files saved by this project
  (e.g. model-highest.ckpt, model-highest-ema.ckpt, model-highest-regular.ckpt, model-highest-ema.ckpt)
  If none of the canonical names are found, pick the most recently modified .ckpt file.
- If directory contains method subfolders (cfg.val_methods), you can pass `method` to pick a specific
  method folder (e.g. /checkpoints/dir/<method>/).
- Loads model into trainer.model (handles DDP .module) and attempts to load trainer.ema if present.
- Optionally loads optimizer state if `optimizer` is provided and checkpoint contains it.

Example usage:
    from checkpoint_utils import load_checkpoint_auto
    # load best training checkpoint (search directory)
    load_checkpoint_auto(trainer, f"{cfg.checkpoint}/{dir}", method="pp_map", prefer_ema=False)
    # load a specific file
    load_checkpoint_auto(trainer, "/path/to/model-highest.ckpt")

"""
import os
import glob
import torch
from log import logger
from config import cfg
from checkpoint_utils import load_checkpoint_from_path as _existing_loader  # if this file co-exists, adjust import

# If this module is used standalone (no earlier checkpoint_utils.load_checkpoint_from_path),
# define a minimal loader function fallback. Otherwise the import above will resolve.
try:
    from checkpoint_utils import load_checkpoint_from_path as _existing_loader
except Exception:
    # Basic fallback - simple wrapper using torch.load and state_dict load logic
    def _existing_loader(trainer, ckpt_path, device=None, load_ema=True, optimizer=None, strict=True, verbose=True):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        if device is None:
            if torch.cuda.is_available():
                try:
                    map_location = {'cuda:%d' % 0: f'cuda:{cfg.gpu_id}'}
                except Exception:
                    map_location = None
            else:
                map_location = 'cpu'
        else:
            map_location = device
        if verbose:
            logger.info(f"Loading checkpoint {ckpt_path} -> map_location={map_location}")
        ckpt = torch.load(ckpt_path, map_location=map_location)
        # attempt to extract state_dict
        state_dict = None
        if isinstance(ckpt, dict):
            for k in ('model', 'state_dict', 'model_state', 'model_state_dict'):
                if k in ckpt:
                    state_dict = ckpt[k]; break
        if state_dict is None and isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        if state_dict is None:
            # nothing to load for model
            logger.warning("No model state dict found in checkpoint.")
            return {'model': False}
        model_target = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        try:
            model_target.load_state_dict(state_dict, strict=strict)
            logger.info("Loaded model state_dict.")
            loaded_model = True
        except Exception as e:
            logger.warning(f"Failed strict load: {e}. Trying non-strict / prefix fixes.")
            # try to strip module. prefix
            def _strip(sd):
                return {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}
            try:
                model_target.load_state_dict(_strip(state_dict), strict=False)
                loaded_model = True
                logger.info("Loaded model after stripping module. prefix (non-strict).")
            except Exception as e2:
                try:
                    model_target.load_state_dict({('module.' + k): v for k, v in state_dict.items()}, strict=False)
                    loaded_model = True
                    logger.info("Loaded model after adding module. prefix (non-strict).")
                except Exception as e3:
                    logger.error(f"Unable to load model: {e3}")
                    loaded_model = False
        result = {'model': loaded_model}
        # EMA loading naive attempt
        if load_ema and hasattr(trainer, 'ema'):
            ema_keys = ('ema', 'ema_state_dict', 'model_ema', 'model_ema_state_dict', 'state_dict_ema')
            ema_state = None
            if isinstance(ckpt, dict):
                for k in ema_keys:
                    if k in ckpt:
                        ema_state = ckpt[k]; break
            if ema_state is not None:
                ema_target = trainer.ema.module if hasattr(trainer.ema, "module") else trainer.ema
                try:
                    ema_target.load_state_dict(ema_state, strict=False)
                    result['ema'] = True
                except Exception as e:
                    logger.warning(f"Failed to load EMA: {e}")
                    result['ema'] = False
            else:
                result['ema'] = False
        return result


def _find_ckpt_in_dir(dir_path, method=None):
    """
    Search for canonical checkpoint filenames within dir_path (or method subdir).
    Returns the chosen checkpoint file path or None.
    Priority (if present) in the directory:
      1) model-highest.ckpt
      2) model-highest-regular.ckpt
      3) model-highest-ema.ckpt (separately loadable)
      4) any other *.ckpt file -> choose most recently modified
    If method is provided and dir contains a subdir with that name, search inside it first.
    """
    if method:
        candidate_dir = os.path.join(dir_path, method)
        if os.path.isdir(candidate_dir):
            dir_path = candidate_dir

    candidates = [
        "model-highest.ckpt",
        "model-highest-regular.ckpt",
        "model-highest-ema.ckpt",
        "model.ckpt",
        "checkpoint.ckpt"
    ]

    for name in candidates:
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            return p

    # search for any .ckpt files
    all_ckpts = glob.glob(os.path.join(dir_path, "*.ckpt"))
    if len(all_ckpts) > 0:
        # pick the most recently modified
        all_ckpts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return all_ckpts[0]

    # no checkpoint found
    return None


def load_checkpoint_auto(trainer,
                         path_or_dir,
                         method: str = None,
                         device: str = None,
                         prefer_ema: bool = False,
                         load_ema: bool = True,
                         optimizer: torch.optim.Optimizer = None,
                         strict: bool = True,
                         verbose: bool = True):
    """
    Load a checkpoint by inspecting given path_or_dir.
    - If path_or_dir is a file, load that file.
    - If path_or_dir is a directory, search for known checkpoint filenames or pick the newest .ckpt.
    - If prefer_ema is True and an ema-only checkpoint exists (model-highest-ema.ckpt), try to load that for EMA.
    Returns a dict reporting what was loaded (keys: model, ema, optimizer, meta).
    """
    if os.path.isfile(path_or_dir):
        # direct file, just load
        return _existing_loader(trainer, path_or_dir, device=device, load_ema=load_ema, optimizer=optimizer, strict=strict, verbose=verbose)

    # it's a directory: try to find a checkpoint
    ckpt_path = _find_ckpt_in_dir(path_or_dir, method=method)
    if ckpt_path is None:
        # maybe the dir contains an 'init' subdir (for init-phase ckpts)
        init_dir = os.path.join(path_or_dir, "init")
        if os.path.isdir(init_dir):
            ckpt_path = _find_ckpt_in_dir(init_dir, method=method)
            if ckpt_path is None:
                # still none: try looking for any ckpt inside subdirs (methods)
                subdirs = [os.path.join(path_or_dir, d) for d in os.listdir(path_or_dir) if os.path.isdir(os.path.join(path_or_dir, d))]
                latest = None
                for sd in subdirs:
                    c = _find_ckpt_in_dir(sd)
                    if c:
                        if latest is None or os.path.getmtime(c) > os.path.getmtime(latest):
                            latest = c
                if latest:
                    ckpt_path = latest

    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint (.ckpt) files found under {path_or_dir} (searched method={method})")

    if verbose:
        logger.info(f"Selected checkpoint to load: {ckpt_path}")

    # If prefer_ema is True and there exists a separate ema ckpt next to chosen file, try to load that for EMA specifically
    dir_of_ckpt = os.path.dirname(ckpt_path)
    ema_candidate = os.path.join(dir_of_ckpt, "model-highest-ema.ckpt")
    # Try to load main checkpoint first; loader will itself try to detect & load EMA if embedded.
    result = _existing_loader(trainer, ckpt_path, device=device, load_ema=load_ema, optimizer=optimizer, strict=strict, verbose=verbose)

    # If prefer_ema and ema file exists independently and wasn't loaded yet, attempt it
    if prefer_ema and load_ema and os.path.isfile(ema_candidate):
        # Only attempt to load ema file separately if result.get('ema') is False
        if not result.get('ema', False):
            if verbose:
                logger.info(f"Attempting to load separate EMA checkpoint: {ema_candidate}")
            try:
                _existing_loader(trainer, ema_candidate, device=device, load_ema=True, optimizer=None, strict=False, verbose=verbose)
                result['ema'] = True
            except Exception as e:
                logger.warning(f"Failed loading separate EMA ckpt: {e}")
                result['ema'] = result.get('ema', False)

    return result