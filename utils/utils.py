'''
Miscellaneous tools / utilities / helper methods.
Created by Basile Van Hoorick, Feb 2022.
'''

from __init__ import *

# Library imports.
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


def cached_listdir(dir_path, allow_exts=[], recursive=False):
    '''
    Returns a list of all files if needed, and caches the result for efficiency.
    NOTE: Manual deletion of listdir.p is required if the directory ever changes.
    :param dir_path (str): Folder to gather file paths within.
    :param allow_exts (list of str): Only retain files matching these extensions.
    :param recursive (bool): Also include contents of all subdirectories within.
    :return (list of str): List of full image file paths.
    '''
    exts_str = '_'.join(allow_exts)
    recursive_str = 'rec' if recursive else ''
    cache_fp = f'{str(pathlib.Path(dir_path))}_{exts_str}_{recursive_str}_cld.p'

    if os.path.exists(cache_fp):
        # Cached result already available.
        print('Loading directory contents from ' + cache_fp + '...')
        with open(cache_fp, 'rb') as f:
            result = pickle.load(f)

    else:
        # No cached result available yet. This call can sometimes be very expensive.
        raw_listdir = os.listdir(dir_path)
        result = copy.deepcopy(raw_listdir)

        # Append root directory to get full paths.
        result = [os.path.join(dir_path, fn) for fn in result]

        # Filter by files only (no folders), not being own cache dump,
        # and belonging to allowed file extensions.
        result = [fp for fp in result if os.path.isfile(fp)]
        result = [fp for fp in result if not fp.endswith('_cld.p')]
        if allow_exts is not None and len(allow_exts) != 0:
            result = [fp for fp in result
                      if any([fp.lower().endswith('.' + ext) for ext in allow_exts])]

        # Recursively append contents of subdirectories within.
        if recursive:
            for dn in raw_listdir:
                dp = os.path.join(dir_path, dn)
                if os.path.isdir(dp):
                    result += cached_listdir(dp, allow_exts=allow_exts, recursive=True)

        print('Caching filtered directory contents to ' + cache_fp + '...')
        with open(cache_fp, 'wb') as f:
            pickle.dump(result, f)

    return result


# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# https://github.com/cvlab-columbia/GREATER/blob/main/render_videos.py
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
