version = '1.0.0'
repo = 'unknown'
commit = 'unknown'
has_repo = False

try:
    import git
    from pathlib import Path

    try:
        r = git.Repo(Path(__file__).parents[1])
        has_repo = True

        if not r.remotes:
            repo = 'local'
        else:
            repo = r.remotes.origin.url

        commit = r.head.commit.hexsha
        if r.is_dirty():
            commit += ' (dirty)'
    except git.InvalidGitRepositoryError:
        raise ImportError()
except ImportError:
    pass

try:
    from . import _dist_info as info
    assert not has_repo, '_dist_info should not exist when repo is in place'
    assert version == info.version
    repo = info.repo
    commit = info.commit
except ImportError:
    pass

__all__ = ['version', 'repo', 'commit', 'has_repo']
