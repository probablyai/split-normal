import git
import re

from setup import read_readme


def write_readme(readme, filename="README.md"):
    with open(filename, "w") as f:
        f.write(readme)


def get_repo_info():
    info = dict()
    repo = git.Repo(".git")
    info['url'] = repo.remotes.origin.url
    m = re.match(r"git@github.com:(.*)/(.*).git", info['url'])
    info['namespace'] = m.group(1)
    info['project'] = m.group(2)
    info['commit'] = repo.head.commit.hexsha
    info['branch'] = None
    info['tag'] = None
    if repo.head.is_detached:
        info['tag'] = next((tag.name for tag in repo.tags if tag.commit == repo.head.commit))
    else:
        info['branch'] = repo.active_branch.name
    return info


def assemble_github_content_url(namespace, project, branch, tag, commit, **kwargs):
    version = branch or tag or commit
    content_url = f"https://raw.githubusercontent.com/{namespace}/{project}/{version}"
    return content_url


def get_readme_with_github_urls():
    readme = read_readme()
    repo_info = get_repo_info()
    content_url = assemble_github_content_url(**repo_info)
    result = re.sub(r"(src=\")(tex/[a-z0-9]*\.svg.*\")", rf"\1{content_url}/\2", readme)
    return result


if __name__ == '__main__':
    write_readme(get_readme_with_github_urls())
