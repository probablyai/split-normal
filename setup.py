import runpy
import git
import re
import setuptools


__version__ = runpy.run_path("split_normal/version.py")["__version__"]


def get_readme(filename="README.md"):
    with open(filename, "r") as f:
        readme = f.read()
    return readme


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
    readme = get_readme()
    repo_info = get_repo_info()
    content_url = assemble_github_content_url(**repo_info)
    result = re.sub(r"(src=\")(tex/[a-z0-9]*\.svg.*\")", rf"\1{content_url}/\2", readme)
    return result


setuptools.setup(
    name="split-normal",
    version=__version__,
    author="Tárik S. Salem",
    description="A tiny package implementing functions of the split normal distribution compatible with Numpy and JAX.",
    long_description=get_readme_with_github_urls(),
    long_description_content_type="text/markdown",
    url="https://github.com/probablyai/split-normal",
    packages=setuptools.find_packages(exclude=("tests", )),
    install_requires=["numpy>=1.17.3", "scipy>=1.4.0", "jax>=0.1.59", "jaxlib>=0.1.40"],
    extras_require={
        "dev": ["pytest>=5.4.2", "readme2tex>=0.0.1b2", "GitPython>=3.1.3"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development"
    ],
    python_requires=">=3.6",
)
