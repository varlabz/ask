from setuptools import setup, find_packages

setup(
    name="ask",
    version="0.1.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic-ai[mcp]",
        "pyyaml"
    ],
    entry_points={
        "console_scripts": [
            "cli=cli_main:cli_main",
            "mcp=mcp_main:mcp_main"
        ]
    },
    python_requires=">=3.8",
)
