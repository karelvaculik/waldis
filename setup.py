#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    'pip>=9.0.1',
    'numpy>=1.12.0',
    'pandas>=0.19.2',
    'matplotlib>=2.0.0'
]

test_requirements = [
    'pip>=9.0.1',
    'numpy>=1.12.0',
    'pandas>=0.19.2',
    'matplotlib>=2.0.0'
]

setup(
    name='waldis',
    version='0.1.0',
    description="Dynamic Graph Miner",
    author="Karel Vaculík",
    author_email='vaculik.dev@gmail.com',
    url='https://github.com/karelvaculik/waldis',
    packages=[
        'waldis',
    ],
    package_dir={'waldis':
                 'waldis'},
    entry_points={
        'console_scripts': [
            'waldis=waldis.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='waldis',
    test_suite='tests',
    tests_require=test_requirements
)
