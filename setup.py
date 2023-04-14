from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.7',
    'License :: OSI Approved :: MIT License',
    "Operating System :: OS Independent"
]

setup(
     name='tigermyo',
     
     version='0.1.0',
     description='Processing MRI images based on deep-learning',
     long_description_content_type='text/markdown',
     url='https://github.com/htylab/tigermyo',
     
     author='Biomedical Imaging Lab, Taiwan Tech',
     author_email='',
     License='MIT',
     classifiers=classifiers,
     
     keywords='MRI segmentation',
     packages=find_packages(),
     entry_points={
        'console_scripts': [
            'tigermyo = tigermyo.myo:run',
        ]
    },
     python_requires='>=3.8',
     install_requires=[
             'numpy>=1.22.3',
             'matplotlib>=3.5.2',
             'scikit-image>=0.19.2',
             'onnxruntime>=1.11.1',
             'SimpleITK>=2.2.0',
             'tqdm>=4.64.0',
             'pydicom>=2.3.0',
             'scipy>=1.6.0'
         ]
)