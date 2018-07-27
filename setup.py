from setuptools import setup, find_packages

setup(name='kerasLSTM', version='0.1', packages=find_packages(), description='run keras on gcloud', author='Will Kamovitch',install_requires=['keras', 'h5py'], zip_safe=False)