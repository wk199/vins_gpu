from distutils.core import setup, Extension
import os

module1 = Extension('kalmanfilter_pywrapper',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0'),
                                     ('PYMODULE_NAME', 'kalmanfilter_pywrapper')],
                    include_dirs = ['/usr/include'],
                    libraries = ['armadillo'],
                    library_dirs = ['/usr/lib'],
                    extra_compile_args = ['-std=c++11'],
                    sources = ['kalmanfilter.cpp', 
                    'kalmanfilter_pywrapper.cpp',
                    '../featuretracker/track2d.cpp',
                    '../featuretracker/conversions.cpp',
                    '../camposehandler/camposehandler.cpp',
                    '../triangulate/triangulate_est.cpp',
                    '../statehandler/statehandler.cpp',
                    '../util/util.cpp',
                    '../util/pywrapper.cpp'])

setup (name = 'KalmanFilter_PyWrapper',
       version = '1.0',
       description = 'Kalmanfilter package',
       long_description = 'Kalman Filter for Visual Odometry.',
       ext_modules = [module1])
