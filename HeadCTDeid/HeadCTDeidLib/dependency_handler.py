import shutil
import subprocess
import logging
import sys
import slicer

from abc import ABC, abstractmethod


class DependenciesBase(ABC):

    minimumTorchVersion = "1.12"

    def __init__(self):
        self.dependenciesInstalled = False  # we don't know yet if dependencies have been installed

    @abstractmethod
    def setupPythonRequirements(self, upgrade=False):
        pass


class NonSlicerPythonDependencies(DependenciesBase):
    def _checkModuleInstalled(self, moduleName):
      try:
        import importlib
        importlib.import_module(moduleName)
        return True
      except ModuleNotFoundError:
        return False

    def setupPythonRequirements(self, upgrade=False):
        def install(package):
          #subprocess.check_call([sys.executable, "-m", "pip", "install", package])
          slicer.util.pip_install(package)
            
        packageName = "openpyxl"
        if not self._checkModuleInstalled(packageName):
          install(packageName)
          if not self._checkModuleInstalled(packageName):
            raise ValueError("openpyxl needs to be installed to use this module.")
        else:  
            from packaging import version
            import openpyxl
                
        packageName = "gdcm"
        if not self._checkModuleInstalled(packageName):
          install('python-gdcm')
          if not self._checkModuleInstalled(packageName):
            raise ValueError("python-gdcm needs to be installed to use this module.")
            
        packageName = "pylibjpeg"
        if not self._checkModuleInstalled(packageName):
          install(packageName)
          if not self._checkModuleInstalled(packageName):
            raise ValueError("pylibjpeg needs to be installed to use this module.")
            
   
        packageName = "pylibjpeg-libjpeg"
        if not self._checkModuleInstalled(packageName):
          install(packageName)
    
    
        packageName = "pylibjpeg-openjpeg"
        if not self._checkModuleInstalled(packageName):
          install(packageName)
    
    
        packageName = "pydicom"
        if not self._checkModuleInstalled(packageName):
          install(packageName)
          if not self._checkModuleInstalled(packageName):
            raise ValueError("pydicom needs to be installed to use this module.")

        self.dependenciesInstalled = True