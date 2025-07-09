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
        
        logging.debug("Initializing openpyxl...")
        packageName = "openpyxl"
        if not self._checkModuleInstalled(packageName):
          logging.debug("openpyxl package is required. Installing... (it may take several minutes)")
          install(packageName)
          if not self._checkModuleInstalled(packageName):
            raise ValueError("openpyxl needs to be installed to use this module.")
        else:  
            from packaging import version
            import openpyxl
                
        logging.debug("Initializing python-gdcm...")
        packageName = "gdcm"
        if not self._checkModuleInstalled(packageName):
          logging.debug("python-gdcm package is required. Installing... (it may take several minutes)")
          install('python-gdcm')
          if not self._checkModuleInstalled(packageName):
            raise ValueError("python-gdcm needs to be installed to use this module.")
            
        logging.debug("Initializing pylibjpeg...")
        packageName = "pylibjpeg"
        if not self._checkModuleInstalled(packageName):
          logging.debug("pylibjpeg package is required. Installing... (it may take several minutes)")
          install(packageName)
          if not self._checkModuleInstalled(packageName):
            raise ValueError("pylibjpeg needs to be installed to use this module.")
            
   
        logging.debug("Initializing pylibjpeg-libjpeg...")
        packageName = "pylibjpeg-libjpeg"
        if not self._checkModuleInstalled(packageName):
          logging.debug("pylibjpeg-libjpeg package is required. Installing... (it may take several minutes)")
          install(packageName)
    
    
        logging.debug("Initializing  pylibjpeg-openjpeg ...")
        packageName = "pylibjpeg-openjpeg"
        if not self._checkModuleInstalled(packageName):
          logging.debug("pylibjpeg-openjpeg package is required. Installing... (it may take several minutes)")
          install(packageName)
    
    
        logging.debug("Initializing pydicom ...")
        packageName = "pydicom"
        if not self._checkModuleInstalled(packageName):
          logging.debug("pydicom package is required. Installing... (it may take several minutes)")
          install(packageName)
          if not self._checkModuleInstalled(packageName):
            raise ValueError("pydicom needs to be installed to use this module.")
