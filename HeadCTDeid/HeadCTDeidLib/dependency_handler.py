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

        def install_gdcm_and_restart_if_needed():
            try:
                import gdcm
            except ModuleNotFoundError as e:
                slicer.util.pip_install("python-gdcm")
                ok = slicer.util.confirmOkCancelDisplay("To support full encoding DICOM.\nSlicer needs to restart to complete the setup.", windowTitle="Restart Required")
                if ok:
                    restart_slicer()
                
        def restart_slicer():
            slicerExecutable = slicer.app.applicationFilePath()
            slicerHome = slicer.app.slicerHome
            subprocess.Popen([slicerExecutable], cwd=slicerHome)
            slicer.util.quit()
                       
        install_gdcm_and_restart_if_needed()
        logging.debug("Done")
