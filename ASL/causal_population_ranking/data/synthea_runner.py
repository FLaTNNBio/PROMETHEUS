from __future__ import annotations
import json,shutil,subprocess
from pathlib import Path


class SyntheaRunner:
    def __init__(self,path="vendor/synthea"): self.path=Path(path)
    def doctor(self):
        java=subprocess.run(["java","-version"],capture_output=True,text=True)
        runner=self.path/("run_synthea.bat" if shutil.which("cmd") else "run_synthea")
        return {"java_available":java.returncode==0,"synthea_path":str(self.path),"runner_exists":runner.exists()}
    def setup(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(["git","clone","https://github.com/synthetichealth/synthea.git",str(self.path)],check=True)
        return self.doctor()
