# PyInstaller build description:  pyinstaller --noconfirm signchat.spec
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs

# The OpenVR client library ships inside the pyopenvr package and is loaded from there,
# so it must end up in the "openvr" folder of the build. Keep only this platform's copy.
ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
openvr_libs = [(src, dest) for src, dest in collect_dynamic_libs("openvr") if src.endswith(ext)]

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=openvr_libs,
    datas=[("signchat/steamvr", "signchat/steamvr")],
    hiddenimports=["signchat.sources.steamvr", "signchat.sources.simulator", "signchat.sources.webcam"],
    excludes=["mediapipe", "cv2", "matplotlib", "tkinter", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SignChat", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="SignChat")
