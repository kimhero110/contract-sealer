# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec：剔除未使用组件给 exe 瘦身。

剔除项（原 onedir 301MB 的主要赘肉）：
- cv2 的 ffmpeg 视频后端（~29MB）：本工具只用图像处理，不碰视频
- opengl32sw（~20MB）：软件 OpenGL 兜底，纯 Widgets 应用用不到
- Qt6 Qml/Quick/Pdf/Network/OpenGL/Multimedia（~20MB）：只用 QtWidgets/Gui/Core
- Pillow 的 AVIF 插件（~7.5MB）
- Qt 翻译文件、SSL/crypto（无网络功能）、tkinter/unittest 等
"""

import os

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=[],
    datas=[("docs/coffee.png", "docs")],  # 「关于」对话框配图，冻结后从 _MEIPASS/docs 读取
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.QtNetwork",
        "PySide6.QtPdf",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtSerialPort",
        "PySide6.QtBluetooth",
        "PySide6.QtNfc",
        "PySide6.QtPositioning",
        "PySide6.QtSensors",
        "PySide6.QtWebSockets",
        "PySide6.QtSql",
        "PySide6.QtTest",
        "PySide6.QtDesigner",
        "PySide6.QtHelp",
        "ssl",
        "tkinter",
        "unittest",
        "pydoc",
        "doctest",
    ],
    noarchive=False,
)

# ── 二进制/数据瘦身 ──
_DROP_BIN_KEYWORDS = (
    "opencv_videoio_ffmpeg",   # cv2 视频后端
    "opengl32sw",              # 软件 OpenGL
    "Qt6Qml", "Qt6Quick", "Qt6Pdf", "Qt6Network", "Qt6OpenGL",
    "Qt6Multimedia", "Qt63D", "Qt6Charts", "Qt6DataVisualization",
    "Qt6SerialPort", "Qt6Bluetooth", "Qt6Nfc", "Qt6Positioning",
    "Qt6Sensors", "Qt6WebSockets", "Qt6Sql", "Qt6Test",
    "Qt6WebEngine", "libcrypto", "libssl",
    "_avif",
)
a.binaries = [
    b for b in a.binaries
    if not any(k in os.path.basename(b[0]) for k in _DROP_BIN_KEYWORDS)
]

# Qt 插件只保留 platforms（qwindows）；imageformats 不需要（图像全部由 numpy 内存构建）
_DROP_PLUGIN_DIRS = ("imageformats", "tls", "networkinformation", "multimedia",
                     "sqldrivers", "printsupport", "texttospeech", "sensors",
                     "position", "geoservices", "sceneparsers", "renderplugins")
a.binaries = [
    b for b in a.binaries
    if not ("plugins" + "/" in b[0].replace("\\", "/")
            and any(p in b[0].replace("\\", "/") for p in _DROP_PLUGIN_DIRS))
]

# 翻译文件（界面文案全是代码内中文，不需要 Qt 翻译包）
a.datas = [d for d in a.datas if "translations" not in d[0].replace("\\", "/")]

pyz = PYZ(a.pure)

# ── 单文件版（GitHub Release / 本地分发）──
exe_onefile = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="合同盖章工具",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX 易被国产杀软误报，且章类工具被误报很伤信任
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── 目录版（Gitee 7z 分发用，压缩率更高）──
exe_onedir = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="contract-sealer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe_onedir,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="contract-sealer",
)
