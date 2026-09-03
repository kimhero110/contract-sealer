"""PyInstaller 打包入口。"""

from app.main import main
import sys

if __name__ == "__main__":
    sys.exit(main())
