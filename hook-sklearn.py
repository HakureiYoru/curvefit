from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('sklearn.metrics')
hiddenimports += collect_submodules('sklearn.utils')
# 添加更多模块如果需要...
