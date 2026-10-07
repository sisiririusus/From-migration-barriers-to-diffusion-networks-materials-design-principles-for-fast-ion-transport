# Validation / 验证记录

Version 0.2 was built from this source tree on both operating systems.

| Platform | Build environment | Result |
| --- | --- | --- |
| Windows x86-64 | Python 3.7, scikit-learn 1.0.2, PyQt5 5.15.9 | Portable EXE and ZIP built; GUI stayed open in offscreen smoke test |
| Linux x86-64 | glibc 2.28, Python 3.9, scikit-learn 1.0.2, PyQt5 5.15.7 | Portable executable and ZIP built; GUI stayed open in offscreen smoke test |

The six source tests passed on Windows and Linux. They include parity of all 713 saved candidate predictions to within `1e-8 eV`, exact Na23 and Ca7 periodic readouts, Na10 candidate membership, custom JSON prediction with OOD information, graph import rejection, and language-switch preservation of the current result.

Both frozen applications passed `packaging/verify_portable.py`: model/resource validation, Na23 recomputation, a user JSON prediction (`0.52805 eV`), CIF and POSCAR imports, CSV periodic graph analysis, and English HTML report generation. The outputs were written to isolated run directories. Linux model loading and numerical parity were checked with the frozen model under Python 3.9.

These checks verify the packaged computation and headless Qt launch. A visual desktop inspection on multiple Linux distributions and a GitHub Actions run have not yet been performed.

六项源码测试和两个发行包的实际计算检查均通过。Linux 图形界面做了无显示器启动测试；尚未在多种 Linux 桌面发行版上逐一人工检查界面，也尚未运行 GitHub Actions。
