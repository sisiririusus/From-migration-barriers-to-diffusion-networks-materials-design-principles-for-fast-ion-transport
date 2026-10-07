"""Desktop interface for paper examples, user projects, and exact network readouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import uuid

from PyQt5.QtCore import Qt, QProcess, QSettings, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QPen, QBrush, QPainter, QTextOption
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget, QInputDialog, QHeaderView,
)

from .examples import catalog, load_case
from .i18n import translate
from .paths import ROOT, RUNS


class GraphView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#f8fafc"))

    def wheelEvent(self, event):
        self.scale(1.18 if event.angleDelta().y() > 0 else 1 / 1.18, 1.18 if event.angleDelta().y() > 0 else 1 / 1.18)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Migration Network Workbench")
        self.resize(960, 620)
        self.current = None
        self.current_folder = None
        self.pending_output = ""
        self.job_status_file = None
        self.process = None
        self.graph_items = {}
        self.cases = catalog()
        self.settings = QSettings("MigrationNetworkWorkbench", "MigrationNetworkWorkbench")
        self.language = self.settings.value("language", "zh")
        if self.language not in ("zh", "en"):
            self.language = "zh"
        self._build_ui()
        self._load_case(self.cases[0]["id"])

    def t(self, text):
        return translate(text, self.language)

    def _change_language(self, index):
        language = self.language_combo.itemData(index)
        if language == self.language:
            return
        old_result, old_folder = self.current, self.current_folder
        old_case = self.case_combo.currentData()
        old_tab = self.tabs.currentIndex()
        old_threshold = self.threshold.value()
        old_log = self.log.toPlainText()
        self.language = language
        self.settings.setValue("language", language)
        self._build_ui()
        self.case_combo.setCurrentIndex(self.case_combo.findData(old_case))
        if old_result is not None:
            self._show_result(old_result, old_folder)
            self.threshold.setValue(old_threshold)
        self.log.setPlainText(old_log)
        self.tabs.setCurrentIndex(old_tab)
        if self.process and self.process.state() != QProcess.NotRunning:
            self.status_label.setText(self.t("正在计算…"))

    def _button(self, title, callback):
        button = QPushButton(self.t(title))
        button.clicked.connect(callback)
        button.setMinimumHeight(34)
        return button

    def _build_ui(self):
        t = self.t
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(22, 18, 22, 16)
        outer.setSpacing(13)

        title = QLabel("Migration Network Workbench")
        title.setObjectName("title")
        outer.addWidget(title)
        subtitle = QLabel(t("迁移势垒预测 · 精确周期网络 · 可追溯 NEB 候选"))
        subtitle.setObjectName("subtitle")
        outer.addWidget(subtitle)

        case_bar = QHBoxLayout()
        case_bar.addWidget(QLabel(t("论文示例")))
        self.case_combo = QComboBox()
        for case in self.cases:
            self.case_combo.addItem(t("%s  ·  %d 节点 / %d 边") % (case["formula_species_group"], case["n_nodes"], case["n_edges"]), case["id"])
        self.case_combo.setMinimumWidth(220)
        case_bar.addWidget(self.case_combo)
        case_bar.addWidget(self._button("打开示例", self._open_selected_case))
        case_bar.addWidget(self._button("实际重算", self._recompute_case))
        case_bar.addStretch(1)
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.language_combo.setCurrentIndex(0 if self.language == "zh" else 1)
        self.language_combo.currentIndexChanged.connect(self._change_language)
        case_bar.addWidget(self.language_combo)
        outer.addLayout(case_bar)
        action_bar = QHBoxLayout()
        action_bar.addWidget(self._button("导入项目 JSON", self._open_json))
        action_bar.addWidget(self._button("导入 CIF / POSCAR", self._open_structure))
        action_bar.addWidget(self._button("导入图 CSV", self._open_graph_csv))
        action_bar.addStretch(1)
        outer.addLayout(action_bar)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        overview = QWidget()
        ov = QVBoxLayout(overview)
        self.name_label = QLabel("—")
        self.name_label.setObjectName("caseTitle")
        ov.addWidget(self.name_label)
        self.metrics_label = QLabel("—")
        self.metrics_label.setObjectName("metrics")
        self.metrics_label.setWordWrap(True)
        ov.addWidget(self.metrics_label)
        self.scope_label = QLabel("—")
        self.scope_label.setWordWrap(True)
        self.scope_label.setObjectName("scope")
        ov.addWidget(self.scope_label)
        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setWordWrapMode(QTextOption.WrapAnywhere)
        ov.addWidget(self.overview_text, 1)
        self.tabs.addTab(overview, t("概览与来源"))

        network = QWidget()
        nv = QVBoxLayout(network)
        selector = QHBoxLayout()
        selector.addWidget(QLabel(t("显示势垒 ≤")))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 20)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.05)
        self.threshold.setSuffix(" eV")
        self.threshold.valueChanged.connect(self._draw_graph)
        selector.addWidget(self.threshold)
        self.graph_hint = QLabel(t("a-b 平面投影；周期像位移写在边详情中"))
        selector.addWidget(self.graph_hint)
        selector.addStretch(1)
        nv.addLayout(selector)
        self.scene = QGraphicsScene()
        self.graph_view = GraphView()
        self.graph_view.setScene(self.scene)
        nv.addWidget(self.graph_view, 1)
        self.selected_detail = QLabel(t("选中下方表格中的边，查看精确身份。"))
        self.selected_detail.setWordWrap(True)
        self.selected_detail.setObjectName("details")
        nv.addWidget(self.selected_detail)
        self.tabs.addTab(network, t("周期图"))

        edges_tab = QWidget()
        ev = QVBoxLayout(edges_tab)
        ev.addWidget(QLabel(t("每一行对应一个 edge_uid；相同端点的不同周期像分别保留。")))
        self.edges_table = QTableWidget()
        self.edges_table.setColumnCount(8)
        self.edges_table.setHorizontalHeaderLabels([t(value) for value in ["edge_uid", "起点", "终点", "像位移", "势垒 eV", "适用域", "OOD95", "来源"]])
        self.edges_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.edges_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.edges_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.edges_table.itemSelectionChanged.connect(self._edge_selected)
        ev.addWidget(self.edges_table)
        self.tabs.addTab(edges_tab, t("候选边"))

        neb_tab = QWidget()
        ne = QVBoxLayout(neb_tab)
        self.neb_note = QLabel("—")
        self.neb_note.setWordWrap(True)
        ne.addWidget(self.neb_note)
        self.neb_table = QTableWidget()
        self.neb_table.setColumnCount(7)
        self.neb_table.setHorizontalHeaderLabels([t(value) for value in ["序号 / 次序", "edge_uid", "预测 eV", "关键边", "适用域", "OOD95", "原因 / 组别"]])
        self.neb_table.setColumnWidth(0, 110)
        self.neb_table.setColumnWidth(1, 330)
        self.neb_table.setColumnWidth(2, 115)
        self.neb_table.setColumnWidth(3, 90)
        self.neb_table.setColumnWidth(4, 90)
        self.neb_table.setColumnWidth(5, 80)
        self.neb_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.neb_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ne.addWidget(self.neb_table)
        self.tabs.addTab(neb_tab, t("NEB 候选"))

        run_tab = QWidget()
        rv = QVBoxLayout(run_tab)
        rv.addWidget(QLabel(t("计算在独立进程运行。取消后不会把未完成输出标为成功。")))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        rv.addWidget(self.log, 1)
        row = QHBoxLayout()
        self.status_label = QLabel(t("就绪"))
        row.addWidget(self.status_label)
        row.addStretch(1)
        row.addWidget(self._button("打开本次输出", self._open_output))
        row.addWidget(self._button("取消正在运行的任务", self._cancel))
        rv.addLayout(row)
        self.tabs.addTab(run_tab, t("运行与导出"))

        self.setStyleSheet("""
            QMainWindow,QWidget{background:#f8fafc;color:#0f172a;font:13px sans-serif;}
            QLabel#title{font-size:27px;font-weight:700;color:#0b2942;}
            QLabel#subtitle{color:#475569;margin-bottom:5px;}
            QLabel#caseTitle{font-size:23px;font-weight:700;margin:12px 0;}
            QLabel#metrics{font-size:18px;color:#0b5c86;padding:12px;background:#eaf6fc;border-radius:7px;}
            QLabel#scope{font-size:14px;color:#7c3d12;background:#fff7ed;padding:12px;border-radius:7px;}
            QLabel#details{padding:10px;background:#e2e8f0;border-radius:5px;}
            QPushButton{background:#0f5278;color:white;border:0;border-radius:5px;padding:5px 12px;font-weight:600;}
            QPushButton:hover{background:#0b3b57;}
            QComboBox,QDoubleSpinBox,QTextEdit,QTableWidget,QGraphicsView{background:white;border:1px solid #cbd5e1;border-radius:5px;}
            QTabWidget::pane{border:1px solid #cbd5e1;background:#fff;}
            QTabBar::tab{padding:8px 17px;background:#e2e8f0;}
            QTabBar::tab:selected{background:white;color:#0f5278;font-weight:700;}
        """)

    def _open_selected_case(self):
        self._load_case(self.case_combo.currentData())

    def _load_case(self, case_id):
        try:
            self._show_result(load_case(case_id, recompute=False), None)
        except Exception as error:
            QMessageBox.critical(self, self.t("示例读取失败"), str(error))

    def _recompute_case(self):
        self._start_job(["paper", str(self.case_combo.currentData()), "--recompute"])

    def _open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, self.t("选择项目 JSON"), str(ROOT / "examples"), "JSON (*.json)")
        if path:
            self._start_job(["analyze", path])

    def _open_structure(self):
        path, _ = QFileDialog.getOpenFileName(self, self.t("选择三维周期结构"), str(ROOT), "Crystal structures (*.cif *.vasp *.poscar POSCAR* *);;All files (*)")
        if not path:
            return
        species, ok = QInputDialog.getText(self, self.t("迁移离子"), self.t("迁移离子元素符号（例如 Na）："))
        if not ok or not species.strip():
            return
        index, ok = QInputDialog.getInt(self, self.t("初始原子"), self.t("初始迁移离子的原子索引（从 0 开始）："), 0, 0, 100000)
        if not ok:
            return
        endpoint, ok = QInputDialog.getText(self, self.t("目标位点"), self.t("目标分数坐标 x,y,z（明确的空位或候选位点）："))
        if not ok or not endpoint.strip():
            return
        shift, ok = QInputDialog.getText(self, self.t("周期像"), self.t("目标位点整数周期位移 a,b,c："), text="0,0,0")
        if not ok:
            return
        self._start_job(["structure", path, "--species", species.strip(), "--moving-atom-index", str(index), "--end-frac", endpoint.strip(), "--shift", shift.strip()])

    def _open_graph_csv(self):
        nodes, _ = QFileDialog.getOpenFileName(self, self.t("选择节点 CSV"), str(ROOT), "CSV (*.csv)")
        if not nodes:
            return
        edges, _ = QFileDialog.getOpenFileName(self, self.t("选择边 CSV"), str(ROOT), "CSV (*.csv)")
        if edges:
            self._start_job(["graph-csv", "--nodes", nodes, "--edges", edges])

    def _start_job(self, cli_arguments):
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, self.t("已有任务"), self.t("请先完成或取消当前任务。"))
            return
        self.pending_output = ""
        self.job_status_file = RUNS / ("job_" + uuid.uuid4().hex + ".json")
        self.log.clear()
        self.tabs.setCurrentIndex(4)
        self.status_label.setText(self.t("正在计算…"))
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process)
        self.process.finished.connect(self._job_finished)
        self.process.setWorkingDirectory(str(ROOT))
        cli_prefix = ["--cli", "--status-file", str(self.job_status_file), "--language", self.language]
        command = cli_prefix + cli_arguments if getattr(sys, "frozen", False) else ["-B", str(ROOT / "start.py")] + cli_prefix + cli_arguments
        self.process.start(sys.executable, command)
        self.log.append(self.t("运行：") + " ".join(cli_arguments))

    def _read_process(self):
        if not self.process:
            return
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in output.splitlines():
            if line.startswith("RESULT_DIR="):
                self.pending_output = line.split("=", 1)[1].strip()
            self.log.append(line)

    def _job_finished(self, code, status):
        self._read_process()
        if self.job_status_file and self.job_status_file.is_file():
            try:
                job = json.loads(self.job_status_file.read_text(encoding="utf-8"))
                if job.get("result_dir"):
                    self.pending_output = str(job["result_dir"])
                if job.get("error"):
                    self.log.append(str(job["error"]))
            except Exception as error:
                self.log.append(self.t("状态文件读取失败：") + str(error))
        if code == 0 and self.pending_output:
            folder = Path(self.pending_output)
            try:
                result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                self._show_result(result, folder)
                self.status_label.setText(self.t("完成"))
                self.log.append(self.t("输出：") + str(folder))
                self.tabs.setCurrentIndex(0)
            except Exception as error:
                self.status_label.setText(self.t("结果读取失败"))
                self.log.append(self.t("结果读取失败：") + str(error))
        else:
            self.status_label.setText(self.t("已取消或失败（退出码 %s）") % code)
        self.process = None

    def _cancel(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.status_label.setText(self.t("已请求取消"))

    def _open_output(self):
        if not self.current_folder:
            QMessageBox.information(self, self.t("没有输出"), self.t("当前是论文示例的已保存数据。点击“实际重算”或导入项目后会生成输出。"))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_folder)))

    def _display_threshold(self, value):
        return self.t("不可达") if value is None or value == "" else "%.4f eV" % float(value)

    def _show_result(self, result, folder):
        self.current = result
        self.current_folder = folder
        self.name_label.setText(result.get("name") or result["graph_id"])
        r = result["readout"]
        self.metrics_label.setText(self.t("%d 节点 · %d 边 · winding rank %s\nEperc(a/b/c): %s / %s / %s") % (
            len(result["nodes"]), len(result["edges"]), r.get("winding_vector_rank", "—"),
            self._display_threshold(r.get("Eperc_a")), self._display_threshold(r.get("Eperc_b")), self._display_threshold(r.get("Eperc_c")),
        ))
        mode = self.t("本次实际计算") if result.get("computed_now") else self.t("冻结论文数据（可点击实际重算）")
        self.scope_label.setText(mode + "\n" + self.t(result.get("interpretation", "")))
        sources = result.get("source_files") or [result.get("input_path") or self.t("当前用户项目")]
        display_sources = []
        for value in sources:
            try:
                display_sources.append(str(Path(value).resolve().relative_to(ROOT.resolve())))
            except (ValueError, TypeError):
                display_sources.append(str(value))
        self.overview_text.setPlainText(
            self.t("图身份：%s\n迁移离子：%s\n数据角色：%s\n\n来源：\n%s\n\n关键边轨道：%s\n图适用域：%s\n\n每条边的详细数值在“候选边”页；导出目录含完整 CSV、JSON 和 HTML 报告。") % (
                result["graph_id"], result.get("migrating_species") or "—", mode,
                "\n".join(display_sources),
                r.get("critical_periodic_edge_count", len(r.get("critical_periodic_edges", []))),
                (result.get("graph_applicability") or {}).get("graph_applicability_score", r.get("graph_applicability_score", "—")) if isinstance(result.get("graph_applicability") or {}, dict) else "—",
            )
        )
        barriers = [float(row["predicted_barrier_eV"]) for row in result["edges"]]
        self.threshold.blockSignals(True)
        self.threshold.setMaximum(max(20.0, max(barriers) + 0.01))
        self.threshold.setValue(max(barriers) + 0.001)
        self.threshold.blockSignals(False)
        self._fill_edges()
        self._fill_recommendations()
        self._draw_graph()

    def _fill_edges(self):
        edges = self.current["edges"]
        self.edges_table.blockSignals(True)
        self.edges_table.clearSelection()
        self.edges_table.setRowCount(len(edges))
        for index, edge in enumerate(edges):
            values = [
                edge["edge_uid"], edge["node_i"], edge["node_j"],
                str(edge["image_shift"]), "%.6f" % float(edge["predicted_barrier_eV"]),
                "—" if edge.get("applicability_score") is None else "%.3f" % float(edge["applicability_score"]),
                str(edge.get("ood_warning_95", "—")), edge.get("prediction_role") or edge.get("barrier_source") or "—",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self.edges_table.setItem(index, column, item)
        self.edges_table.blockSignals(False)

    def _fill_recommendations(self):
        rows = self.current.get("recommendations", [])
        self.neb_table.setRowCount(len(rows))
        if self.current["kind"] == "paper_example":
            self.neb_note.setText(self.t("论文来源表的候选分组；左列是来源行序号，并非新计算的排名。"))
        else:
            self.neb_note.setText(self.t("本次计算的字典序预筛。没有验证的路径家族多样性分量；树间分歧不是校准区间。"))
        for index, row in enumerate(rows):
            raw_barrier = row.get("full_model_predicted_barrier_eV") or row.get("oof_predicted_barrier_eV") or row.get("predicted_barrier_eV")
            raw_applicability = row.get("applicability_score")
            raw_critical = row.get("critical", row.get("periodic_critical_edge", row.get("category_periodic_critical_edge", "—")))
            raw_ood = row.get("ood_warning_95")
            values = [
                row.get("exploit_rank", index + 1), row.get("edge_uid"),
                "—" if raw_barrier in (None, "") else "%.4f" % float(raw_barrier),
                self.t("是") if raw_critical in (True, 1, "1") else self.t("否") if raw_critical in (False, 0, "0") else raw_critical,
                "—" if raw_applicability in (None, "") else "%.3f" % float(raw_applicability),
                self.t("是") if raw_ood in (True, 1, "1") else self.t("否") if raw_ood in (False, 0, "0") else "—",
                row.get("prospective_group_membership") or row.get("pareto_category") or row.get("ranking_note") or "—",
            ]
            for column, value in enumerate(values):
                self.neb_table.setItem(index, column, QTableWidgetItem(str(value)))

    def _edge_selected(self):
        if not self.current:
            return
        selections = self.edges_table.selectionModel().selectedRows()
        if not selections:
            return
        edge = self.current["edges"][selections[0].row()]
        self.selected_detail.setText(
            self.t("%s  |  %s → %s @ %s  |  势垒 %.6f eV  |  适用域 %s  |  %s") % (
                edge["edge_uid"], edge["node_i"], edge["node_j"], edge["image_shift"],
                float(edge["predicted_barrier_eV"]), edge.get("applicability_score", "—"),
                edge.get("prediction_role") or edge.get("barrier_source") or "—",
            )
        )
        self._draw_graph()

    def _draw_graph(self):
        if not self.current:
            return
        self.scene.clear()
        self.graph_items.clear()
        nodes = {node["id"]: node["frac_coords"] for node in self.current["nodes"]}
        size = 480.0
        margin = 70.0
        self.scene.addRect(margin, margin, size, size, QPen(QColor("#94a3b8"), 2))
        chosen = None
        selected = self.edges_table.selectionModel().selectedRows() if self.edges_table.selectionModel() else []
        if selected:
            chosen = self.current["edges"][selected[0].row()]["edge_uid"]
        visible = 0
        for edge in self.current["edges"]:
            barrier = float(edge["predicted_barrier_eV"])
            if barrier > self.threshold.value() + 1e-9:
                continue
            a = nodes[edge["node_i"]]
            b = nodes[edge["node_j"]]
            shift = edge["image_shift"]
            x1, y1 = margin + (a[0] % 1.0) * size, margin + (1.0 - a[1] % 1.0) * size
            x2, y2 = margin + ((b[0] % 1.0) + shift[0]) * size, margin + (1.0 - (b[1] % 1.0) - shift[1]) * size
            color = QColor("#e75a29") if edge["edge_uid"] == chosen else QColor("#287f9b")
            pen = QPen(color, 4 if edge["edge_uid"] == chosen else 1.4)
            item = self.scene.addLine(x1, y1, x2, y2, pen)
            item.setToolTip("%s · %.4f eV · shift %s" % (edge["edge_uid"], barrier, shift))
            self.graph_items[edge["edge_uid"]] = item
            visible += 1
        for node_id, coords in nodes.items():
            x = margin + (coords[0] % 1.0) * size
            y = margin + (1.0 - coords[1] % 1.0) * size
            self.scene.addEllipse(x - 5, y - 5, 10, 10, QPen(QColor("#0b2942"), 1), QBrush(QColor("white")))
            label = self.scene.addText(node_id.split("_")[-1], QFont("Arial", 8))
            label.setDefaultTextColor(QColor("#0b2942"))
            label.setPos(x + 6, y - 15)
        self.graph_hint.setText(self.t("a-b 平面投影 · 显示 %d / %d 边 · 跨胞线保留整数像位移，c 方向见表格") % (visible, len(self.current["edges"])))
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-30, -30, 30, 30))
        self.graph_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Migration Network Workbench")
    window = MainWindow()
    window.show()
    return app.exec_()
