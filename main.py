import sys
from PyQt6 import QtGui, QtCore
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QToolBar
from Ui_EVTestDataAnalysis import Ui_MainWindow
from callBack import *
from setup_logging import *


def main():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # 核心业务逻辑
    # 锁定窗口大小
    MainWindow.setFixedSize(1200, 855)

    # 记录日志
    setup_logging()
    sys.excepthook = handle_exception

    # 菜单栏action单选
    ui.actionGroup = QtGui.QActionGroup(MainWindow)
    ui.actionGroup.addAction(ui.actionSort)
    ui.actionGroup.addAction(ui.actionCycle)
    ui.cycleGrope = QtGui.QActionGroup(MainWindow)
    ui.cycleGrope.addAction(ui.actionCLTC)
    ui.cycleGrope.addAction(ui.actionWLTC)
    ui.cycleGrope.addAction(ui.actionNEDC)
    ui.unitGrope = QtGui.QActionGroup(MainWindow)
    ui.unitGrope.addAction(ui.action_4)
    ui.unitGrope.addAction(ui.action_5)
    ui.freqGrope = QtGui.QActionGroup(MainWindow)
    ui.freqGrope.addAction(ui.action1Hz)
    ui.freqGrope.addAction(ui.action10Hz)
    ui.freqGrope.addAction(ui.action20Hz_2)
    ui.freqGrope.addAction(ui.action100Hz)

    # 配置lineEdit为只能输入数字
    ui.lineEdit_9.setValidator(QtGui.QDoubleValidator(-80.0, 200.0, 2))
    ui.lineEdit_10.setValidator(QtGui.QDoubleValidator(-2.5, 2.5, 4))
    ui.lineEdit_11.setValidator(QtGui.QDoubleValidator(0, 0.1, 6))
    ui.lineEdit_12.setValidator(QtGui.QDoubleValidator(0, 3000, 1))

    # 工具栏
    # run
    tool_bar: QToolBar = ui.toolBar
    run_action = QtGui.QAction(QtGui.QIcon('./picture/start.webp'), '运行')
    tool_bar.addAction(run_action)
    run_action.triggered.connect(lambda: run_callback(ui, MainWindow))
    tool_bar.addSeparator()
    # save
    save_action = QtGui.QAction(QtGui.QIcon('./picture/save.png'), '另存为')
    tool_bar.addAction(save_action)
    save_action.triggered.connect(lambda: save_as_callback(ui, MainWindow))
    tool_bar.addSeparator()
    # output
    output_action = QtGui.QAction(QtGui.QIcon('./picture/output.png'), '结果输出')
    tool_bar.addAction(output_action)
    output_action.triggered.connect(lambda: output_callback(ui, MainWindow))
    tool_bar.addSeparator()
    tool_bar.setIconSize(QSize(40, 40))
    tool_bar.setMovable(False)

    # 设置action的回调
    # 文件-保存配置
    ui.action.triggered.connect(lambda: generate_filterd_signals_list(ui))
    # 文件-导入配置
    ui.action_2.triggered.connect(lambda: load_filterd_signals_list(ui))
    # 工具-分循环保存文件
    ui.actionLOG.triggered.connect(
        lambda: save_log_as_segments(ui, MainWindow))
    ui.action_8.triggered.connect(
        lambda: save_power_as_segments(ui, MainWindow))
    ui.action_9.triggered.connect(
        lambda: save_dyno_as_segments(ui, MainWindow))
    # 工具-手动添加循环
    ui.menu_10.setEnabled(False)
    ui.action_10.triggered.connect(ui.showAddCycleDialog)
    ui.action_11.triggered.connect(
        lambda: clear_attribute(ui, 'add_cycle_info'))
    # 工具-手动删除循环
    ui.menu_11.setEnabled(False)
    ui.action_12.triggered.connect(lambda: del_cycle_manual(ui, MainWindow))
    ui.action_13.triggered.connect(
        lambda: clear_attribute(ui, 'del_cycle_info'))
    # 工具-导出标准工况
    ui.action_MDF.triggered.connect(lambda: save_std_cycle_as_segment(ui))
    # 设置-工况识别相关性系数
    ui.correlation_coef = 0.8
    ui.action_3.triggered.connect(lambda: set_correlation_coef(ui, MainWindow))
    # 帮助
    ui.menu_3.aboutToShow.connect(help_callback)

    # 导入MDF文件
    ui.pushButton.clicked.connect(lambda: load_mdf_file(ui))
    # 车速
    ui.lineEdit_1.textChanged.connect(lambda: filter_combobox_items(
        ui.comboBox_1, ui.unique_signals, ui.lineEdit_1.text()))
    ui.comboBox_1.activated.connect(
        lambda: update_signal_data(ui, ui.comboBox_1, ui.lineEdit_1, 'mdf_vehspd'))
    # 实际SOC
    ui.lineEdit_2.textChanged.connect(lambda: filter_combobox_items(
        ui.comboBox_2, ui.unique_signals, ui.lineEdit_2.text()))
    ui.comboBox_2.activated.connect(
        lambda: update_signal_data(ui, ui.comboBox_2, ui.lineEdit_2, 'mdf_socact'))
    # 电池电流
    ui.lineEdit_3.textChanged.connect(lambda: filter_combobox_items(
        ui.comboBox_3, ui.unique_signals, ui.lineEdit_3.text()))
    ui.comboBox_3.activated.connect(
        lambda: update_signal_data(ui, ui.comboBox_3, ui.lineEdit_3, 'mdf_battcurr'))
    # 电池电压
    ui.lineEdit_4.textChanged.connect(lambda: filter_combobox_items(
        ui.comboBox_4, ui.unique_signals, ui.lineEdit_4.text()))
    ui.comboBox_4.activated.connect(
        lambda: update_signal_data(ui, ui.comboBox_4, ui.lineEdit_4, 'mdf_battvolt'))
    # DCDC电流
    ui.lineEdit_5.textChanged.connect(lambda: filter_combobox_items(
        ui.comboBox_5, ui.unique_signals, ui.lineEdit_5.text()))
    ui.comboBox_5.activated.connect(
        lambda: update_signal_data(ui, ui.comboBox_5, ui.lineEdit_5, 'mdf_dcdccurr'))
    # DCDC电压
    ui.lineEdit_6.textChanged.connect(lambda: filter_combobox_items(
        ui.comboBox_6, ui.unique_signals, ui.lineEdit_6.text()))
    ui.comboBox_6.activated.connect(
        lambda: update_signal_data(ui, ui.comboBox_6, ui.lineEdit_6, 'mdf_dcdcvolt'))
    # ECP功耗
    ui.lineEdit_7.textChanged.connect(lambda: filter_combobox_items(
        ui.comboBox_7, ui.unique_signals, ui.lineEdit_7.text()))
    ui.comboBox_7.activated.connect(
        lambda: update_signal_data(ui, ui.comboBox_7, ui.lineEdit_7, 'mdf_ecppwr'))
    # PTC功耗
    ui.lineEdit_8.textChanged.connect(lambda: filter_combobox_items(
        ui.comboBox_8, ui.unique_signals, ui.lineEdit_8.text()))
    ui.comboBox_8.activated.connect(
        lambda: update_signal_data(ui, ui.comboBox_8, ui.lineEdit_8, 'mdf_ptcpwr'))

    # 导入CSV文件
    ui.pushButton_2.clicked.connect(lambda: load_csv_file(ui))
    # 电池
    ui.comboBox_9.activated.connect(
        lambda: select_node_data(ui, ui.comboBox_9, 'csv_batt'))
    # 电驱
    ui.comboBox_10.activated.connect(
        lambda: select_node_data(ui, ui.comboBox_10, 'csv_ipu'))
    # IPS
    ui.comboBox_15.activated.connect(
        lambda: select_node_data(ui, ui.comboBox_15, 'csv_ips'))
    # DCDC高压
    ui.comboBox_11.activated.connect(
        lambda: select_node_data(ui, ui.comboBox_11, 'csv_dcdc_high'))
    # DCDC低压
    ui.comboBox_12.activated.connect(
        lambda: select_node_data(ui, ui.comboBox_12, 'csv_dcdc_low'))
    # ECP
    ui.comboBox_13.activated.connect(
        lambda: select_node_data(ui, ui.comboBox_13, 'csv_ecp'))
    # PTC
    ui.comboBox_14.activated.connect(
        lambda: select_node_data(ui, ui.comboBox_14, 'csv_ptc'))

    # 导入Dyno文件
    ui.pushButton_4.clicked.connect(lambda: load_dyno_file(ui))
    ui.comboBox_23.activated.connect(
        lambda: select_dyno_data(ui, ui.comboBox_23, 'dyno_vehspd'))
    ui.comboBox_24.activated.connect(
        lambda: select_dyno_data(ui, ui.comboBox_24, 'dyno_vehacc'))

    # 显示
    MainWindow.show()

    # 启动APP提示
    QtCore.QTimer.singleShot(100, ui.show_must_read_info)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
