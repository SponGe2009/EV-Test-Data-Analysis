import os
import shutil
import logging
from PyQt6 import QtCore
from PyQt6.QtWidgets import QFileDialog, QProgressDialog, QInputDialog
from asammdf import MDF, Signal
from scipy.io import loadmat
from Ui_EVTestDataAnalysis import Ui_MainWindow
from funcs import *


def load_mdf_file(ui: Ui_MainWindow):

    ui.tabWidget.setEnabled(False)
    logger = logging.getLogger('AppLogger')
    logger.debug('The load_mdf_file method is called')

    # Open file dialog to select an MDF file
    filePath, _ = QFileDialog.getOpenFileName(
        None, "Select MDF file", "", "MDF Files (*.mf4 *.mdf)")
    if filePath:  # Proceed only if a file was selected
        ui.textEdit.clear()
        ui.textEdit.setText(filePath)
        ui.mdf = MDF(filePath)
        all_signals = list(ui.mdf.channels_db.keys())  # Get all signal names
        # Remove duplicates and sort
        ui.unique_signals = sorted(set(all_signals))

        # Update and enable widgets
        update_and_enable_widgets(ui, ui.unique_signals)
        ui.menu_10.setEnabled(False)
        ui.menu_11.setEnabled(False)
        if hasattr(ui, 'add_cycle_info'):
            delattr(ui, 'add_cycle_info')
    ui.tabWidget.setEnabled(True)


def load_csv_file(ui: Ui_MainWindow):

    ui.tabWidget.setEnabled(False)
    logger = logging.getLogger('AppLogger')
    logger.debug('The load_csv_file method is called')

    full_df, file_paths = load_and_merge_csvs()
    if file_paths:
        extract_signals(ui, full_df)
        ui.listWidget.clear()
        ui.listWidget.addItems(file_paths)
        num = sum('Idc' in s for s in ui.csv_data.columns.tolist())
        node_list = [f'测点{i+1}' for i in range(num)]
        update_and_enable_combobox(ui, node_list)
    ui.tabWidget.setEnabled(True)


def load_dyno_file(ui: Ui_MainWindow):

    ui.tabWidget.setEnabled(False)
    logger = logging.getLogger('AppLogger')
    logger.debug('The load_dyno_file method is called')

    # Let the user select a file
    file_path, _ = QFileDialog.getOpenFileName(
        None, "Select Data File", "", "Data Files (*.log *.csv *.txt)")

    # Check if a file was selected
    if file_path:
        # Determine the file extension
        if file_path.endswith('.csv'):
            data = read_csv_file(ui, file_path)
        elif file_path.endswith('.log'):
            data = read_log_file(ui, file_path)
        elif file_path.endswith('.txt'):
            data = read_txt_file(ui, file_path)

        ui.comboBox_23.clear()
        ui.comboBox_23.addItems(data.columns)
        ui.comboBox_23.addItem('无')
        ui.comboBox_23.setCurrentText('无')
        ui.comboBox_24.clear()
        ui.comboBox_24.addItems(data.columns)
        ui.comboBox_24.addItem('无')
        ui.comboBox_24.setCurrentText('无')
        ui.textEdit_20.clear()
        ui.textEdit_20.setText(file_path)
        ui.dyno_data = data
    ui.tabWidget.setEnabled(True)


def filter_combobox_items(comboBox, all_items, search_text):
    """
    Filters the items displayed in the comboBox based on the search_text.

    Args:
        comboBox (QComboBox): The ComboBox to update.
        all_items (list): The list of all possible items.
        search_text (str): The text to filter items.
    """
    filtered_items = [
        item for item in all_items if search_text.lower() in item.lower()]
    comboBox.clear()
    comboBox.addItems(filtered_items)


def update_signal_data(ui: Ui_MainWindow, comboBox, lineEdit, attribute_name):
    """
    Retrieves and stores the selected signal data from an MDF file.

    Args:
        ui: The main UI instance.
        comboBox: The ComboBox from which the signal name is selected.
        attribute_name: The attribute name where the signal data will be stored in the UI.
    """
    # Retrieve the selected signal name
    signal_name = comboBox.currentText()

    # Check if the MDF file is loaded and the signal name is not empty
    if hasattr(ui, 'mdf') and signal_name:
        # Extract time and value data for the selected signal
        time, values = retrieve_signal_data(ui.mdf, signal_name)

        # Store the signal data in the specified UI attribute
        setattr(ui, attribute_name, np.hstack(
            (time.reshape(-1, 1), values.reshape(-1, 1))))
        lineEdit.setText(signal_name)


def select_node_data(ui: Ui_MainWindow, comboBox, attribute_name):

    node_num = pick_number_from_string(comboBox.currentText())
    if node_num:
        filtered_columns = [
            col for col in ui.csv_data.columns if node_num in col]
        new_df = ui.csv_data[filtered_columns + ['TimeDiff']]
        # Remove all numbers from the column headers
        new_columns = [re.sub(r'\d+', '', col) for col in new_df.columns]
        new_df.columns = new_columns
        random_sample = new_df['Udc'].sample(n=100)
        if new_df['WP'].iloc[-1] >= 0:
            if random_sample.mean() < 0:
                new_df.loc[:, 'Idc'] = -new_df.loc[:, 'Idc']
                new_df.loc[:, 'Udc'] = -new_df.loc[:, 'Udc']
        else:
            new_df.loc[:, 'P'] = -new_df.loc[:, 'P']
            if random_sample.mean() < 0:
                new_df.loc[:, 'Udc'] = -new_df.loc[:, 'Udc']
            else:
                new_df.loc[:, 'Idc'] = -new_df.loc[:, 'Idc']

        setattr(ui, attribute_name, new_df)
    else:
        if hasattr(ui, attribute_name):
            delattr(ui, attribute_name)


def select_dyno_data(ui: Ui_MainWindow, comboBox, attribute_name):

    ui.tabWidget.setEnabled(False)
    if comboBox.currentText() != '无':
        for index, string in enumerate(ui.dyno_data.columns):
            # Check if the substring is in the current string
            if 'TimeDiff' in string:
                # Return the index if the substring is found
                time = ui.dyno_data[ui.dyno_data.columns[index]]
                break

        data = ui.dyno_data[comboBox.currentText()]
        selected_unit = ui.unitGrope.checkedAction()
        if selected_unit.text() == '英制':
            if attribute_name == 'dyno_vehspd':
                data = data * 1.609
                ui.dyno_data.loc[:, comboBox.currentText(
                )] = ui.dyno_data.loc[:, comboBox.currentText()] * 1.609
            elif attribute_name == 'dyno_vehacc':
                data = data * 0.447
                ui.dyno_data.loc[:, comboBox.currentText(
                )] = ui.dyno_data.loc[:, comboBox.currentText()] * 0.447

        setattr(ui, attribute_name, np.hstack(
            (time.to_numpy().astype(float).reshape(-1, 1), data.to_numpy().astype(float).reshape(-1, 1))))
    else:
        if hasattr(ui, attribute_name):
            delattr(ui, attribute_name)
    ui.tabWidget.setEnabled(True)


def run_callback(ui: Ui_MainWindow, mainWindow):

    logger = logging.getLogger('AppLogger')
    logger.debug('The run_callback method is called')

    if not hasattr(ui, 'mdf_vehspd') and not hasattr(ui, 'dyno_vehspd') and not hasattr(ui, 'dyno_vehacc'):
        show_warning("请先导入数据！")
        mainWindow.setEnabled(True)
        return

    progressDialog = QProgressDialog(
        "正在分析计算...", None, 0, 100, mainWindow)
    progressDialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    progressDialog.setAutoClose(True)
    progressDialog.setCancelButton(None)
    progressDialog.setWindowTitle("Task Progress")
    progressDialog.resize(500, 60)
    progressDialog.show()
    progressDialog.setValue(10)
    QtCore.QTimer.singleShot(100, lambda: data_analysis(ui, progressDialog))


def output_callback(ui: Ui_MainWindow, mainWindow):

    mainWindow.setEnabled(False)
    logger = logging.getLogger('AppLogger')
    logger.debug('The output_callback method is called')

    mdf_file_path = ui.textEdit.toPlainText()
    if mdf_file_path:
        initial_path = os.path.dirname(mdf_file_path)
    else:
        initial_path = os.getcwd()
    file_path, _ = QFileDialog.getSaveFileName(
        None,
        "Save File",
        initial_path,
        "Excel Files (*.xlsx)"
    )
    if file_path:
        selected_type = ui.actionGroup.checkedAction()
        if selected_type.text() == '缩短法':
            # template_details = deserialize_template_details(
            #     './data/template_details.json')
            # create_excel_from_details(template_details, file_path)
            shutil.copy('./data/template_NT.xlsx', file_path)

            if hasattr(ui, 'mdf_distance'):
                write_ndarray_to_excel(ui.mdf_distance, 0, 'B2', file_path)
                write_ndarray_to_excel(
                    np.array(ui.start_time), 0, 'R10', file_path)
                write_ndarray_to_excel(
                    np.array(ui.end_time), 0, 'S10', file_path)
            if hasattr(ui, 'mdf_battenergy'):
                write_ndarray_to_excel(ui.mdf_battenergy, 0, 'C2', file_path)
            if hasattr(ui, 'mdf_dcdcenergy'):
                write_ndarray_to_excel(
                    ui.mdf_dcdcenergy[:, 1], 0, 'F2', file_path)
            if hasattr(ui, 'mdf_ecpenergy'):
                write_ndarray_to_excel(
                    ui.mdf_ecpenergy[:, 1], 0, 'G2', file_path)
            if hasattr(ui, 'mdf_ptcenergy'):
                write_ndarray_to_excel(
                    ui.mdf_ptcenergy[:, 1], 0, 'H2', file_path)
            if hasattr(ui, 'csv_battenergy'):
                write_ndarray_to_excel(ui.csv_battenergy, 0, 'C10', file_path)
            if hasattr(ui, 'csv_ipuenergy'):
                write_ndarray_to_excel(ui.csv_ipuenergy, 0, 'I10', file_path)
            if hasattr(ui, 'csv_ipsenergy'):
                write_ndarray_to_excel(ui.csv_ipsenergy, 0, 'F10', file_path)
            if hasattr(ui, 'csv_dcdchigh_energy'):
                write_ndarray_to_excel(
                    ui.csv_dcdchigh_energy, 0, 'L10', file_path)
            if hasattr(ui, 'csv_dcdclow_energy'):
                write_ndarray_to_excel(
                    ui.csv_dcdclow_energy, 0, 'O10', file_path)
            if hasattr(ui, 'dyno_distance'):
                write_ndarray_to_excel(ui.dyno_distance, 0, 'B10', file_path)
            if hasattr(ui, 'drive_shaft_energy'):
                write_ndarray_to_excel(
                    ui.drive_shaft_energy, 0, 'I2', file_path)
            if hasattr(ui, 'resistance_energy'):
                write_ndarray_to_excel(
                    ui.resistance_energy, 0, 'L2', file_path)
            if hasattr(ui, 'kinetic_energy'):
                write_ndarray_to_excel(ui.kinetic_energy, 0, 'O2', file_path)
            if hasattr(ui, 'drive_shaft_energy_std'):
                write_ndarray_to_excel(
                    np.array([ui.drive_shaft_energy_std[0][1]]), 1, 'G10', file_path)
                write_ndarray_to_excel(
                    np.array([abs(ui.drive_shaft_energy_std[0][2])]), 1, 'G20', file_path)
            if hasattr(ui, 'kinetic_energy_std'):
                write_ndarray_to_excel(
                    np.array([ui.kinetic_energy_std[0][1]]), 0, 'S3', file_path)
                write_ndarray_to_excel(
                    np.array([ui.kinetic_energy_std[0][1]]), 1, 'G15', file_path)
            if hasattr(ui, 'resistance_energy_std'):
                write_ndarray_to_excel(
                    np.array([ui.resistance_energy_std[0][1]]), 1, 'G14', file_path)
                write_ndarray_to_excel(
                    np.array([abs(ui.resistance_energy_std[0][2])]), 1, 'G19', file_path)
            show_must_read_info("数据分析文件已生成！")
        else:
            # template_details = deserialize_template_details(
            #     './data/template_details_low_temp.json')
            # create_excel_from_details(template_details, file_path)
            shutil.copy('./data/template_LT.xlsx', file_path)

            if hasattr(ui, 'mdf_distance'):
                write_ndarray_to_excel(ui.mdf_distance, 0, 'B2', file_path)
                write_ndarray_to_excel(
                    np.array(ui.start_time), 0, 'AW2', file_path)
                write_ndarray_to_excel(
                    np.array(ui.end_time), 0, 'AX2', file_path)
            if hasattr(ui, 'mdf_battenergy'):
                write_ndarray_to_excel(ui.mdf_battenergy, 0, 'D2', file_path)
            if hasattr(ui, 'mdf_dcdcenergy'):
                write_ndarray_to_excel(
                    ui.mdf_dcdcenergy[:, 1], 0, 'G2', file_path)
            if hasattr(ui, 'mdf_ecpenergy'):
                write_ndarray_to_excel(
                    ui.mdf_ecpenergy[:, 1], 0, 'H2', file_path)
            if hasattr(ui, 'mdf_ptcenergy'):
                write_ndarray_to_excel(
                    ui.mdf_ptcenergy[:, 1], 0, 'I2', file_path)
            if hasattr(ui, 'csv_battenergy'):
                write_ndarray_to_excel(ui.csv_battenergy, 0, 'J2', file_path)
            if hasattr(ui, 'csv_ipuenergy'):
                write_ndarray_to_excel(ui.csv_ipuenergy, 0, 'M2', file_path)
            if hasattr(ui, 'csv_ipsenergy'):
                write_ndarray_to_excel(ui.csv_ipsenergy, 0, 'P2', file_path)
            if hasattr(ui, 'csv_dcdchigh_energy'):
                write_ndarray_to_excel(
                    ui.csv_dcdchigh_energy, 0, 'S2', file_path)
            if hasattr(ui, 'csv_dcdclow_energy'):
                write_ndarray_to_excel(
                    ui.csv_dcdclow_energy, 0, 'V2', file_path)
            if hasattr(ui, 'dyno_distance'):
                write_ndarray_to_excel(ui.dyno_distance, 0, 'C2', file_path)
            if hasattr(ui, 'drive_shaft_energy'):
                write_ndarray_to_excel(
                    ui.drive_shaft_energy, 0, 'AE2', file_path)
            if hasattr(ui, 'resistance_energy'):
                write_ndarray_to_excel(
                    ui.resistance_energy, 0, 'AH2', file_path)
            if hasattr(ui, 'kinetic_energy'):
                write_ndarray_to_excel(ui.kinetic_energy, 0, 'AK2', file_path)
            if hasattr(ui, 'kinetic_energy_std'):
                write_ndarray_to_excel(
                    np.array([ui.kinetic_energy_std[0][1]]), 0, 'AN2', file_path)
            show_must_read_info("数据分析文件已生成！")
    mainWindow.setEnabled(True)


def save_as_callback(ui: Ui_MainWindow, mainWindow):

    if hasattr(ui, 'filtered_signals'):
        if hasattr(ui, 'mdf'):
            initial_path = os.path.dirname(ui.textEdit.toPlainText())
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save MDF File",
                initial_path,
                "MDF Files (*.mf4 *.mdf)"
            )
            if file_path:
                mainWindow.setEnabled(False)
                show_must_read_info("该过程耗时很长，请耐心等待！")
                QtCore.QTimer.singleShot(
                    100, lambda: save_mdf_as_filterd_signals(ui, file_path, mainWindow))
        else:
            show_must_read_info("请先导入MDF数据!")
    else:
        show_must_read_info("请先导入或生成信号配置文件!")


def save_log_as_segments(ui: Ui_MainWindow, mainWindow):

    logger = logging.getLogger('AppLogger')
    logger.debug('The save_log_as_segments method is called')

    if hasattr(ui, 'mdf'):
        original_mdf = ui.mdf
        initial_path = os.path.dirname(ui.textEdit.toPlainText())
        if hasattr(ui, 'start_time'):
            folder_path = QFileDialog.getExistingDirectory(
                None, "Select Folder", initial_path)
            if folder_path:
                progressDialog = QProgressDialog(
                    "正在生成文件，耗时较长，请耐心等待...", None, 0, 100, mainWindow)
                progressDialog.setWindowModality(
                    QtCore.Qt.WindowModality.WindowModal)
                progressDialog.setAutoClose(True)
                progressDialog.setCancelButton(None)
                progressDialog.setWindowTitle("Task Progress")
                progressDialog.resize(500, 60)
                progressDialog.show()
                progressDialog.setValue(1)
                QtCore.QTimer.singleShot(100, lambda: loop_for_save_log(
                    ui, progressDialog, original_mdf, folder_path))
        else:
            show_must_read_info("请先导入数据并运行计算程序!")
    else:
        show_must_read_info("请先导入数据!")


def save_power_as_segments(ui: Ui_MainWindow, mainWindow):

    logger = logging.getLogger('AppLogger')
    logger.debug('The save_power_as_segments method is called')

    if ui.listWidget.item(0):
        initial_path = os.path.dirname(ui.listWidget.item(0).text())
        if hasattr(ui, 'start_time_csv'):
            folder_path = QFileDialog.getExistingDirectory(
                None, "Select Folder", initial_path)
            if folder_path:
                progressDialog = QProgressDialog(
                    "正在生成文件，耗时较长，请耐心等待...", None, 0, 100, mainWindow)
                progressDialog.setWindowModality(
                    QtCore.Qt.WindowModality.WindowModal)
                progressDialog.setAutoClose(True)
                progressDialog.setCancelButton(None)
                progressDialog.setWindowTitle("Task Progress")
                progressDialog.resize(500, 60)
                progressDialog.show()
                progressDialog.setValue(1)
                QtCore.QTimer.singleShot(
                    100, lambda: loop_for_save_power(ui, progressDialog, folder_path))
        else:
            show_must_read_info("请先导入数据并运行计算程序!")
    else:
        show_must_read_info("请先导入数据!")


def save_dyno_as_segments(ui: Ui_MainWindow, mainWindow):

    logger = logging.getLogger('AppLogger')
    logger.debug('The save_dyno_as_segments method is called')

    if ui.textEdit_20.toPlainText():
        initial_path = os.path.dirname(ui.textEdit_20.toPlainText())
        if hasattr(ui, 'start_time_dyno'):
            folder_path = QFileDialog.getExistingDirectory(
                None, "Select Folder", initial_path)
            if folder_path:
                progressDialog = QProgressDialog(
                    "正在生成文件，耗时较长，请耐心等待...", None, 0, 100, mainWindow)
                progressDialog.setWindowModality(
                    QtCore.Qt.WindowModality.WindowModal)
                progressDialog.setAutoClose(True)
                progressDialog.setCancelButton(None)
                progressDialog.setWindowTitle("Task Progress")
                progressDialog.resize(500, 60)
                progressDialog.show()
                progressDialog.setValue(1)
                QtCore.QTimer.singleShot(
                    100, lambda: loop_for_save_dyno(ui, progressDialog, folder_path))
        else:
            show_must_read_info("请先导入数据并运行计算程序!")
    else:
        show_must_read_info("请先导入数据!")


def save_std_cycle_as_segment(ui: Ui_MainWindow):

    dyno_file_path = ui.textEdit_20.toPlainText()
    if dyno_file_path:
        initial_path = os.path.dirname(dyno_file_path)
    else:
        initial_path = os.getcwd()
    file_path, _ = QFileDialog.getSaveFileName(
        None,
        "Save File",
        initial_path,
        "MDF Files (*.mf4)"
    )
    if file_path:
        if not hasattr(ui, 'std_cycle'):
            selected_cycle = ui.cycleGrope.checkedAction()
            std_cycle_file = './data/' + selected_cycle.text() + '.mat'
            data = loadmat(std_cycle_file)
            ui.std_cycle = data

        std_cycle = ui.std_cycle
        signal_vehspd_std = Signal(
            samples=np.array(
                std_cycle['vehspd_trgt'], dtype=np.float64).reshape(-1, 1),
            timestamps=np.array(
                std_cycle['time'], dtype=np.float64).reshape(-1, 1),
            name='vehspd_std [km/h]'
        )
        signal_vehacc_std = Signal(
            samples=np.array(
                std_cycle['vehacc'], dtype=np.float64).reshape(-1, 1),
            timestamps=np.array(
                std_cycle['time'], dtype=np.float64).reshape(-1, 1),
            name='vehacc_std [m/s^2]'
        )
        signals_std = []
        signals_std.append(signal_vehspd_std)
        signals_std.append(signal_vehacc_std)
        mdf_std = MDF()
        mdf_std.append(signals_std)
        mdf_std.save(file_path, overwrite=True)
        show_must_read_info("已将标准工况生成MDF数据!")


def help_callback():

    show_must_read_info("功能建议、Bug反馈、问题咨询请联系开发者: 李凯2853")


def generate_filterd_signals_list(ui: Ui_MainWindow):

    if hasattr(ui, 'mdf'):
        group_name = ['BCS', 'BMC', 'VCU', 'DCU', 'HAVC', 'IPB', 'ITS', 'TPMS']
        all_signals = list(ui.mdf.channels_db.keys())
        matched_signals = [signal for signal in all_signals if any(
            signal.startswith(prefix) for prefix in group_name)]
        delete_signals = ['BMC_Cell', 'BMC_cell']
        filtered_signals = [signal for signal in matched_signals if not any(
            signal.startswith(ds) for ds in delete_signals)]
        ui.filtered_signals = filtered_signals
        serialize_to_json(
            filtered_signals, './signals list/default_signals_list.json')
        show_must_read_info("信号配置文件已生成!")
    else:
        show_must_read_info("请先导入MDF数据!")


def load_filterd_signals_list(ui: Ui_MainWindow):

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Open File",
        './signals list/',
        "JSON Files (*.json)"
    )
    if file_path:
        ui.filtered_signals = deserialize_from_json(file_path)


def del_cycle_manual(ui: Ui_MainWindow, mainWindow):

    cycle_list = [
        f'Cycle_{i+1}' for i in range(len(ui.cycle_start_time))]
    del_cycle, ok = QInputDialog.getItem(
        mainWindow, '手动删除循环', '请选择要删除的循环:', cycle_list, 0, False)
    if ok:
        ui.del_cycle_info = del_cycle


def set_correlation_coef(ui: Ui_MainWindow, mainWindow):
    coef, ok = QInputDialog.getDouble(
        mainWindow, "工况识别相关性系数", "请输入系数", ui.correlation_coef, 0, 1, 2)
    if ok:
        ui.correlation_coef = coef


def clear_attribute(ui: Ui_MainWindow, attribute_name):
    if hasattr(ui, attribute_name):
        delattr(ui, attribute_name)
        if attribute_name == 'add_cycle_info':
            show_must_read_info('已清除手动添加的循环!')
        else:
            show_must_read_info('已清除手动删除的循环!')
