import os
import re
import json
import pickle
import threading
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QCheckBox, QSlider, QFileDialog, QListWidget, QListWidgetItem,
    QMessageBox, QDoubleSpinBox, QProgressBar, QGridLayout, QGraphicsEllipseItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QTimer, QPoint, QEvent

from app_state import AppState
from utils import (
    resize_if_needed, normalize_and_dilate,
    make_json_serializable
)



class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent_window):
        super().__init__(scene)
        self.parent_window = parent_window
        self.app_state = parent_window.app_state
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.blobs = []
        self.visible_colors = set()
        self.highlight_items = []
        self.union_dict = {}
        self.current_qimage = None

    def wheelEvent(self, event):
        cursor_pos = event.pos()
        scene_pos = self.mapToScene(cursor_pos)
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)
        mouse_centered = self.mapFromScene(scene_pos)
        delta = cursor_pos - mouse_centered
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        pos = self.mapToScene(event.pos())
        self.parent_window.update_mouse_coordinates(pos)
        self.parent_window.handle_hover(event, pos)

    def update_blobs(self, blobs, visible_colors):
        self.blobs = blobs
        self.visible_colors = visible_colors

    def highlight_selected_boxes(self, selected_items):
        for item in self.highlight_items:
            self.scene().removeItem(item)
        self.highlight_items.clear()
    
        for item in selected_items:
            text = item.toolTip()
            center_match = re.search(r"Center: \((\d+), (\d+)\)", text)
            length_match = re.search(r"Length: (\d+)\s*px", text)
    
            if center_match and length_match:
                x, y, length = int(center_match.group(1)), int(center_match.group(2)), int(length_match.group(1))
                radius = length / 2 + 5
                circle = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
                circle.setPen(QPen(QColor("yellow"), 2, Qt.SolidLine))
                circle.setZValue(100)
                self.scene().addItem(circle)
                self.highlight_items.append(circle)

class MainWindow(QWidget):
    def __init__(self, app_state):
        super().__init__()
        self.app_state = app_state
        self.setWindowTitle("X-AutoMap")
        self.setGeometry(100, 100, 1900, 1000)
        self._init_ui_elements()
        self._init_ui()
        self.blob_items = []
        self.union_box_items = []
        self.source_images = []
        self.norm_dilated = []

    def _init_ui_elements(self):
        self.file_list_widget = QListWidget()
        self.float_input_micron_x = QDoubleSpinBox()
        self.float_input_micron_y = QDoubleSpinBox()
        self.origin_x_input = QDoubleSpinBox()
        self.origin_y_input = QDoubleSpinBox()
        self.graphics_view = None
        self.pixmap_item = None
        self.checkboxes = {}
        self.sliders = {}
        self.area_sliders = {}
        self.slider_labels = {}
        self.area_slider_labels = {}
        self.union_list_widget = QListWidget()
        self.queue_server_list = QListWidget()
        self.union_checkbox = QCheckBox("Union Boxes")
        self.hover_label = QLabel(self)
        self.hover_label.setWindowFlags(Qt.ToolTip)
        self.custom_box_number = 1
        self.x_label = QLabel("X: 0")
        self.y_label = QLabel("Y: 0")
        self.x_micron_label = QLabel("X Real: 0")
        self.y_micron_label = QLabel("Y Real: 0")
        self.progress_bar = QProgressBar()

    def _init_ui(self):
        self.outer_layout = QVBoxLayout(self)
        self.setup_widget = self._create_setup_screen()
        self.outer_layout.addWidget(self.setup_widget)

    def _create_setup_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # --- Top buttons and label using QGridLayout ---
        top_grid = QGridLayout()
        
        dir_btn = QPushButton("Select Directory")
        dir_btn.clicked.connect(self.on_dir_selected)
        top_grid.addWidget(dir_btn, 0, 0)

        self.dir_label = QLabel("No directory selected.")
        top_grid.addWidget(self.dir_label, 0, 1, Qt.AlignLeft)

        load_backup_btn = QPushButton("Load Backup")
        load_backup_btn.clicked.connect(self.on_load_backup_clicked)
        top_grid.addWidget(load_backup_btn, 1, 0)
        
        top_grid.setColumnStretch(1, 1) # Allow column 1 to expand
        layout.addLayout(top_grid)

        self.file_list_widget.itemChanged.connect(self.update_selection)
        layout.addWidget(self.file_list_widget)
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("Microns per Pixel X:"), 0, 0)
        self.float_input_micron_x.setRange(0, 1000)
        self.float_input_micron_x.setValue(1.00)
        param_layout.addWidget(self.float_input_micron_x, 0, 1)
        param_layout.addWidget(QLabel("Microns per Pixel Y:"), 1, 0)
        self.float_input_micron_y.setRange(0, 1000)
        self.float_input_micron_y.setValue(1.00)
        param_layout.addWidget(self.float_input_micron_y, 1, 1)
        param_layout.addWidget(QLabel("True Origin X:"), 2, 0)
        self.origin_x_input.setRange(-10000, 10000)
        param_layout.addWidget(self.origin_x_input, 2, 1)
        param_layout.addWidget(QLabel("True Origin Y:"), 3, 0)
        self.origin_y_input.setRange(-10000, 10000)
        param_layout.addWidget(self.origin_y_input, 3, 1)
        layout.addLayout(param_layout)

        # --- Confirm button ---
        confirm_layout = QHBoxLayout()
        confirm_layout.addStretch()
        confirm_btn = QPushButton("Confirm and Load Images")
        confirm_btn.clicked.connect(self.on_confirm_clicked)
        confirm_layout.addWidget(confirm_btn)
        layout.addLayout(confirm_layout)

        return widget

    def on_load_backup_clicked(self):
        tiff_paths, _ = QFileDialog.getOpenFileNames(self, "Select 3 TIFF files", "", "TIFF Files (*.tif *.tiff)")
        if len(tiff_paths) != 3:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly 3 TIFF files.")
            return

        pkl_path, _ = QFileDialog.getOpenFileName(self, "Select precomputed_blobs.pkl file", "", "Pickle Files (*.pkl)")
        if not pkl_path:
            return

        self.app_state.microns_per_pixel_x = self.float_input_micron_x.value()
        self.app_state.microns_per_pixel_y = self.float_input_micron_y.value()
        self.app_state.true_origin_x = self.origin_x_input.value()
        self.app_state.true_origin_y = self.origin_y_input.value()

        self.app_state.img_paths = sorted(tiff_paths)
        self.app_state.file_names = [os.path.basename(p) for p in self.app_state.img_paths]
        self.app_state.element_colors = ['red', 'green', 'blue']
        self.app_state.thresholds = {color: 100 for color in self.app_state.element_colors}
        self.app_state.area_thresholds = {color: 200 for color in self.app_state.element_colors}

        try:
            with open(pkl_path, 'rb') as f:
                self.app_state.precomputed_blobs = pickle.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load pickle file: {e}")
            return

        self._init_analysis_gui(from_backup=True)

    def on_dir_selected(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory: return
        self.app_state.selected_directory = directory
        self.dir_label.setText(directory)
        self.app_state.file_paths = []
        self.app_state.selected_files_order = []
        self.file_list_widget.clear()
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(('.tif', '.tiff')):
                item = QListWidgetItem(fname)
                item.setCheckState(Qt.Unchecked)
                self.file_list_widget.addItem(item)
                self.app_state.file_paths.append(os.path.join(directory, fname))

    def update_selection(self, item):
        if self.app_state.selected_files_order is None:
            self.app_state.selected_files_order = []

        checked_indices = [i for i in range(self.file_list_widget.count()) if self.file_list_widget.item(i).checkState() == Qt.Checked]

        self.app_state.selected_files_order = [idx for idx in self.app_state.selected_files_order if idx in checked_indices]

        for i in checked_indices:
            if i not in self.app_state.selected_files_order:
                if len(self.app_state.selected_files_order) < 3:
                    self.app_state.selected_files_order.append(i)
                else:
                    item.setCheckState(Qt.Unchecked)

    def on_confirm_clicked(self):
        self.app_state.microns_per_pixel_x = self.float_input_micron_x.value()
        self.app_state.microns_per_pixel_y = self.float_input_micron_y.value()
        self.app_state.true_origin_x = self.origin_x_input.value()
        self.app_state.true_origin_y = self.origin_y_input.value()
        
        if len(self.app_state.selected_files_order) != 3:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly 3 TIFF files.")
            return
        
        self.app_state.img_paths = [self.app_state.file_paths[i] for i in self.app_state.selected_files_order]
        self.app_state.file_names = [os.path.basename(p) for p in self.app_state.img_paths]
        self.app_state.element_colors = ['red', 'green', 'blue']
        self.app_state.thresholds = {color: 100 for color in self.app_state.element_colors}
        self.app_state.area_thresholds = {color: 200 for color in self.app_state.element_colors}
        self._init_analysis_gui()

    def _init_analysis_gui(self, from_backup=False):
        self.setup_widget.setParent(None)
        main_widget = QWidget()
        self.main_layout = QHBoxLayout(main_widget)
        
        left_panel = QVBoxLayout()
        self._create_image_view_panel()
        left_panel.addWidget(self.graphics_view)
        
        self.main_layout.addLayout(left_panel)

        self._create_controls_panel()
        self.outer_layout.addWidget(main_widget)
        
        self.progress_bar.setParent(None)
        self.outer_layout.addWidget(self.progress_bar)

        if from_backup:
            self.progress_bar.hide()
            self.update_boxes()
        else:
            QTimer.singleShot(100, self._start_blob_computation)


    def _create_image_view_panel(self):
        img_r, img_g, img_b = [tiff.imread(p).astype(np.float32) for p in self.app_state.img_paths]
        shapes = [img.shape for img in (img_r, img_g, img_b)]
        self.app_state.target_shape = Counter(shapes).most_common(1)[0][0]
        
        img_r = resize_if_needed(img_r, self.app_state.file_names[0], self.app_state.target_shape)
        img_g = resize_if_needed(img_g, self.app_state.file_names[1], self.app_state.target_shape)
        img_b = resize_if_needed(img_b, self.app_state.file_names[2], self.app_state.target_shape)
        
        self.source_images = [img_r, img_g, img_b]
        self.norm_dilated = [normalize_and_dilate(im) for im in self.source_images]
        
        merged_rgb = cv2.merge([nd[0] for nd in self.norm_dilated])
        scene = QGraphicsScene()
        q_img = QImage(merged_rgb.data, merged_rgb.shape[1], merged_rgb.shape[0], merged_rgb.shape[1] * 3, QImage.Format_RGB888)
        
        self.graphics_view = ZoomableGraphicsView(scene, self)
        self.graphics_view.current_qimage = q_img
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
        scene.addItem(self.pixmap_item)

    def _create_controls_panel(self):
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Exit/Reset
        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(lambda: self.graphics_view.resetTransform())
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(exit_btn)
        controls_layout.addLayout(button_layout)

        # Lists and buttons
        self.union_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.union_list_widget.itemSelectionChanged.connect(self.on_union_item_selected)
        
        send_to_list_btn = QPushButton("Add to list")
        send_to_list_btn.clicked.connect(self.send_to_list)
        get_elements_btn = QPushButton("Get all elements")
        get_elements_btn.clicked.connect(self.get_elements_list)
        union_btn = QPushButton("Get unions")
        union_btn.clicked.connect(self.union_function)
        add_box_btn = QPushButton("Add Box")
        add_box_btn.clicked.connect(self.add_box)

        union_list_layout = QVBoxLayout()
        union_list_layout.addWidget(self.union_list_widget)
        union_list_layout.addWidget(send_to_list_btn)
        union_list_layout.addWidget(get_elements_btn)
        union_list_layout.addWidget(union_btn)
        union_list_layout.addWidget(add_box_btn)

        send_to_queue_btn = QPushButton("Send to Queue Server")
        send_to_queue_btn.clicked.connect(self.send_to_queue_server)
        clear_queue_btn = QPushButton("Clear")
        clear_queue_btn.clicked.connect(self.clear_queue_server_list)

        queue_list_layout = QVBoxLayout()
        queue_list_layout.addWidget(self.queue_server_list)
        queue_list_layout.addWidget(send_to_queue_btn)
        queue_list_layout.addWidget(clear_queue_btn)

        dual_list_layout = QHBoxLayout()
        dual_list_layout.addLayout(union_list_layout)
        dual_list_layout.addLayout(queue_list_layout)
        controls_layout.addLayout(dual_list_layout)

        # Coordinates
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(self.x_label)
        coord_layout.addWidget(self.y_label)
        coord_layout.addWidget(self.x_micron_label)
        coord_layout.addWidget(self.y_micron_label)
        controls_layout.addLayout(coord_layout)

        # Legend
        legend_layout = QHBoxLayout()
        legend_label = QLabel("Legend")
        legend_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        legend_layout.addWidget(legend_label)
        for i, color in enumerate(self.app_state.element_colors):
            cb = QCheckBox(self.app_state.file_names[i])
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {color}")
            cb.stateChanged.connect(self.update_boxes)
            self.checkboxes[color] = cb
            legend_layout.addWidget(cb)

        self.union_checkbox.setChecked(True)
        self.union_checkbox.setStyleSheet("color: black")
        self.union_checkbox.stateChanged.connect(self.update_boxes)
        legend_layout.addWidget(self.union_checkbox)
        self.checkboxes['union'] = self.union_checkbox
        legend_layout.addStretch()
        controls_layout.addLayout(legend_layout)

        # Sliders
        slider_layout = QHBoxLayout()
        for color in self.app_state.element_colors:
            i = self.app_state.element_colors.index(color)
            vbox = QVBoxLayout()
            label = QLabel(f"{self.app_state.file_names[i]}_threshold: {self.app_state.thresholds[color]}")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 255)
            slider.setTickInterval(10)
            slider.setValue(self.app_state.thresholds[color])
            slider.setTickPosition(QSlider.TicksBelow)
            slider.valueChanged.connect(lambda val, c=color: self.on_slider_change(val, c))
            self.sliders[color] = slider
            self.slider_labels[color] = label
            vbox.addWidget(label)
            vbox.addWidget(slider)
            slider_layout.addLayout(vbox)
        controls_layout.addLayout(slider_layout)

        # Area Sliders
        area_slider_layout = QHBoxLayout()
        for color in self.app_state.element_colors:
            i = self.app_state.element_colors.index(color)
            vbox = QVBoxLayout()
            label = QLabel(f"{self.app_state.file_names[i]}_min_area: {self.app_state.area_thresholds[color]}")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(10, 400)
            slider.setTickInterval(10)
            slider.setValue(self.app_state.area_thresholds[color])
            slider.setTickPosition(QSlider.TicksBelow)
            slider.valueChanged.connect(lambda val, c=color: self.on_area_slider_change(val, c))
            self.area_sliders[color] = slider
            self.area_slider_labels[color] = label
            vbox.addWidget(label)
            vbox.addWidget(slider)
            area_slider_layout.addLayout(vbox)
        controls_layout.addLayout(area_slider_layout)

        controls_widget.setLayout(controls_layout)
        self.main_layout.addWidget(controls_widget)

    def _start_blob_computation(self):
        self.app_state.precomputed_blobs = {color: {} for color in self.app_state.element_colors}
        self.app_state.current_iteration = 0
        
        thresholds_range = list(range(0, 256, 10))
        area_range = list(range(10, 501, 10))
        total_iterations = len(self.app_state.element_colors) * len(thresholds_range) * len(area_range)
        
        self.progress_bar.setRange(0, total_iterations)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Computing blobs... %p%")
        self.progress_bar.show()

        threading.Thread(target=self._blob_computation_thread, args=(thresholds_range, area_range), daemon=True).start()

    def _blob_computation_thread(self, thresholds_range, area_range):
        dilated_imgs = [nd[1] for nd in self.norm_dilated]
        img_r, img_g, img_b = self.source_images
        
        for i, color in enumerate(self.app_state.element_colors):
            for t_val in thresholds_range:
                for a_val in area_range:
                    blobs = self._detect_blobs(
                        dilated_imgs[i],
                        [img_r, img_g, img_b][i],
                        t_val, a_val, color,
                        self.app_state.file_names[i]
                    )
                    self.app_state.precomputed_blobs[color][(t_val, a_val)] = blobs
                    self.app_state.current_iteration += 1
                    QApplication.instance().postEvent(self, UpdateProgressEvent(self.app_state.current_iteration))
        
        QApplication.instance().postEvent(self, ComputationFinishedEvent())

    def customEvent(self, event):
        if isinstance(event, UpdateProgressEvent):
            self.progress_bar.setValue(event.value)
        elif isinstance(event, ComputationFinishedEvent):
            self.progress_bar.hide()
            if self.app_state.selected_directory and self.app_state.precomputed_blobs:
                output_path = Path(self.app_state.selected_directory) / "precomputed_blobs.pkl"
                with open(output_path, "wb") as f:
                    pickle.dump(self.app_state.precomputed_blobs, f)
            self.update_boxes()


    def _detect_blobs(self, img_norm, img_orig, min_thresh, min_area, color, file_name):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = min_thresh
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = 50000
        params.thresholdStep = 5
        params.filterByColor = False
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        params.minRepeatability = 1
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img_norm)
        blobs = []

        for idx, kp in enumerate(keypoints, start=1):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            box_size = 2 * radius
            box_x, box_y = x - radius, y - radius

            x1, y1 = max(0, box_x), max(0, box_y)
            x2, y2 = min(img_orig.shape[1], x + radius), min(img_orig.shape[0], y + radius)
            roi_orig = img_orig[y1:y2, x1:x2]
            roi_dilated = img_norm[y1:y2, x1:x2]

            if roi_orig.size > 0:
                blobs.append({
                    'Box': f"{file_name} Box #{idx}",
                    'center': (x, y), 'radius': radius, 'color': color, 'file': file_name,
                    'max_intensity': roi_orig.max(), 'mean_intensity': roi_orig.mean(),
                    'mean_dilation': float(roi_dilated.mean()),
                    'box_x': box_x, 'box_y': box_y, 'box_size': box_size
                })
        return blobs

    def update_boxes(self):
        selected_colors = {c for c, cb in self.checkboxes.items() if cb.isChecked() and c != 'union'}
        blobs = self.get_current_blobs()
        self.graphics_view.update_blobs(blobs, selected_colors)
        self.redraw_boxes(blobs, selected_colors)

    def get_current_blobs(self):
        blobs = []
        if not self.app_state.precomputed_blobs:
            return blobs

        for color in self.app_state.element_colors:
            if color in self.app_state.thresholds and color in self.app_state.area_thresholds:
                threshold = self.app_state.thresholds[color]
                area = self.app_state.area_thresholds[color]
                key = (threshold, area)
                if color in self.app_state.precomputed_blobs and key in self.app_state.precomputed_blobs[color]:
                    blobs.extend(self.app_state.precomputed_blobs[color][key])
        return blobs

    def redraw_boxes(self, blobs, selected_colors):
        for item in self.blob_items:
            self.graphics_view.scene().removeItem(item)
        self.blob_items.clear()

        for blob in blobs:
            if blob['color'] in selected_colors:
                pen = QPen(QColor(blob['color']), 2)
                rect_item = self.graphics_view.scene().addRect(blob['box_x'], blob['box_y'], blob['box_size'], blob['box_size'], pen)
                self.blob_items.append(rect_item)

        for item in self.union_box_items:
            self.graphics_view.scene().removeItem(item)
        self.union_box_items.clear()

        if self.union_checkbox.isChecked():
            for idx, ub in self.graphics_view.union_dict.items():
                cx, cy = ub['center']
                length = ub['length']
                pen = QPen(QColor("white"), 2, Qt.DashLine)
                rect_item = self.graphics_view.scene().addRect(cx - length / 2, cy - length / 2, length, length, pen)
                self.union_box_items.append(rect_item)

    def update_mouse_coordinates(self, pos):
        x, y = int(pos.x()), int(pos.y())
        self.x_label.setText(f"X: {x}")
        self.y_label.setText(f"Y: {y}")
        self.x_micron_label.setText(f"X Real (µm): {(x * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x:.2f}")
        self.y_micron_label.setText(f"Y Real (µm): {(y * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y:.2f}")

    def handle_hover(self, event, scene_pos):
        x, y = int(scene_pos.x()), int(scene_pos.y())
        
        for blob in self.graphics_view.blobs:
            if blob['color'] not in self.graphics_view.visible_colors: continue
            cx, cy = blob['center']
            r = blob['radius']
            if abs(x - cx) <= r and abs(y - cy) <= r:
                self._show_tooltip(event, self._format_blob_tooltip(blob))
                return
        
        if self.union_checkbox.isChecked():
            for idx, ub in self.graphics_view.union_dict.items():
                cx, cy = ub['center']
                length = ub['length']
                rect = QRect(
                    int(cx - length / 2),
                    int(cy - length / 2),
                    int(length),
                    int(length)
                )
                if rect.contains(scene_pos.toPoint()):
                    self._show_tooltip(event, self._format_union_tooltip(ub, idx))
                    return

        self.hover_label.hide()

    def _show_tooltip(self, event, html):
        self.hover_label.setText(html)
        self.hover_label.adjustSize()
        mouse_pos = self.graphics_view.mapTo(self, event.pos())
        new_pos = QPoint(mouse_pos.x() + 20, mouse_pos.y() - self.hover_label.height() - 10)
        self.hover_label.move(new_pos)
        self.hover_label.show()

    def _format_blob_tooltip(self, blob):
        cx, cy = blob['center']
        real_cx = (cx * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x
        real_cy = (cy * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y
        real_w = blob['box_size'] * self.app_state.microns_per_pixel_x
        real_h = blob['box_size'] * self.app_state.microns_per_pixel_y
        return (
            f"<b>{blob['Box']}</b><br>"
            f"Center: ({cx}, {cy})<br>"
            f"Length: {blob['box_size']} px<br>"
            f"Area: {blob['box_size']**2} px²<br><br>"
            f"Real Center: ({real_cx:.2f}, {real_cy:.2f}) µm<br>"
            f"Real Size: {real_w:.2f} × {real_h:.2f} µm<br>"
            f"Real Area: {real_w * real_h:.2f} µm²<br><br>"
            f"Max intensity: {blob['max_intensity']:.3f}"
        )

    def _format_union_tooltip(self, ub, idx):
        return (
            f"<b>Union Box #{idx}</b><br>"
            f"Center: ({ub['center'][0]}, {ub['center'][1]})<br>"
f"Length: {ub['length']} px<br>"
            f"Area: {ub['area']} px²<br><br>"
            f"Real Center: ({ub['real_center'][0]:.2f}, {ub['real_center'][1]:.2f}) µm<br>"
            f"Real Size: {ub['real_size'][0]:.2f} × {ub['real_size'][1]:.2f} µm<br>"
            f"Real Area: {ub['real_area']:.2f} µm²"
        )

    def on_slider_change(self, value, color):
        snapped = round(value / 10) * 10
        if self.app_state.thresholds[color] != snapped:
            self.app_state.thresholds[color] = snapped
            self.sliders[color].blockSignals(True)
            self.sliders[color].setValue(snapped)
            self.sliders[color].blockSignals(False)
            self.slider_labels[color].setText(f"{self.checkboxes[color].text()}_threshold: {snapped}")
            self.update_boxes()

    def on_area_slider_change(self, value, color):
        snapped = round(value / 10) * 10
        if self.app_state.area_thresholds[color] != snapped:
            self.app_state.area_thresholds[color] = snapped
            self.area_sliders[color].blockSignals(True)
            self.area_sliders[color].setValue(snapped)
            self.area_sliders[color].blockSignals(False)
            self.area_slider_labels[color].setText(f"{self.checkboxes[color].text()}_min_area: {snapped}")
            self.update_boxes()

    def on_union_item_selected(self):
        selected_items = self.union_list_widget.selectedItems()
        self.graphics_view.highlight_selected_boxes(selected_items)

    def add_box(self):
        QMessageBox.information(self, "Add Union Box", "Click and drag to define a new union box.")

        self.original_mouse_press_event = self.graphics_view.mousePressEvent
        self.original_mouse_release_event = self.graphics_view.mouseReleaseEvent
        
        temp_state = {'start': None}

        def on_press(event):
            if event.button() == Qt.LeftButton:
                temp_state['start'] = self.graphics_view.mapToScene(event.pos()).toPoint()
            else:
                self.original_mouse_press_event(event)

        def on_release(event):
            if event.button() != Qt.LeftButton or temp_state['start'] is None:
                self.original_mouse_release_event(event)
                return

            end = self.graphics_view.mapToScene(event.pos()).toPoint()
            start = temp_state['start']
            temp_state['start'] = None

            self.graphics_view.mousePressEvent = self.original_mouse_press_event
            self.graphics_view.mouseReleaseEvent = self.original_mouse_release_event

            x1, y1 = start.x(), start.y()
            x2, y2 = end.x(), end.y()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = max(abs(x2 - x1), abs(y2 - y1))
            area = length * length

            real_cx = (cx * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x
            real_cy = (cy * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y
            real_length_x = length * self.app_state.microns_per_pixel_x
            real_length_y = length * self.app_state.microns_per_pixel_y
            real_area = real_length_x * real_length_y
            
            new_union = {
                'center': (cx, cy),
                'length': length,
                'area': area,
                'real_center': (real_cx, real_cy),
                'real_size': (real_length_x, real_length_y),
                'real_area': real_area,
            }

            next_index = max(self.graphics_view.union_dict.keys(), default=0) + 1
            self.graphics_view.union_dict[next_index] = new_union

            item_text = f"Custom Box #{self.custom_box_number}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, self.custom_box_number)
            item.setToolTip(self._format_union_tooltip(new_union, self.custom_box_number))
            self.union_list_widget.addItem(item)
            
            self.custom_box_number += 1

            if self.app_state.selected_directory:
                output_path = Path(self.app_state.selected_directory) / "union_blobs.json"
                serializable_dict = make_json_serializable(self.graphics_view.union_dict)
                with open(output_path, "w") as f:
                    json.dump(serializable_dict, f, indent=4)

            self.update_boxes()

        self.graphics_view.mousePressEvent = on_press
        self.graphics_view.mouseReleaseEvent = on_release

    def union_function(self):
        blobs = self.get_current_blobs()
        blobs_by_color = {color: [] for color in self.app_state.element_colors}
        for blob in blobs:
            blobs_by_color[blob['color']].append(blob)

        union_objects = {}
        union_index = 1
        
        reds = blobs_by_color.get('red', [])
        greens = blobs_by_color.get('green', [])
        blues = blobs_by_color.get('blue', [])
    
        for r in reds:
            for g in greens:
                if not self._boxes_intersect(r, g): continue
                for b in blues:
                    if self._boxes_intersect(r, b) and self._boxes_intersect(g, b):
                        cx, cy = self._union_center(r, g, b)
                        length, area = self._union_box_dimensions(r, g, b)
                        
                        real_cx = (cx * self.app_state.microns_per_pixel_x) + self.app_state.true_origin_x
                        real_cy = (cy * self.app_state.microns_per_pixel_y) + self.app_state.true_origin_y
                        real_length_x = length * self.app_state.microns_per_pixel_x
                        real_length_y = length * self.app_state.microns_per_pixel_y
                        
                        union_objects[union_index] = {
                            'center': (cx, cy), 'length': length, 'area': area,
                            'real_center': (real_cx, real_cy),
                            'real_size': (real_length_x, real_length_y),
                            'real_area': real_length_x * real_length_y
                        }
                        union_index += 1
        
        self.graphics_view.union_dict = union_objects
        self.union_list_widget.clear()
        for idx, ub in union_objects.items():
            item = QListWidgetItem(f"Union Box #{idx}")
            item.setToolTip(self._format_union_tooltip(ub, idx))
            self.union_list_widget.addItem(item)

        if self.app_state.selected_directory:
            output_path = Path(self.app_state.selected_directory) / "union_blobs.json"
            serializable_dict = make_json_serializable(self.graphics_view.union_dict)
            with open(output_path, "w") as f:
                json.dump(serializable_dict, f, indent=4)
        
        self.update_boxes()

    def _boxes_intersect(self, b1, b2):
        x1_min, y1_min = b1['box_x'], b1['box_y']
        x1_max = x1_min + b1['box_size']
        y1_max = y1_min + b1['box_size']
        x2_min, y2_min = b2['box_x'], b2['box_y']
        x2_max = x2_min + b2['box_size']
        y2_max = y2_min + b2['box_size']
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

    def _union_center(self, b1, b2, b3):
        x_vals = [b1['center'][0], b2['center'][0], b3['center'][0]]
        y_vals = [b1['center'][1], b2['center'][1], b3['center'][1]]
        return (sum(x_vals) // 3, sum(y_vals) // 3)

    def _union_box_dimensions(self, b1, b2, b3):
        xs = [b['box_x'] for b in [b1, b2, b3]]
        ys = [b['box_y'] for b in [b1, b2, b3]]
        sizes = [b['box_size'] for b in [b1, b2, b3]]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(x + s for x, s in zip(xs, sizes))
        max_y = max(y + s for y, s in zip(ys, sizes))
        length = max(max_x - min_x, max_y - min_y)
        return length, length**2

    def send_to_list(self):
        existing_texts = {self.queue_server_list.item(i).text() for i in range(self.queue_server_list.count())}
        for item in self.union_list_widget.selectedItems():
            if item.text() not in existing_texts:
                new_item = QListWidgetItem(item.text())
                new_item.setToolTip(item.toolTip())
                self.queue_server_list.addItem(new_item)

    def get_elements_list(self):
        self.union_list_widget.clear()
        all_blobs = self.get_current_blobs()
        for blob in all_blobs:
            item = QListWidgetItem(blob['Box'])
            item.setToolTip(self._format_blob_tooltip(blob))
            self.union_list_widget.addItem(item)

    def send_to_queue_server(self):
        # This requires interaction with a queue server, which is out of scope for the GUI implementation.
        QMessageBox.information(self, "Queue Server", "Sending data to queue server is not implemented in this version.")
        pass

    def clear_queue_server_list(self):
        self.queue_server_list.clear()

class UpdateProgressEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.User + 1)
    def __init__(self, value):
        super().__init__(self.EVENT_TYPE)
        self.value = value

class ComputationFinishedEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.User + 2)
    def __init__(self):
        super().__init__(self.EVENT_TYPE)