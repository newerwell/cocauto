"""
实时游戏画面透视与鼠标交互框选/遥控画布 (Interactive Game Canvas Widget)

高性能改造：
1. 视频底图与识别标注完全分离：主线程仅在帧序号 seq 变化时将最新 BGR 帧转为 QPixmap（单次转换、无冗余整帧拷贝）；
2. OCR 资源框、气泡/按钮匹配框、战斗红线四边点与点击准星波纹均在 paintEvent 中使用 QPainter 按 1280x720 坐标矢量叠加，
   切换图层或清除标注 0 ms 立即响应，且绝不污染底层无损原图 `_raw_bgr`；
3. 鼠标实时坐标探针、点击直控模拟器、无损 PNG 拖拽框选严格共享同一套 1280x720 坐标系。
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QLabel

from .worker_threads import VisionOverlayData


def _bgr_to_qcolor(bgr: Tuple[int, int, int], alpha: int = 255) -> QColor:
    b, g, r = bgr
    return QColor(int(r), int(g), int(b), alpha)


class InteractiveGameCanvas(QLabel):
    """支持 30FPS 实时视频流、矢量识别标注叠加、鼠标坐标探针、点击直控与无损拖拽框选 ROI 的 1280x720 交互画布"""

    # 当用户用鼠标完成框选时发出信号: (template_key, (y1, y2, x1, x2), cropped_bgr)
    roiSelected = pyqtSignal(str, tuple, object)
    # 当用户在画布上移动鼠标时发出信号: (img_x, img_y, bgr_tuple)
    mouseMovedOnGame = pyqtSignal(int, int, tuple)
    # 当用户在直控模式下左键点击画布时发出信号: (img_x, img_y)
    canvasTapped = pyqtSignal(int, int)

    STANDARD_W = 1280
    STANDARD_H = 720

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(680, 382)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #090C14; border: 2px solid #2D3748; border-radius: 8px;")

        self._raw_bgr: Optional[np.ndarray] = None
        self._frame_seq: int = 0
        self._display_pixmap: Optional[QPixmap] = None

        self._overlay_data: Optional[VisionOverlayData] = None
        self._show_ocr: bool = True
        self._show_templates: bool = True
        self._show_red_zone: bool = False

        self._cropping_key: Optional[str] = None
        self._drag_start: Optional[QPoint] = None
        self._drag_current: Optional[QPoint] = None
        self._hover_game_pt: Optional[Tuple[int, int]] = None
        self._last_tap_marker: Optional[Tuple[int, int, float]] = None
        self.direct_tap_enabled: bool = False

        # 点击准星自动消退定时器（避免静帧时准星一直停留）
        self._tap_fade_timer = QTimer(self)
        self._tap_fade_timer.setSingleShot(True)
        self._tap_fade_timer.timeout.connect(self._clear_expired_tap_marker)

        self.set_placeholder_frame("正在连接模拟器开启 1280x720 实时画面流...")

    def set_placeholder_frame(self, message: str) -> None:
        canvas = np.zeros((self.STANDARD_H, self.STANDARD_W, 3), dtype=np.uint8)
        canvas[:, :] = (22, 28, 42)
        cv2.rectangle(canvas, (30, 30), (1250, 690), (65, 85, 120), 2)
        cv2.putText(
            canvas,
            "COC-AutoApp Real-Time Live Stream Canvas",
            (310, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 230, 180),
            2,
        )
        cv2.putText(
            canvas,
            "1280x720 H.264 Live Stream & Interactive Debugger Ready",
            (315, 375),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 200, 230),
            2,
        )
        self._overlay_data = None
        self.set_video_frame(canvas, seq=0, copy_raw=True)

    def _clear_expired_tap_marker(self) -> None:
        if self._last_tap_marker is not None and (time.time() - self._last_tap_marker[2]) > 1.2:
            self._last_tap_marker = None
            self.update()

    def show_tap_ripple(self, game_x: int, game_y: int) -> None:
        """在画面上记录一次点击准星反馈 (持续显示 1.2 秒)"""
        self._last_tap_marker = (game_x, game_y, time.time())
        self._tap_fade_timer.start(1250)
        self.update()

    def set_layer_visibility(self, show_ocr: bool, show_templates: bool, show_red_zone: bool) -> None:
        """动态切换各识别图层的显示状态，关闭图层时立即清除画布上的对应标注"""
        self._show_ocr = bool(show_ocr)
        self._show_templates = bool(show_templates)
        self._show_red_zone = bool(show_red_zone)
        self.update()

    def set_overlay_data(self, overlay: Optional[VisionOverlayData]) -> None:
        """更新最新一轮结构化识别结果，立即触发矢量重绘"""
        self._overlay_data = overlay
        self.update()

    def clear_overlay(self) -> None:
        """清空画布上的全部识别标注（切换设备、场景变化、停止流或进入框选时调用）"""
        if self._overlay_data is not None:
            self._overlay_data = None
            self.update()

    def start_interactive_crop(self, template_key: str) -> None:
        self._cropping_key = template_key
        self._drag_start = None
        self._drag_current = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def cancel_interactive_crop(self) -> None:
        self._cropping_key = None
        self._drag_start = None
        self._drag_current = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_video_frame(self, bgr_img: np.ndarray, seq: int = 0, copy_raw: bool = False) -> bool:
        """
        更新最新视频底图：
        - 若 seq > 0 且与当前已渲染帧序号一致，则跳过转换与重绘，返回 False；
        - 仅执行一次 BGR->RGB->QPixmap 转换，不污染 `_raw_bgr`。
        """
        if bgr_img is None or bgr_img.size == 0:
            return False
        if seq > 0 and seq == self._frame_seq:
            return False

        self._frame_seq = seq
        self._raw_bgr = bgr_img.copy() if copy_raw else bgr_img
        if hasattr(QImage, "Format_BGR888") and bgr_img.flags["C_CONTIGUOUS"]:
            h, w, ch = bgr_img.shape
            qimg = QImage(bgr_img.data, w, h, ch * w, QImage.Format_BGR888)
        else:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._display_pixmap = QPixmap.fromImage(qimg)
        self.update()
        return True

    def update_frame(self, bgr_img: np.ndarray, overlay_bgr: Optional[np.ndarray] = None) -> None:
        """兼容旧调用签名：直接更新画布图像"""
        if bgr_img is None or bgr_img.size == 0:
            return
        self._frame_seq = 0
        self._raw_bgr = bgr_img.copy()
        show_img = overlay_bgr if overlay_bgr is not None else bgr_img
        rgb = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._display_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def _widget_to_image_coords(self, pt: QPoint) -> Tuple[int, int]:
        rect = self._drawn_image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return 0, 0
        rel_x = (pt.x() - rect.x()) / float(rect.width())
        rel_y = (pt.y() - rect.y()) / float(rect.height())
        ix = int(np.clip(rel_x * self.STANDARD_W, 0, self.STANDARD_W - 1))
        iy = int(np.clip(rel_y * self.STANDARD_H, 0, self.STANDARD_H - 1))
        return ix, iy

    def _drawn_image_rect(self) -> QRect:
        W, H = self.width() - 8, self.height() - 8
        target_w = W
        target_h = int(W * self.STANDARD_H / self.STANDARD_W)
        if target_h > H:
            target_h = H
            target_w = int(H * self.STANDARD_W / self.STANDARD_H)
        x = (self.width() - target_w) // 2
        y = (self.height() - target_h) // 2
        return QRect(x, y, target_w, target_h)

    def mousePressEvent(self, event):
        if self._cropping_key and event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_current = event.pos()
            self.update()
        elif self._cropping_key and event.button() == Qt.RightButton:
            self.cancel_interactive_crop()
        elif self.direct_tap_enabled and event.button() == Qt.LeftButton:
            gx, gy = self._widget_to_image_coords(event.pos())
            self.show_tap_ripple(gx, gy)
            self.canvasTapped.emit(gx, gy)

    def mouseMoveEvent(self, event):
        gx, gy = self._widget_to_image_coords(event.pos())
        self._hover_game_pt = (gx, gy)
        if self._raw_bgr is not None:
            h, w = self._raw_bgr.shape[:2]
            px = min(max(gx, 0), w - 1)
            py = min(max(gy, 0), h - 1)
            b, g, r = [int(v) for v in self._raw_bgr[py, px]]
            self.mouseMovedOnGame.emit(gx, gy, (b, g, r))

        if self._cropping_key and self._drag_start is not None:
            self._drag_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._cropping_key and self._drag_start is not None and event.button() == Qt.LeftButton:
            p1 = self._widget_to_image_coords(self._drag_start)
            p2 = self._widget_to_image_coords(event.pos())
            x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
            y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
            key = self._cropping_key
            self.cancel_interactive_crop()

            if (x2 - x1) >= 8 and (y2 - y1) >= 8 and self._raw_bgr is not None:
                crop = self._raw_bgr[y1:y2, x1:x2].copy()
                self.roiSelected.emit(key, (y1, y2, x1, x2), crop)

    def _draw_vision_overlays(self, painter: QPainter, rect: QRect) -> None:
        overlay = self._overlay_data
        if overlay is None or rect.width() <= 0 or rect.height() <= 0:
            return

        sx = rect.width() / float(self.STANDARD_W)
        sy = rect.height() / float(self.STANDARD_H)
        ox = float(rect.x())
        oy = float(rect.y())

        label_font = QFont("Consolas", 9, QFont.Bold)
        painter.setFont(label_font)

        # 1. 气泡与按钮模板匹配框
        if self._show_templates and overlay.template_boxes:
            for box in overlay.template_boxes:
                color = _bgr_to_qcolor(box.bgr_color)
                painter.setPen(QPen(color, 2))
                painter.setBrush(Qt.NoBrush)
                r = QRectF(ox + box.x * sx, oy + box.y * sy, box.w * sx, box.h * sy)
                painter.drawRect(r)
                tx = ox + box.x * sx
                ty = max(oy + 12.0, oy + box.y * sy - 4.0)
                painter.drawText(QPointF(tx, ty), box.label)

        # 2. OCR 资源与工人 ROI 框
        if self._show_ocr and overlay.ocr_rois:
            for item in overlay.ocr_rois:
                y1, y2, x1, x2 = item.roi
                color = _bgr_to_qcolor(item.bgr_color)
                painter.setPen(QPen(color, 2))
                painter.setBrush(Qt.NoBrush)
                r = QRectF(ox + x1 * sx, oy + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
                painter.drawRect(r)
                tx = ox + x1 * sx
                ty = max(oy + 12.0, oy + y1 * sy - 4.0)
                painter.drawText(QPointF(tx, ty), item.label)

        # 3. 战斗红线菱形与四边下兵点
        if self._show_red_zone and overlay.red_zone_sides:
            if overlay.map_diamond is not None:
                poly = QPolygonF([QPointF(ox + pt[0] * sx, oy + pt[1] * sy) for pt in overlay.map_diamond])
                painter.setPen(QPen(QColor(130, 130, 130), 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(poly)

            side_colors = {
                "NW": (255, 220, 0),
                "NE": (0, 255, 255),
                "SW": (255, 100, 255),
                "SE": (0, 255, 120),
            }
            for s_name, pts in overlay.red_zone_sides.items():
                c = _bgr_to_qcolor(side_colors.get(s_name, (0, 255, 0)))
                for px, py in pts:
                    center = QPointF(ox + px * sx, oy + py * sy)
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    painter.setBrush(QBrush(c))
                    painter.drawEllipse(center, max(3.5, 5.0 * sx), max(3.5, 5.0 * sy))

    def _draw_tap_marker(self, painter: QPainter, rect: QRect) -> None:
        if self._last_tap_marker is None:
            return
        tx, ty, ts = self._last_tap_marker
        if (time.time() - ts) > 1.2:
            self._last_tap_marker = None
            return
        sx = rect.width() / float(self.STANDARD_W)
        sy = rect.height() / float(self.STANDARD_H)
        cx = rect.x() + tx * sx
        cy = rect.y() + ty * sy

        painter.setPen(QPen(QColor(120, 255, 0), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), 18.0 * sx, 18.0 * sy)
        painter.setPen(QPen(QColor(255, 255, 0), 1))
        painter.setBrush(QBrush(QColor(255, 255, 0)))
        painter.drawEllipse(QPointF(cx, cy), 5.0 * sx, 5.0 * sy)
        painter.setPen(QPen(QColor(120, 255, 0), 1))
        painter.drawLine(QPointF(cx - 24.0 * sx, cy), QPointF(cx + 24.0 * sx, cy))
        painter.drawLine(QPointF(cx, cy - 24.0 * sy), QPointF(cx, cy + 24.0 * sy))

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        rect = self._drawn_image_rect()
        if self._display_pixmap is not None:
            # 高帧率视频底图关闭昂贵的软件双线性 SmoothPixmapTransform，显著降低主线程绘制耗时
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            painter.drawPixmap(rect, self._display_pixmap)

        painter.setRenderHint(QPainter.Antialiasing, True)

        # 框选模式下隐藏识别标注，保证无损底图清晰可见；常规模式下即时矢量叠加
        if not self._cropping_key:
            self._draw_vision_overlays(painter, rect)
            self._draw_tap_marker(painter, rect)
        else:
            painter.fillRect(QRect(10, 10, self.width() - 20, 34), QColor(217, 119, 6, 220))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
            painter.drawText(
                QRect(10, 10, self.width() - 20, 34),
                Qt.AlignCenter,
                f"🎯 无损 PNG 冻结框选模式：请按住左键拖拽框选 [{self._cropping_key}] (右键取消)",
            )

            if self._drag_start and self._drag_current:
                pen = QPen(QColor(0, 255, 180), 2, Qt.DashLine)
                painter.setPen(pen)
                sel_rect = QRect(self._drag_start, self._drag_current).normalized()
                painter.drawRect(sel_rect)
                painter.fillRect(sel_rect, QColor(0, 255, 180, 45))
        painter.end()
