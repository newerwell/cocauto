"""Red-line and terrain checks for the current battle frame."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np


Point = Tuple[int, int]


def is_battle_frame(frame: np.ndarray | None) -> bool:
    return frame is not None and frame.shape == (720, 1280, 3) and frame.dtype == np.uint8


@dataclass
class DeploymentArea:
    clearance: np.ndarray
    side_span: tuple[int, int] | None = None

    def accepts(self, point: tuple[float, float]) -> bool:
        x, y = map(round, point)
        h, w = self.clearance.shape
        return 0 <= x < w and 0 <= y < h and self.clearance[y, x] >= 7

    def sample(self, hints: List[Point], eligible: np.ndarray | None = None) -> List[Point]:
        """Find a nearby interior point for each tactical hint."""
        h, w = self.clearance.shape
        points = []
        for x, y in hints:
            radius = 88
            left, top = max(0, x - radius), max(0, y - radius)
            right, bottom = min(w, x + radius + 1), min(h, y + radius + 1)
            region = self.clearance[top:bottom, left:right]
            available = region >= 7
            if eligible is not None:
                available &= eligible[top:bottom, left:right]
            rows, cols = np.nonzero(available)
            if not len(rows):
                continue
            dx, dy = cols + left - x, rows + top - y
            score = np.minimum(region[rows, cols], 24) - 0.65 * np.hypot(dx, dy)
            best = int(np.argmax(score))
            point = int(cols[best] + left), int(rows[best] + top)
            if point not in points:
                points.append(point)
        return points

    def sample_side(self, span: tuple[int, int], eligible: np.ndarray, fractions: tuple[float, ...]) -> List[Point]:
        rows, cols = np.nonzero(eligible & (self.clearance >= 8))
        points = []
        if cols.size:
            left, right = span
            clearance = np.minimum(self.clearance[rows, cols], 20)
            for fraction in fractions:
                target = left + (right - left) * fraction
                best = np.argmax(clearance - 0.18 * np.abs(cols - target))
                point = int(cols[best]), int(rows[best])
                if point not in points:
                    points.append(point)
        return points


class RedZoneDetector:
    """用现帧红线、地表和 HUD 约束候选点。"""

    # 战场有效可点击多边形 (严格避开顶部信息栏、举报按钮、右上资源栏、左下放弃按钮 [18..162, 508..568]、底部卡牌栏 Y>=575、右下摧毁率/下一个按钮)
    PLAYABLE_SAFE_POLY = np.array([
        [440, 42],    # North-West Top (避开左上战利品与举报按钮)
        [780, 22],    # North-East Top
        [988, 215],   # Upper-Right (避开右上资源条 X>990, Y<215)
        [1175, 245],  # East Upper Apex (移镜至底部后 3点钟顶点位置，低于右上 Y=215 资源栏)
        [1168, 438],  # East Lower Apex (移镜至顶部后 3点钟顶点位置，高于 Y=485 的摧毁率框)
        [1030, 472],  # Lower-Right (避开右下 [摧毁率] X>1035, Y>480)
        [750, 562],   # South-East Bottom (移镜后右下外圈底端，高于底部卡牌栏 Y=575)
        [480, 562],   # South-West Bottom (移镜后左下外圈底端，避开左下 [放弃] X<170, Y>500)
        [170, 495],   # Above [放弃] button
        [82, 420],    # West Lower Apex (移镜至顶部后 9点钟顶点位置)
        [82, 245],    # West Upper Apex (移镜至底部后 9点钟顶点位置，低于左上 Y=235 的战利品文字区)
        [185, 225],   # Upper-Left (避开左上战利品文字区)
    ], dtype=np.int32)

    # 下兵前将视角转移到下兵一侧的定向滑动坐标对: side -> (swipe_start, swipe_end)
    # 原理：向目标侧的反方向拖拽地图，利用《部落冲突》全景边界阻尼将目标侧外圈草地完整拉入屏幕开阔区
    CAMERA_FOCUS_SWIPES: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
        "NW": ((300, 185), (900, 495)),  # 将左上侧 (10~11点钟) 外圈草地拉入左上开阔区
        "NE": ((980, 185), (380, 495)),  # 将右上侧 (1~2点钟) 外圈草地拉入右上开阔区
        "SW": ((300, 495), (900, 185)),  # 将左下侧 (7~8点钟) 海岸外圈草地从底部选兵栏后方拉出
        "SE": ((980, 495), (380, 185)),  # 将右下侧 (4~5点钟) 森林外圈草地从底部选兵栏与摧毁率框后方拉出
    }

    # 移镜聚焦后各侧对应的阵型中心坐标 (用于计算背离阵型中心的径向向外单位向量与法术内圈补偿)
    SHIFTED_MAP_CENTERS: Dict[str, Tuple[int, int]] = {
        "NW": (635, 435),
        "NE": (635, 435),
        "SW": (635, 255),
        "SE": (635, 255),
    }

    # 移镜后的主搜索折线。
    SHIFTED_OUTER_PERIMETER_POLYLINES: Dict[str, List[Tuple[int, int]]] = {
        "NW": [(145, 408), (255, 326), (365, 244), (475, 162), (585, 80)],
        "NE": [(685, 80), (795, 162), (905, 244), (1015, 326), (1120, 405)],
        "SW": [(145, 280), (235, 348), (325, 415), (415, 482), (495, 542)],
        "SE": [(1120, 280), (1030, 348), (940, 415), (850, 482), (770, 542)],
    }

    # 127.0.0.1:5561 实机移镜后像素级标定：四侧外圈【第二层：径向安全双保险折线】（与主线间隔 ~14px 构成双层外圈下兵带）
    SHIFTED_OUTER_GUARD_POLYLINES: Dict[str, List[Tuple[int, int]]] = {
        "NW": [(133, 396), (243, 314), (353, 232), (463, 150), (573, 68)],
        "NE": [(697, 68), (807, 150), (917, 232), (1027, 314), (1132, 393)],
        "SW": [(156, 270), (246, 338), (336, 405), (426, 472), (506, 532)],
        "SE": [(1109, 270), (1019, 338), (929, 405), (839, 472), (759, 532)],
    }

    # 未移镜时供诊断预览使用的搜索折线。
    OUTER_PERIMETER_POLYLINES: Dict[str, List[Tuple[int, int]]] = {
        "NW": [(110, 328), (195, 258), (280, 188), (365, 118), (450, 48)],
        "NE": [(768, 24), (860, 94), (952, 164), (1045, 234), (1138, 305)],
        "SW": [(96, 352), (162, 404), (228, 456), (295, 488), (355, 515)],
        "SE": [(1155, 348), (1095, 398), (1030, 450), (965, 502), (900, 548)],
    }

    MAP_DIAMOND = np.array([[635, 36], [1165, 345], [635, 565], [92, 345]], dtype=np.int32)
    MAP_CENTER = (635, 315)

    @classmethod
    def get_camera_focus_swipe(cls, side_name: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """获取将战场视角转移到指定下兵侧 ('NW', 'NE', 'SW', 'SE') 的滑动起止坐标"""
        return cls.CAMERA_FOCUS_SWIPES.get(side_name, cls.CAMERA_FOCUS_SWIPES["NW"])

    @classmethod
    def _sample_polyline(cls, poly: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
        """沿折线按弧长等距插值采样 n 个点"""
        if n <= 0 or not poly:
            return []
        pts = [np.array(p, dtype=float) for p in poly]
        segs = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
        total_len = sum(segs)
        if n <= 1 or total_len <= 0:
            return [(int(pts[0][0]), int(pts[0][1]))]
        res: List[Tuple[int, int]] = []
        for k in range(n):
            target = total_len * k / (n - 1)
            acc = 0.0
            for i, slen in enumerate(segs):
                if acc + slen >= target or i == len(segs) - 1:
                    t = (target - acc) / max(1e-6, slen)
                    t = min(1.0, max(0.0, t))
                    p = pts[i] + t * (pts[i + 1] - pts[i])
                    res.append((int(round(p[0])), int(round(p[1]))))
                    break
                acc += slen
        return res

    def _red_mask(self, hsv: np.ndarray, max_hue: int) -> np.ndarray:
        red = cv2.inRange(hsv, (0, 105, 108), (max_hue, 255, 255))
        red |= cv2.inRange(hsv, (168, 105, 115), (180, 255, 255))
        # Preserve contours behind conservative HUD insets. Only terrain inside
        # the playable polygon can later be considered for a touch.
        red[582:] = 0
        red[:210, :180] = 0
        red[:170, 990:] = 0
        red[445:582, :170] = 0
        red[480:, 1030:] = 0
        return red

    @staticmethod
    def _close_bottom_cut(lines: np.ndarray) -> None:
        # A continuous trace reaching the card-bar divider at both ends
        # can be closed along that divider, which is excluded from clicks.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(lines)
        for label in range(1, count):
            x, y, w, h, _ = stats[label]
            if w < 150 or h < 70 or y + h < 578:
                continue
            hits = np.where(np.any(labels[576:582, x:x+w] == label, axis=0))[0]
            if hits.size and hits[-1] - hits[0] >= 65:
                cv2.line(lines, (x + hits[0], 577), (x + hits[-1], 577), 255, 3)

    def _fill_red_boundary(self, red: np.ndarray, gap: int) -> tuple[np.ndarray, bool, np.ndarray]:
        closed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
        self._close_bottom_cut(closed)
        outlines, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        forbidden = red.copy()
        clipped_lines = np.zeros_like(red)
        has_enclosure = False
        if hierarchy is not None:
            for index, contour in enumerate(outlines):
                _, _, child, parent = hierarchy[0, index]
                if parent != -1:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                hole_area = max((cv2.contourArea(hole) for hole, node in zip(outlines, hierarchy[0])
                                 if node[3] == index), default=0) if child != -1 else 0
                if w >= 65 and h >= 65 and hole_area >= 2800 and hole_area >= area * 0.5:
                    cv2.drawContours(forbidden, [contour], -1, 255, cv2.FILLED)
                    has_enclosure = True
                elif max(w, h) >= 150 and min(w, h) <= 9:
                    if x <= 4 or y <= 4 or x + w >= 1276 or y + h >= 578:
                        cv2.drawContours(clipped_lines, [contour], -1, 255, cv2.FILLED)
        forbidden = cv2.dilate(forbidden, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        return forbidden, has_enclosure, clipped_lines

    def _red_boundary(self, hsv: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
        strong = self._red_mask(hsv, 22)
        result = self._fill_red_boundary(strong, 9)
        if result[1]:
            return result
        for hue, gap in ((25, 5), (25, 11), (27, 5), (27, 11)):
            alternate = self._fill_red_boundary(self._red_mask(hsv, hue), gap)
            if alternate[1]:
                return alternate
            if np.count_nonzero(alternate[2]) > np.count_nonzero(result[2]):
                result = alternate
        return result

    @staticmethod
    def terrain_mask(hsv: np.ndarray, world: str) -> np.ndarray:
        grass_range = ((56, 28, 65), (102, 194, 180)) if world == "night" else ((29, 44, 82), (85, 215, 232))
        grass = cv2.inRange(hsv, *grass_range)
        value = hsv[:, :, 2]
        neighborhood = np.ones((5, 5), np.uint8)
        texture = cv2.dilate(value, neighborhood) - cv2.erode(value, neighborhood)
        grass[texture > 60] = 0
        return grass

    @staticmethod
    def _surface_palette(hsv: np.ndarray, world: str) -> str:
        sample = hsv[90:485:4, 180:1090:4]
        if world == "night":
            teal = cv2.inRange(sample, (69, 58, 72), (94, 115, 171))
            slate = cv2.inRange(sample, (103, 55, 74), (115, 138, 190))
            return "night_teal" if cv2.countNonZero(teal) > cv2.countNonZero(slate) else "night_stone"
        # The winter scenery uses bright blue-white paving throughout the
        # battlefield, including the playable strip inside its stone parapet.
        winter = cv2.inRange(sample, (113, 20, 150), (132, 130, 255))
        if np.mean(winter > 0) > 0.05:
            return "winter_stone"
        gray = cv2.inRange(sample, (0, 0, 42), (179, 25, 180))
        return "gray_stone" if np.mean(gray > 0) > 0.28 else "grass"

    @staticmethod
    def _outer_ground(hsv: np.ndarray, palette: str) -> np.ndarray:
        # Undeployable home turf is ochre, while the night overlay tints the
        # blue-grey paving violet. Dark cliff ledges have much higher saturation.
        bounds = {"night_stone": ((103, 55, 74), (115, 138, 190)),
                  "night_teal": ((69, 54, 72), (94, 120, 175)),
                  "winter_stone": ((113, 20, 145), (132, 138, 255)),
                  "gray_stone": ((0, 0, 46), (179, 25, 177)),
                  "grass": ((35, 85, 95), (52, 215, 243))}[palette]
        return cv2.inRange(hsv, *bounds)

    @staticmethod
    def _dense_ground(ground: np.ndarray) -> np.ndarray:
        # Individual blades of grass vary sharply in brightness even on a
        # level patch. Require a patch and a ground-coloured centre.
        return cv2.medianBlur(ground, 3) & cv2.inRange(cv2.boxFilter(ground, -1, (11, 11)), 201, 255)

    def _playable(self, hsv: np.ndarray, world: str) -> np.ndarray:
        playable = np.zeros(hsv.shape[:2], np.uint8)
        cv2.fillPoly(playable, [self.PLAYABLE_SAFE_POLY], 255)
        playable[:230, :206] = 0
        playable[:202, 985:] = 0
        playable[:78, 510:760] = 0
        playable[:68, 270:414] = 0
        playable[444:575, :174] = 0
        if world == "night":
            playable[470:539, :605] = 0
        elif self.preview_toolbar_visible(hsv):
            playable[509:] = 0
        return playable

    def _observed_side(self, hsv: np.ndarray, world: str, side: str, playable: np.ndarray) -> DeploymentArea | None:
        dx = -1 if side.endswith("W") else 1
        dy = -1 if side.startswith("N") else 1
        palette = self._surface_palette(hsv, world)
        ground = self._outer_ground(hsv, palette)
        tinted_range = {"night_stone": ((123, 15, 61), (169, 110, 195)),
                        "night_teal": ((24, 20, 62), (63, 128, 175)),
                        "winter_stone": ((134, 18, 122), (161, 131, 255)),
                        "gray_stone": ((0, 30, 40), (14, 158, 173)),
                        "grass": ((20, 60, 70), (36, 242, 235))}[palette]
        tinted = cv2.inRange(hsv, *tinted_range)
        if palette == "gray_stone":
            tinted |= cv2.inRange(hsv, (167, 30, 40), (179, 158, 173))
        # A border pixel must have bare ground outwards and overlay-tinted
        # ground inwards. Red buildings and ordinary paths lack this transition.
        outside = self._translate(cv2.boxFilter(ground, -1, (13, 13)), -8 * dx, -10 * dy)
        inside = self._translate(cv2.boxFilter(tinted, -1, (13, 13)), 4 * dx, 5 * dy)
        if palette == "winter_stone":
            line = cv2.inRange(hsv, (165, 67, 128), (179, 255, 255))
            line |= cv2.inRange(hsv, (0, 67, 128), (7, 255, 255))
        elif palette == "gray_stone":
            # The asphalt has little colour. The overlay and its edge are
            # both darker than those on grass, but still add distinct red.
            line = cv2.inRange(hsv, (0, 96, 60), (11, 255, 255))
            line |= cv2.inRange(hsv, (167, 96, 60), (179, 255, 255))
        elif palette in {"night_stone", "night_teal"}:
            line = self._red_mask(hsv, 22)
        else:
            line = cv2.inRange(hsv, (0, 140, 103), (22, 255, 255))
        line[(outside <= 130) | (inside <= 68)] = 0
        # Isometric tile sides have a screen slope of +/-3:4. Check a run
        # in this direction before projecting any touch onto its outer side.
        kernel = np.zeros((19, 25), np.float32)
        y0 = 18 if dx == dy else 0
        cv2.line(kernel, (0, y0), (24, 18 - y0), 1, 1)
        thick = cv2.dilate(line, np.ones((3, 3), np.uint8))
        support = cv2.filter2D(thick, -1, kernel / kernel.sum())
        line[support < 148] = 0
        if not np.any(line):
            return None
        line = cv2.dilate(line, np.ones((5, 5), np.uint8))
        strip = np.zeros_like(line)
        for step in range(5, 27):
            strip |= self._translate(line, dx * step, dy * step)
        usable = strip & playable & self._dense_ground(ground)
        return self._continuous_side(usable, dx, dy)

    @staticmethod
    def _continuous_side(usable: np.ndarray, dx: int, dy: int) -> DeploymentArea | None:
        clearance = cv2.distanceTransform(usable, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        rows, cols = np.nonzero(clearance >= 8)
        if not cols.size:
            return None
        # A stepped red perimeter can expose a parallel edge on the far
        # side of a building. Choose one coherent band for all three regions.
        offset = 1700
        normal = np.rint(0.75 * dx * cols + dy * rows).astype(int) + offset
        density = np.convolve(np.bincount(normal, minlength=3401), np.ones(91), mode="same")
        peak = int(np.argmax(density))
        band = np.abs(normal - peak) <= 45
        left, right = np.percentile(cols[band], (2, 98)).astype(int)
        if right - left < 110:
            return None
        ys, xs = np.ogrid[:usable.shape[0], :usable.shape[1]]
        usable[np.abs(0.75 * dx * xs + dy * ys + offset - peak) > 45] = 0
        return DeploymentArea(cv2.distanceTransform(usable, cv2.DIST_L2, cv2.DIST_MASK_PRECISE),
                              (int(left), int(right)))

    def _night_board_edge(self, hsv: np.ndarray, side: str, playable: np.ndarray) -> DeploymentArea | None:
        # In the active second stage the overlay can disappear. The tiled
        # clearing still ends at a sloping edge of dense vegetation. Use the
        # outermost flat tiles alongside that edge, with a margin from any red.
        dx = -1 if side.endswith("W") else 1
        dy = -1 if side.startswith("N") else 1
        palette = self._surface_palette(hsv, "night")
        ground = self._outer_ground(hsv, palette)
        surface = cv2.morphologyEx(ground, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(surface)
        if count < 2:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest, cv2.CC_STAT_AREA] < 40000:
            return None
        board = np.uint8(labels == largest) * 255
        rows, cols = np.nonzero(board)
        normal = 0.75 * dx * cols + dy * rows
        edge = np.percentile(normal, 99)
        ys, xs = np.ogrid[:board.shape[0], :board.shape[1]]
        depth = edge - (0.75 * dx * xs + dy * ys)
        border = (cv2.inRange(hsv, (90, 145, 25), (122, 255, 180)) if palette == "night_stone"
                  else cv2.inRange(hsv, (48, 82, 22), (88, 255, 126)))
        border &= cv2.bitwise_not(board)
        outward = self._translate(cv2.boxFilter(border, -1, (23, 23)), -30 * dx, -38 * dy)
        safe = self._dense_ground(ground) & board & playable
        safe[(depth < 9) | (depth > 40) | (outward < 95)] = 0
        red = cv2.dilate(self._red_mask(hsv, 20), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        return self._continuous_side(safe & cv2.bitwise_not(red), dx, dy)

    @staticmethod
    def preview_toolbar_visible(hsv: np.ndarray) -> bool:
        # The adjacent purple boost buttons occupy the same lower strip as
        # the green copy/auto-battle controls during the multiplayer preview.
        purple = [np.mean(cv2.inRange(hsv[520:550, a:b], (115, 35, 215), (140, 185, 255)) > 0) > 0.50
                  for a, b in ((190, 350), (418, 538))]
        copy_button = cv2.inRange(hsv[520:547, 628:793], (32, 70, 185), (55, 239, 255))
        return any(purple) and (all(purple) or np.mean(copy_button > 0) > 0.45)

    def current_terrain_accepts(self, frame: np.ndarray, world: str, *, observed: bool = False) -> DeploymentArea:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ground = (self._dense_ground(self._outer_ground(hsv, self._surface_palette(hsv, world)))
                  if observed else self.terrain_mask(hsv, world))
        usable = ground & self._playable(hsv, world) & cv2.bitwise_not(self._red_mask(hsv, 25))
        return DeploymentArea(cv2.distanceTransform(usable, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))

    def _build_forbidden_and_playable_masks(
        self, screen_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        playable = np.zeros((720, 1280), np.uint8)
        cv2.fillPoly(playable, [self.PLAYABLE_SAFE_POLY], 255)
        if not is_battle_frame(screen_bgr):
            return np.full_like(playable, 255), playable
        forbidden, _, _ = self._red_boundary(cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV))
        return forbidden, playable

    def deployment_area(self, frame: np.ndarray, world: str, side: str) -> DeploymentArea:
        valid = np.zeros((720, 1280), np.uint8)
        if not is_battle_frame(frame) or side not in self.CAMERA_FOCUS_SWIPES:
            return DeploymentArea(valid)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        playable = self._playable(hsv, world)
        observed = self._observed_side(hsv, world, side, playable)
        if observed is not None:
            return observed
        forbidden, enclosed, clipped = self._red_boundary(hsv)
        if world == "night" and not enclosed:
            board = self._night_board_edge(hsv, side, playable)
            if board is not None:
                return board
        if not enclosed and not np.any(clipped):
            return DeploymentArea(valid)
        grass = self.terrain_mask(hsv, world)
        valid = grass & playable & cv2.bitwise_not(forbidden)
        if not enclosed:
            valid &= self._exterior_strip(grass, clipped, side)
        return DeploymentArea(cv2.distanceTransform(valid, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))

    @staticmethod
    def _exterior_strip(grass: np.ndarray, clipped: np.ndarray, side: str) -> np.ndarray:
        """An open line needs both an inward red trace and an outward terrain edge."""
        dx = -4 if side.endswith("W") else 4
        dy = -5 if side.startswith("N") else 5
        line = cv2.dilate(clipped, np.ones((7, 7), np.uint8))
        terrain = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        outside = cv2.erode(cv2.bitwise_not(terrain), np.ones((19, 19), np.uint8))
        inward_red = np.zeros_like(grass)
        outward_edge = np.zeros_like(grass)
        for step in range(2, 27):
            if step < 19:
                inward_red |= RedZoneDetector._translate(line, dx * step, dy * step)
            outward_edge |= RedZoneDetector._translate(outside, -dx * step, -dy * step)
        return inward_red & outward_edge

    @staticmethod
    def _translate(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = mask.shape
        result = np.zeros_like(mask)
        x0, x1 = max(dx, 0), min(w + dx, w)
        y0, y1 = max(dy, 0), min(h + dy, h)
        result[y0:y1, x0:x1] = mask[y0-dy:y1-dy, x0-dx:x1-dx]
        return result

    def detect_shifted_side_points(
        self, screen_bgr: np.ndarray, side_name: str, points_per_side: int = 12,
    ) -> Tuple[List[Point], List[Point]]:
        if side_name not in self.SHIFTED_OUTER_PERIMETER_POLYLINES:
            return [], []
        area = self.deployment_area(screen_bgr, "home", side_name)
        main = self._sample_polyline(self.SHIFTED_OUTER_PERIMETER_POLYLINES[side_name], points_per_side)
        guard = self._sample_polyline(self.SHIFTED_OUTER_GUARD_POLYLINES[side_name], points_per_side)
        return area.sample(main), area.sample(guard)

    def detect_deployable_sides(
        self, screen_bgr: np.ndarray, points_per_side: int = 12,
    ) -> Dict[str, List[Point]]:
        return {
            side: self.deployment_area(screen_bgr, "home", side).sample(self._sample_polyline(poly, points_per_side))
            for side, poly in self.OUTER_PERIMETER_POLYLINES.items()
        }
