import cv2
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from config import CombatConfig, load_config
from core.device import AdbDeviceController, BaseDeviceController
from core.tasks.combat_ai import CombatAiTask
from core.tasks.deployment import BattleDeployment
from core.tasks.night_world import NightWorldTask
from core.vision import DeadBaseChecker, OcrScanner, RedZoneDetector, TemplateMatcher
from core.vision.deployment_bar import DeploymentBarReader
from core.vision.flank_damage import FlankWatch


def test_filled_red_zone_rejects_grass_inside_the_boundary():
    screen = np.full((720, 1280, 3), (45, 115, 55), np.uint8)
    cv2.rectangle(screen, (297, 211), (393, 307), (20, 20, 235), 4)
    forbidden, _ = RedZoneDetector()._build_forbidden_and_playable_masks(screen)
    assert forbidden[259, 345] > 0


def test_no_candidates_on_fully_red_battlefield():
    frame = np.full((720, 1280, 3), (0, 0, 255), np.uint8)
    assert RedZoneDetector().detect_shifted_side_points(frame, 'NW') == ([], [])


def test_missing_frame_leaves_card_state_unknown():
    assert CombatAiTask.detect_active_battle_slots(None) == []
    assert CombatAiTask.is_slot_depleted(None, 125) is None
    assert CombatAiTask.is_hero_or_siege_deployed(None, None, 125) is None
    assert NightWorldTask.detect_active_night_slots(None) == (None, [])
    assert NightWorldTask.has_undeployed_night_troops(None, 242) is None


def test_projected_short_swipe_endpoint_must_be_inside_the_area():
    device = AdbDeviceController()
    device._phys_width, device._phys_height = 1600, 900
    with patch.object(device, '_run_adb_interruptible') as send:
        sent = device.deployment_batch_tap([(400, 400)], lambda p: p == (400, 400), lambda: False)
        assert sent == []
        send.assert_not_called()


def test_adb_deployment_sequence_uses_one_shell_request():
    device = AdbDeviceController()
    touches = [device.deployment_touch(p) for p in [(154, 645), (400, 400), (410, 400)]]
    with patch.object(device, '_run_adb_interruptible', return_value=b'1\n2\n3\n') as send:
        assert [t.logical for t in device.send_deployment_sequence(touches, lambda: False)] == [t.logical for t in touches]
        send.assert_called_once()
        script = send.call_args.kwargs['input_bytes'].decode()
        assert script.count('press_point ') == 3
        assert '400 400 401 401' in script


def test_unreachable_selected_device_is_reported_as_disconnected():
    device = AdbDeviceController(serial='127.0.0.1:16480')
    with patch.object(device, 'list_online_devices', return_value=['127.0.0.1:16384']), \
            patch('core.device.adb_controller.subprocess.run') as connect:
        assert not device.connect()
        connect.assert_called_once()


def test_red_decoration_and_missing_line_do_not_establish_deployment_area():
    frame = np.full((720, 1280, 3), (45, 115, 55), np.uint8)
    d = RedZoneDetector()
    assert d.detect_shifted_side_points(frame, 'NW') == ([], [])
    cv2.circle(frame, (600, 300), 20, (15, 20, 235), 3)
    assert d.detect_shifted_side_points(frame, 'NW') == ([], [])


def test_short_gap_is_closed_and_night_cliff_is_excluded():
    screen = np.full((720, 1280, 3), (120, 145, 74), np.uint8)
    cv2.rectangle(screen, (390, 170), (850, 455), (19, 23, 230), 3)
    screen[167:174, 498:503] = (120, 145, 74)
    screen[230:285, 270:340] = (131, 87, 80)
    area = RedZoneDetector().deployment_area(screen, 'night', 'NW')
    assert not area.accepts((500, 171))
    assert not area.accepts((550, 260))
    assert not area.accepts((303, 244))
    assert not area.accepts((90, 541))
    assert area.sample([(353, 274)])


def test_clipped_boundary_requires_ground_outside_the_line():
    frame = np.full((720, 1280, 3), (120, 145, 74), np.uint8)
    cv2.line(frame, (450, 0), (450, 580), (20, 20, 235), 3)
    frame[:, :330] = (130, 80, 70)
    d = RedZoneDetector()
    strip = d.deployment_area(frame, 'night', 'NW')
    assert not strip.accepts((470, 244))
    assert not strip.accepts((320, 244))
    assert strip.accepts((395, 244))


def test_preview_toolbar_green_buttons_are_never_sampled_as_grass():
    frame = BattleReplay().screencap()
    d = RedZoneDetector()
    assert d.deployment_area(frame, 'home', 'SE').accepts((650, 533))
    frame[520:550, 190:350] = (250, 170, 166)
    frame[520:550, 418:538] = (250, 170, 166)
    assert not d.deployment_area(frame, 'home', 'SE').accepts((650, 533))
    assert not d.current_terrain_accepts(frame, 'home').accepts((650, 533))


def draw_card(frame, x, count, *, selected=False, top=595, kind='troop'):
    y = top - 6 if selected else top
    border = (250, 250, 250) if selected else (12, 12, 12)
    fill = (208, 151, 77) if kind == 'troop' else (225, 89, 115)
    cv2.rectangle(frame, (x - 44, y), (x + 44, top + 116), border, -1)
    cv2.rectangle(frame, (x - 41, y + 4), (x + 41, top + 112), fill, -1)
    cv2.rectangle(frame, (x - 29, y + 26), (x + 31, top + 105), (155, 119, 220), -1)
    cv2.rectangle(frame, (x - 37, top + 85), (x - 13, top + 108), (24, 24, 24), -1)
    cv2.putText(frame, '2', (x - 33, top + 101), cv2.FONT_HERSHEY_DUPLEX, .55, (255, 255, 255), 1)
    if count is not None:
        cv2.putText(frame, f'x{count}', (x + 6, y + 20), cv2.FONT_HERSHEY_DUPLEX, .61, (255, 255, 255), 2)


class ReplayClock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def wait(self, duration):
        self.value += duration


class BattleReplay(BaseDeviceController):
    def __init__(self, *, world='home', quantity=8, accept=None, stop_at=None, end_at=None, transition=False):
        self.world = world
        self.clock = ReplayClock()
        self.initial = quantity
        self.slots = {154: quantity}
        self.selected = None
        self.points = []
        self.swipes = []
        self.taps = []
        self.zoom_count = 0
        self.accept = accept or (lambda _: True)
        self.stop_at, self.end_at = stop_at, end_at
        self.transition = transition
        self.stage2 = False
        self.complete_at = None

    def connect(self):
        return True

    def stopped(self):
        return self.stop_at is not None and len(self.points) >= self.stop_at

    def screencap(self):
        time_since_clear = self.clock.now() - self.complete_at if self.complete_at is not None else None
        if (self.end_at is not None and len(self.points) >= self.end_at) or (
            self.stage2 and self.slots[252] == 0 and time_since_clear >= 3
        ):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[585:651, 560:721] = (20, 200, 50)
            return frame
        if self.transition and time_since_clear is not None and not self.stage2:
            if 1.5 <= time_since_clear < 2.7:
                return np.full((720, 1280, 3), 244, np.uint8)
            if time_since_clear >= 2.7:
                self.stage2, self.selected = True, None
                self.slots = {252: 2}
        grass = (120, 145, 74) if self.world == 'night' else (45, 115, 55)
        frame = np.full((720, 1280, 3), grass, np.uint8)
        box = ((430, 190), (850, 445))
        if self.stage2:
            frame[135:462, 190:1039] = (165, 178, 97)
            box = ((420, 178), (810, 390))
        cv2.rectangle(frame, *box, (9, 15, 235), 4)
        if self.world == 'night':
            frame[25:63, 20:125] = (86, 183, 90)
            frame[484:517, 35:148] = (20, 20, 225)
        else:
            frame[512:560, 20:160] = (20, 20, 225)
        frame[580:] = (37, 39, 42)
        for x, remaining in self.slots.items():
            draw_card(frame, x, remaining, selected=self.selected == x)
        return frame

    def tap(self, x, y, humanize=True):
        self.taps.append((x, y))
        if y in (622, 645):
            self.selected = next((cx for cx in self.slots if abs(cx - x) < 8), None)
        elif y < 495 and self.selected is not None:
            self.points.append((x, y))
            if self.slots[self.selected] > 0 and self.accept(len(self.points)):
                self.slots[self.selected] -= 1
                if self.transition and not any(self.slots.values()):
                    self.complete_at = self.clock.now()
        return x, y

    def swipe(self, start, end, duration_ms=350, use_bezier=True):
        self.swipes.append((start, end))

    def pinch_zoom_out(self):
        self.zoom_count += 1


class ReplayOcr(OcrScanner):
    def __init__(self):
        super().__init__(custom_ocr_fn=lambda _: '')
        self.real = OcrScanner()

    def read_text_from_roi(self, frame, roi):
        if roi[0] > 550 and roi[1] < 640:
            return self.real.read_text_from_roi(frame, roi)
        return ''

    def scan_destruction_rate(self, frame):
        return 0


def test_dimmed_x0_badge_is_recognized_when_ocr_reads_letter_o():
    frame = BattleReplay(quantity=0).screencap()
    card = frame[593:714, 109:200]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    card[:] = (gray[:, :, None] * 0.52).astype(np.uint8)
    reader = DeploymentBarReader(OcrScanner(custom_ocr_fn=lambda _: 'XO'), 'home')
    bar = reader.scan(frame)
    assert bar.supported
    slot, = bar.cards.values()
    assert (slot.kind, slot.remaining) == ('spent', 0)


def test_known_card_can_show_zero_without_a_detectable_outer_frame():
    frame = BattleReplay().screencap()
    frame[580:] = (18, 18, 18)
    frame[598:617, 120:192] = (65, 65, 65)
    cv2.putText(frame, 'x0', (160, 614), cv2.FONT_HERSHEY_DUPLEX, .61, (125, 125, 125), 2)
    reader = DeploymentBarReader(OcrScanner(custom_ocr_fn=lambda _: 'XO'), 'home')
    assert not reader.scan(frame).cards
    assert reader.scan(frame, (154,)).cards[154].remaining == 0
    assert 257 not in reader.scan(frame, (257,)).cards


@pytest.mark.parametrize('strategy', ['line_air_spam', 'surround_barch'])
def test_home_entry_counts_only_actual_decrements_and_recovers(strategy):
    replay = BattleReplay(accept=lambda n: n not in {3, 4, 7})
    task = CombatAiTask(replay, RedZoneDetector(), DeadBaseChecker(), ReplayOcr(),
                        CombatConfig(strategy=strategy, max_battle_duration_sec=32),
                        now=replay.clock.now, wait=replay.clock.wait, stopped=replay.stopped)
    phase, = task.execute_attack()['deployment']
    assert phase['status'] == 'completed', phase
    assert len(replay.points) == 11
    assert phase['touch_count'] == len(replay.points)
    assert phase['candidate_counts']['center'] > 0
    assert phase['center_wave']['touches'] == 8
    assert phase['center_wave']['recovery_touches'] == 3
    assert [(s['initial'], s['deployed'], s['remaining']) for s in phase['slots']] == [(8, 8, 0)]
    assert any(swipe in RedZoneDetector.CAMERA_FOCUS_SWIPES.values() for swipe in replay.swipes)


def test_adb_feedback_still_confirms_with_two_slow_screenshots():
    class SlowCapture(BattleReplay):
        def screencap(self):
            self.clock.wait(2.8)
            return super().screencap()

    replay = SlowCapture(quantity=1)
    phase = run_home_replay(replay)
    assert phase['status'] == 'completed'
    assert phase['touch_count'] == 1
    assert phase['slots'][0]['deployed'] == 1


@pytest.fixture
def dragon_portrait(monkeypatch):
    scan = DeploymentBarReader.scan

    def with_dragon(reader, frame, known_slots=()):
        bar = scan(reader, frame, known_slots)
        if reader.world == 'home':
            for x, card in bar.cards.items():
                if abs(x - 154) <= 6 and card.kind == 'troop':
                    bar.cards[x] = SimpleNamespace(**(vars(card) | {'unit': 'dragon'}))
        return bar

    monkeypatch.setattr(DeploymentBarReader, 'scan', with_dragon)
    monkeypatch.setattr('core.tasks.combat_ai.random.choice', lambda _: 'NW')


def run_home_replay(replay):
    task = CombatAiTask(replay, RedZoneDetector(), DeadBaseChecker(), ReplayOcr(),
                        CombatConfig(strategy='line_air_spam', max_battle_duration_sec=60),
                        now=replay.clock.now, wait=replay.clock.wait, stopped=replay.stopped)
    phase, = task.execute_attack()['deployment']
    return phase


def position_on_nw_edge(point):
    x, y = point
    return ((x - 145) * 440 - (y - 408) * 328) / (440**2 + 328**2)


def test_dragons_clear_each_wing_once_then_push_through_center(dragon_portrait):
    replay = BattleReplay(quantity=16)
    phase = run_home_replay(replay)
    left, right = (phase['flanks'][wing] for wing in ('left', 'right'))
    assert (left['before'], left['after'], right['before'], right['after']) == (16, 15, 15, 14)
    assert left['side'] == right['side'] == 'NW'
    assert position_on_nw_edge(replay.points[0]) < .25
    assert position_on_nw_edge(replay.points[1]) > .75
    assert all(.20 < position_on_nw_edge(p) < .80 for p in replay.points[2:])
    assert len(replay.points) == 16
    assert phase['center_wave']['touches'] == 14
    assert phase['center_wave']['dispatch_seconds'] < 5
    assert phase['panorama_status'] == 'confirmed' and replay.zoom_count == 1
    assert phase['center_push'] == 'flank_wait_timeout'


def test_one_dragon_goes_directly_through_the_middle(dragon_portrait):
    replay = BattleReplay(quantity=1)
    phase = run_home_replay(replay)
    assert phase['center_push'] == 'insufficient_dragons'
    assert len(replay.points) == 1
    assert .20 < position_on_nw_edge(replay.points[0]) < .80


def test_rejected_right_wing_retries_without_repeating_the_left(dragon_portrait, monkeypatch):
    replay = BattleReplay(quantity=4, accept=lambda attempt: attempt != 2)
    monkeypatch.setattr(BattleDeployment, '_red_zone_message', lambda self, frame: len(replay.points) == 2)
    phase = run_home_replay(replay)
    assert phase['status'] == 'completed', phase
    assert [row['after'] for row in phase['flanks'].values()] == [3, 2]
    assert len(replay.points) == 5
    assert all(position_on_nw_edge(p) > .75 for p in replay.points[1:3])
    assert replay.points[1] != replay.points[2]


def test_delayed_wing_confirmation_does_not_send_duplicate_troop(dragon_portrait):
    class DelayedReplay(BattleReplay):
        pending_until = None

        def tap(self, x, y, humanize=True):
            if y < 495 and self.selected is not None and not self.points:
                self.points.append((x, y))
                self.pending_until = self.clock.now() + 2.75
                return x, y
            return super().tap(x, y, humanize)

        def screencap(self):
            if self.pending_until is not None and self.clock.now() >= self.pending_until:
                self.slots[154] -= 1
                self.pending_until = None
            frame = super().screencap()
            if self.pending_until is not None:
                frame[585:621, 160:200] = (80, 80, 80)
            return frame

    replay = DelayedReplay(quantity=4)
    phase = run_home_replay(replay)
    assert phase['flanks']['left']['after'] == 3
    assert phase['flanks']['right']['after'] == 2
    assert len(replay.points) == 4


def test_a_confirmed_left_wing_never_continues_mid_when_right_is_unconfirmed(dragon_portrait):
    replay = BattleReplay(quantity=4, accept=lambda attempt: attempt == 1)
    phase = run_home_replay(replay)
    assert phase['flanks']['left']['after'] == 3
    assert phase['flanks']['right']['status'] == 'no_progress'
    assert phase['center_push'] is None
    assert phase['remaining_slots'][0]['remaining'] == 3
    assert all(position_on_nw_edge(p) > .75 for p in replay.points[1:])


def test_center_push_starts_as_soon_as_both_flanks_show_damage(dragon_portrait, monkeypatch):
    replay = BattleReplay(quantity=3)
    samples = []

    class Building:
        def __init__(self, required):
            self.required = required
            self.observations = 0

        def damaged(self, frame):
            self.observations += 1
            samples.append((len(replay.points), replay.clock.now()))
            return self.observations >= self.required

    monkeypatch.setattr(FlankWatch, 'find', lambda frame, world, point:
                        Building(2 if position_on_nw_edge(point) < .25 else 4))
    phase = run_home_replay(replay)
    assert phase['center_push'] == 'both_flanks_engaged'
    assert all(count == 2 for count, _ in samples)
    assert samples[-1][1] - samples[0][1] < 6
    assert len(replay.points) == 3


def test_heroes_and_siege_use_only_the_segment_between_confirmed_wings(dragon_portrait, monkeypatch):
    class MixedArmy(BattleReplay):
        drops = []

        def tap(self, x, y, humanize=True):
            before = len(self.points)
            point = super().tap(x, y, humanize)
            if len(self.points) > before:
                self.drops.append((self.selected, point))
            return point

    replay = MixedArmy(quantity=3)
    replay.slots.update({252: 1, 350: 1})
    scan = DeploymentBarReader.scan

    def mixed(reader, frame, known_slots=()):
        bar = scan(reader, frame, known_slots)
        for x, card in bar.cards.items():
            kind = 'siege' if abs(x - 252) <= 6 else 'hero' if abs(x - 350) <= 6 else None
            if kind:
                bar.cards[x] = SimpleNamespace(**(vars(card) | {'kind': kind}))
        return bar

    monkeypatch.setattr(DeploymentBarReader, 'scan', mixed)
    phase = run_home_replay(replay)
    assert phase['status'] == 'completed', phase
    assert {slot for slot, _ in replay.drops[2:]} == {154, 252, 350}
    assert all(.20 < position_on_nw_edge(point) < .80 for _, point in replay.drops[2:])


@pytest.mark.parametrize('mode', ['success', 'refused', 'troops_left'])
def test_spells_wait_for_all_troops_and_the_new_panorama(mode, dragon_portrait, monkeypatch):
    class SpellReplay(BattleReplay):
        def pinch_zoom_out(self):
            assert self.slots[154] == 0 and self.slots[252] == 1
            super().pinch_zoom_out()

        def screencap(self):
            frame = super().screencap()
            if self.zoom_count:
                if mode == 'refused':
                    return np.full_like(frame, 245)
                frame[100:580, 165:1115] = (25, 166, 101)
                frame[512:560, 20:160] = (20, 20, 225)
            return frame

        def tap(self, x, y, humanize=True):
            if y < 495 and self.selected == 252:
                assert self.zoom_count == 1
            return super().tap(x, y, humanize)

    replay = SpellReplay(quantity=3, accept=lambda n: mode != 'troops_left' or n <= 2)
    replay.slots[252] = 1
    scan = DeploymentBarReader.scan

    def with_spell(reader, frame, known_slots=()):
        bar = scan(reader, frame, known_slots)
        if 252 in bar.cards and bar.cards[252].kind == 'troop':
            bar.cards[252] = SimpleNamespace(**(vars(bar.cards[252]) | {'kind': 'spell'}))
        return bar

    monkeypatch.setattr(DeploymentBarReader, 'scan', with_spell)
    phase = run_home_replay(replay)
    if mode == 'success':
        assert phase['panorama_status'] == 'confirmed'
        assert replay.points[-1] == (639, 339) and replay.slots[252] == 0
    else:
        assert phase['panorama_status'] == ('view_unconfirmed' if mode == 'refused' else 'pending')
        assert replay.slots[252] == 1


@pytest.mark.parametrize('stop_at,end_at,reason', [(1, None, 'stopped'), (None, 1, 'natural_end')])
def test_battle_exit_preserves_last_known_troops(stop_at, end_at, reason):
    replay = BattleReplay(quantity=3, stop_at=stop_at, end_at=end_at)
    task = CombatAiTask(replay, RedZoneDetector(), DeadBaseChecker(), ReplayOcr(), CombatConfig(),
                        now=replay.clock.now, wait=replay.clock.wait, stopped=replay.stopped)
    phase, = task.execute_attack()['deployment']
    assert phase['reason'] == reason
    assert phase['remaining_slots'][0]['last_known_remaining'] == 3
    assert len(replay.points) == 1
    assert replay.zoom_count == 0
    assert (CombatAiTask.BTN_RETURN_HOME in replay.taps) == (reason == 'natural_end')


@pytest.mark.parametrize('split_camps', [False, True])
def test_night_entry_detects_second_stage_without_100_percent_reading(split_camps):
    replay = BattleReplay(world='night', quantity=1 if split_camps else 2, transition=True)
    if split_camps:
        replay.slots[252] = 1
    cfg = load_config()
    cfg.combat.max_battle_duration_sec = 32
    task = NightWorldTask(replay, TemplateMatcher(), ReplayOcr(), None, cfg.combat,
                          now=replay.clock.now, wait=replay.clock.wait)
    phases = task.execute_night_battle()['deployment']
    assert [(p['phase'], p['status'], p['reason']) for p in phases] == [
        (1, 'completed', 'stage_advanced'), (2, 'completed', 'natural_end')]
    assert [[s['x'] for s in p['slots']] for p in phases] == [[154, 252] if split_camps else [154], [252]]
    assert [p['flanks']['left']['slot'] for p in phases] == [154, 252]
    assert [p['flanks']['right']['slot'] for p in phases] == [252 if split_camps else 154, 252]
    assert all(p['flanks']['right']['after'] == 0 for p in phases)
    assert len(replay.points) == 4
    assert [p['panorama_status'] for p in phases] == ['confirmed', 'confirmed']
    assert replay.zoom_count == 2


def test_second_visible_row_is_reported_as_unsupported():
    frame = BattleReplay().screencap()
    draw_card(frame, 251, 3, top=477)
    assert not DeploymentBarReader(ReplayOcr(), 'home').scan(frame).supported


def test_no_input_when_camera_keeps_moving_after_focus():
    class MovingBattle(BattleReplay):
        captures = 0

        def screencap(self):
            frame = super().screencap()
            if self.swipes:
                self.captures += 1
                if self.captures % 2:
                    frame[130:465, 190:1040] = (140, 190, 120)
            return frame

    replay = MovingBattle(quantity=1)
    task = CombatAiTask(replay, RedZoneDetector(), DeadBaseChecker(), ReplayOcr(), CombatConfig(),
                        now=replay.clock.now, wait=replay.clock.wait, stopped=replay.stopped)
    phase, = task.execute_attack()['deployment']
    assert not replay.points
    assert phase['remaining_slots'][0]['last_known_remaining'] == 1


def test_last_troop_is_preserved_when_no_touches_decrement_it():
    replay = BattleReplay(quantity=1, accept=lambda _: False)
    task = CombatAiTask(replay, RedZoneDetector(), DeadBaseChecker(), ReplayOcr(), CombatConfig(),
                        now=replay.clock.now, wait=replay.clock.wait)
    phase, = task.execute_attack()['deployment']
    assert phase['remaining_slots'][0]['remaining'] == 1
    assert phase['remaining_slots'][0]['deployed'] == 0
    assert len(replay.points) <= 12


def test_red_line_fade_does_not_discard_the_area_on_the_same_aligned_camera():
    class FadingLineReplay(BattleReplay):
        def screencap(self):
            frame = super().screencap()
            if self.points:
                cv2.rectangle(frame, (430, 190), (850, 445), (45, 115, 55), 4)
            return frame

    replay = FadingLineReplay(quantity=8)

    def wait(duration):
        replay.clock.wait(6 if duration == 1.8 else duration)

    task = CombatAiTask(replay, RedZoneDetector(), DeadBaseChecker(), ReplayOcr(), CombatConfig(),
                        now=replay.clock.now, wait=wait)
    phase, = task.execute_attack()['deployment']
    assert phase['status'] == 'completed'
    assert len(replay.points) == 8


def test_desktop_summary_uses_same_phase_and_residuals_for_loop_and_step():
    from platforms.pc_desktop.main_window import deployment_lines

    phase = {'phase': 1, 'status': 'partial', 'reason': 'natural_end', 'remaining_slots': [
        {'x': 154, 'remaining': None, 'last_known_remaining': 1, 'reason': 'quantity_unknown'}
    ]}
    step = {'combat': {'deployment': [phase]}}
    lines = deployment_lines(step)
    assert lines == deployment_lines({'result': {'home': step}})
    assert '自然结算' in lines[0][0]
    assert '槽位 X=154，剩余 未知（上次确认 1）' in lines[0][1]
