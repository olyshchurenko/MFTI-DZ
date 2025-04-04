import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):

        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = {
                    'positions': [self._center(det)],
                    'missed': 0
                }
                self.next_id += 1
            return self.tracks

        matched_det = set()
        updated_tracks = {}

        for track_id, track in self.tracks.items():
            last_pos = track['positions'][-1]
            min_dist = float('inf')
            best_match = None

            for i, det in enumerate(detections):
                if i in matched_det:
                    continue

                dist = self._distance(last_pos, self._center(det))
                if dist < 25:  # MAX_DISTANCE
                    if dist < min_dist:
                        min_dist = dist
                        best_match = i

            if best_match is not None:
                matched_det.add(best_match)
                track['positions'].append(self._center(detections[best_match]))
                track['missed'] = 0
                updated_tracks[track_id] = track

        for i, det in enumerate(detections):
            if i not in matched_det:
                self.tracks[self.next_id] = {
                    'positions': [self._center(det)],
                    'missed': 0
                }
                self.next_id += 1

        return self.tracks

    def _center(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class BloodCellTracker:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        self.MIN_CELL_AREA = 100
        self.MAX_CELL_AREA = 3000
        self.THRESHOLD = 15
        self.MIN_DISTANCE = 50
        self.MIN_ASPECT_RATIO = 1.0
        self.MAX_ASPECT_RATIO = 2.5

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.trackers = []
        self.bboxes = []
        self.track_ids = []
        self.track_history = {}
        self.next_id = 0
        self.colors = np.random.randint(0, 255, (1000, 3))
        self.tracker = SimpleTracker()

    def preprocess_frame(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = hsv[:, :, 1]

            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, self.kernel)

            return morphed
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            return None

    def detect_cells(self, frame):
        mask = self.preprocess_frame(frame)
        if mask is None:
            return []

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if (self.MIN_CELL_AREA < area < self.MAX_CELL_AREA and
                    0.7 < circularity < 1.3):  # Фильтр по "круглости"
                x, y, w, h = cv2.boundingRect(cnt)
                valid_boxes.append((x, y, w, h))
                aspect_ratio = max(w, h) / min(w, h)
                if self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO:
                    valid_boxes.append((x, y, w, h))

        return valid_boxes

    def update_tracking(self, frame):
        detections = self.detect_cells(frame)
        tracks = self.tracker.update(detections)

        self.track_history = {
            track_id: data['positions']
            for track_id, data in tracks.items()
            if data['missed'] < 5
        }

        return len(self.track_history)

    def visualize(self, frame):
        display_frame = frame.copy()

        for track_id, positions in self.track_history.items():
            color = self.colors[track_id % len(self.colors)].tolist()

            for i in range(1, len(positions)):
                cv2.line(display_frame, positions[i - 1], positions[i], color, 2)

            if positions:
                x, y = positions[-1]
                cv2.circle(display_frame, (x, y), 5, color, -1)
                cv2.putText(display_frame, f"ID:{track_id}", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('Cell Tracking', display_frame)
        cv2.waitKey(1)

    def save_results(self, output_dir='output'):
        min_track_length = 10
        for track_id, positions in list(self.track_history.items()):
            if len(positions) < min_track_length:
                del self.track_history[track_id]
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'tracks.csv'), 'w') as f:
            f.write("track_id,frame,x,y\n")
            for track_id, positions in self.track_history.items():
                for frame_num, (x, y) in enumerate(positions):
                    f.write(f"{track_id},{frame_num},{x},{y}\n")

        plt.figure(figsize=(10, 10))
        for track_id, positions in self.track_history.items():
            if len(positions) > 1:
                xs, ys = zip(*positions)
                plt.plot(xs, ys, label=f'Track {track_id}')

        plt.gca().invert_yaxis()
        plt.legend()
        plt.title("Cell Trajectories")
        plt.savefig(os.path.join(output_dir, 'trajectories.png'))
        plt.close()

    def run(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc="Tracking cells") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                try:
                    detections = self.detect_cells(frame)
                    active_count = self.update_tracking(frame)
                    self.visualize(frame)

                    if cv2.waitKey(1) == 27:  # ESC для выхода
                        break

                    pbar.update(1)
                except Exception as e:
                    print(f"Ошибка обработки кадра: {e}")
                    break

        self.save_results()
        self.cap.release()
        cv2.destroyAllWindows()


video_path = 'cells.mp4'
tracker = BloodCellTracker(video_path)
tracker.run()
print("Трекинг завершен. Результаты сохранены в папке output/")