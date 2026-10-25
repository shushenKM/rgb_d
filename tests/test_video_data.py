"""Synthetic read-only data checks, including parent split isolation and depth maps."""

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.rgbd_annotations import VIEW_COLUMNS, read_rgbd_annotations
from vision.dataset import (PROJECT_ROOT, VideoDatasetPreparationError, VideoGestureDataset,
                            _training_splits, prepare_video_dataset)
from vision.recordings import (inspect_recording, read_depth_frames, read_rgb_frames,
                               sample_frame_indices, source_path)


HEADERS = ["session_id", "gesture_label", "object", "repetition"] + [c for cols in VIEW_COLUMNS.values() for c in (cols[2], *cols[:2])]


def make_video(path, count=300, fps=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 32))
    if not writer.isOpened():
        raise RuntimeError("MJPG video fixture writer is unavailable")
    try:
        for frame in range(count):
            image = np.full((32, 32, 3), (frame % 200, 60, 180), dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()


def make_annotation(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerow({})
        for session, label, repetition, start, end in rows:
            writer.writerow(dict(session_id=session, gesture_label="\t"+label+"  ", object="1", repetition=repetition,
                start_time=start+.1, end_time=end+.1, start_time2=start, end_time2=end))


class VideoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = PROJECT_ROOT / "runs" / "test_fixtures"
        base.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="video_data_", dir=base)
        cls.root = Path(cls.temporary.name)
        rows = [("step1", "approach", rep, rep*2, rep*2+1) for rep in range(1, 6)]
        rows += [("step2", "screw", rep, (rep-1)*7+1, (rep-1)*7+7) for rep in range(1, 4)]
        for rep in range(1, 11):
            rows.extend([("step3", "insert", rep, rep*2, rep*2+.5),
                         ("step3", "deinsert", rep, rep*2+.8, rep*2+1.3)])
        for subject in ("s1", "s2", "s3"):
            make_annotation(cls.root / subject / "annotation" / "rgb_depth_gesture_segments_EN.csv", rows)
            for session in ("session1", "session2", "session3"):
                for view in ("front", "above"):
                    make_video(cls.root / subject / "rgbd" / session / view / "color.avi")
        cls.dataset = dict(dataset_root=str(cls.root), read_only=True, annotation_locale="EN")
        cls.config = dict(modality="rgb", views=["front", "above"], expected_class_count=4,
            session_ids=["step1", "step2", "step3"], input=dict(num_frames=4, image_size=32),
            screw_sampling=dict(duration_s=1, segments_per_trial=5),
            training=dict(seed=42, validation=dict(strategy="auto", fraction=.2, allow_train_validation_overlap=True)))
        cls.p1 = dict(name="protocol1", type="leave_one_subject_out", test_subject_id=None,
                      validation=None, split_before_windowing=True)
        cls.p2 = dict(name="protocol2", type="held_out_repetitions", train_repetitions=[1,2,3,4], test_repetitions=[5],
            validation=None, split_before_windowing=True,
            gesture_rules={"screw":dict(train_repetitions=[1,2],test_repetitions=[3]),
                           "insert":dict(train_repetitions=[1,2,3,4,6,7,8,9],test_repetitions=[5,10]),
                           "deinsert":dict(train_repetitions=[1,2,3,4,6,7,8,9],test_repetitions=[5,10])})

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_csv_cleanup_and_source_provenance(self):
        path = self.root / "s1/annotation/rgb_depth_gesture_segments_EN.csv"
        before = hashlib.sha256(path.read_bytes()).digest()
        records = read_rgbd_annotations(path, "s1")
        self.assertEqual(len(records), 28)
        self.assertEqual(records[0]["gesture_label"], "approach")
        self.assertEqual(records[0]["source_row"], 3)
        self.assertEqual(records[0]["intervals"]["above"]["start_s"], 2.1)
        self.assertIsNone(records[0]["intervals"]["left"]["start_s"])
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).digest())

    def test_repetition_split_keeps_parents_and_views_together(self):
        prepared = prepare_video_dataset(self.dataset, self.config, self.p2)
        fold = prepared["folds"][0]
        self.assertEqual(sum(len(items) for items in fold["splits"].values()), 120)
        self.assertEqual(len(fold["splits"]["test"]), 30)
        assigned = {}
        for part, clips in fold["splits"].items():
            for clip in clips:
                self.assertEqual(len(clip["views"]), 2)
                parent = clip["parent_record_id"]
                self.assertEqual(assigned.setdefault(parent, part), part)
                for view in clip["views"]:
                    self.assertFalse(Path(view["video_path"]).is_absolute())
                    self.assertEqual(len(view["frame_indices"]), 4)
                if part == "test":
                    allowed = [3] if clip["gesture_label"] == "screw" else [5,10] if clip["gesture_label"] in ("insert","deinsert") else [5]
                    self.assertIn(clip["repetition"], allowed)
        screw = [s for items in fold["splits"].values() for s in items if s["gesture_label"] == "screw" and s["subject_id"] == "s1"]
        self.assertEqual(len(screw), 15)  # synthetic subject has one object, not the real dataset's two
        first_parent = [s for s in screw if s["repetition"] == 1]
        first_parent.sort(key=lambda s:s["clip_index"])
        for a,b in zip(first_parent,first_parent[1:]):
            self.assertLessEqual(a["views"][0]["end_s"],b["views"][0]["start_s"])

    def test_subject_split_has_disjoint_subjects(self):
        prepared = prepare_video_dataset(self.dataset, self.config, self.p1)
        self.assertEqual(len(prepared["folds"]), 3)
        for fold in prepared["folds"]:
            sets = [{s["subject_id"] for s in fold["splits"][part]} for part in ("train","validation","test")]
            self.assertEqual([len(s) for s in sets], [1,1,1])
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_missing_view_annotations_fail_without_copying(self):
        config = dict(self.config, views=["left"])
        with self.assertRaises(VideoDatasetPreparationError) as error:
            prepare_video_dataset(self.dataset, config, self.p2)
        self.assertIn("invalid_video_interval", str(error.exception))

    def test_half_open_sampling_and_short_repetition(self):
        indices = sample_frame_indices(.15,.41,10,100,8)
        self.assertEqual(min(indices),2)
        self.assertEqual(max(indices),4)
        self.assertEqual(len(indices),8)
        self.assertEqual(sample_frame_indices(0,1,10,10,2), [2,7])
        with self.assertRaises(ValueError):
            sample_frame_indices(0,1.1,10,10,4)

    def test_source_path_cannot_escape_dataset_root(self):
        with self.assertRaises(ValueError):
            source_path(self.root,"../other/video.avi")

    def test_rgb_read_is_rgb_order_and_failure_is_explicit(self):
        path = "s1/rgbd/session1/front/color.avi"
        frames = read_rgb_frames(self.root,path,[0,5,5],16)
        self.assertEqual(frames.shape,(3,16,16,3))
        self.assertGreater(frames[0,:,:,0].mean(),frames[0,:,:,2].mean())
        np.testing.assert_array_equal(frames[1],frames[2])
        with self.assertRaises(ValueError):
            read_rgb_frames(self.root,path,[999999],16)

    def test_depth_uses_explicit_nonidentical_frame_map(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            root = Path(temporary)
            folder = root / "recording"
            make_video(folder/"color.avi",count=3,fps=10)
            (folder/"depth_raw").mkdir()
            for index,value in [(10,1000),(30,3000)]:
                cv2.imwrite(str(folder/"depth_raw"/("%06d.png"%index)),np.full((32,32),value,dtype=np.uint16))
            with (folder/"timestamps.csv").open("w",newline="") as stream:
                writer=csv.writer(stream); writer.writerow(["color_idx","depth_saved_idx","depth_saved"])
                writer.writerows([[1,10,1],[2,20,1],[3,30,1]])
            info=inspect_recording(folder,root,"depth")
            self.assertEqual(info["missing_depth_frame_count"],1)
            self.assertIsNone(info["depth_map"][1])
            array=read_depth_frames(root,[info["depth_map"][2],info["depth_map"][0]],16)
            self.assertEqual(array.dtype,np.uint16)
            self.assertEqual(array[:,0,0].tolist(),[3000,1000])

    def test_overlap_exception_preserves_identity_checks(self):
        records=read_rgbd_annotations(self.root/"s1/annotation/rgb_depth_gesture_segments_EN.csv","s1")
        config=deepcopy(self.config)
        # Ordinary repetitions 4 and 5 share an interval, deliberately.
        records[4]["intervals"]=deepcopy(records[3]["intervals"])
        with self.assertRaises(ValueError):
            _training_splits(records,config,self.p2)
        config["data_quality"]={"allow_within_subject_test_overlap":True}
        folds=_training_splits(records,config,self.p2)
        overlaps=[i for i in folds[0]["issues"] if i["code"]=="overlapping_intervals"]
        self.assertTrue(overlaps)
        self.assertTrue(all(i["severity"]=="warning" for i in overlaps))
        with self.assertRaises(ValueError):
            _training_splits(records+[deepcopy(records[0])],config,self.p2)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"Torch not installed in this inspection runtime")
    def test_lazy_rgb_tensor_dimensions(self):
        prepared=prepare_video_dataset(self.dataset,self.config,self.p2)
        loader=VideoGestureDataset(prepared["folds"][0]["splits"]["train"],self.root,self.config)
        tensor,label=loader[0]
        self.assertEqual(tuple(tensor.shape),(2,4,3,32,32))
        self.assertTrue(np.isfinite(tensor.numpy()).all())
        self.assertIsInstance(label,int)


if __name__ == "__main__":
    unittest.main()
