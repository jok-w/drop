import copy
import unittest

from tracking.model_artifacts import (
    validate_engine, validate_source, ENGINE_STRATEGY, FIXED_BATCH_ENGINE_STRATEGY,
)
from tracking.image_size import image_shape, size_label


class ArtifactTests(unittest.TestCase):
    def metadata(self):
        runtime = dict(tensorrt="10.3.0", cuda="12.6", gpu_name="test", gpu_capability=[8, 7], ultralytics="8.4.158")
        return dict(task="detect", names={0: "item"}, end2end=False, tracking=dict(
            schema=1, source_sha256="original", postprocess="external-nms", strategy=ENGINE_STRATEGY,
            max_batch=1, dynamic=False, image_size=1280, precision="fp16", runtime=runtime))

    def test_wrong_source_size_or_runtime_cannot_silently_run(self):
        metadata = self.metadata()
        runtime = metadata["tracking"]["runtime"]
        validate_engine(metadata, "original", 1280, runtime)
        for source, size, environment in [("other", 1280, runtime), ("original", 640, runtime),
                                           ("original", 1280, dict(runtime, tensorrt="10.4.0"))]:
            with self.subTest(source=source, size=size, environment=environment):
                with self.assertRaises(ValueError):
                    validate_engine(metadata, source, size, environment)

    def test_nms_free_export_is_rejected_for_external_nms_pipeline(self):
        metadata = self.metadata()
        metadata["end2end"] = True
        with self.assertRaises(ValueError):
            validate_source(metadata, "original")

    def test_fixed_batch_engine_must_match_requested_batch(self):
        metadata = self.metadata()
        metadata["tracking"].update(strategy=FIXED_BATCH_ENGINE_STRATEGY, max_batch=4)
        runtime = metadata["tracking"]["runtime"]
        validate_engine(metadata, "original", 1280, runtime, batch_size=4)
        for batch in (1, 2, 8):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                validate_engine(metadata, "original", 1280, runtime, batch_size=batch)
        with self.assertRaises(ValueError):
            validate_engine(metadata, "original", 1280, runtime, batch_size=0)
        metadata["end2end"] = False
        del metadata["tracking"]["postprocess"]
        with self.assertRaises(ValueError):
            validate_source(metadata, "original")

    def test_rectangular_shape_and_header_must_both_match(self):
        metadata = self.metadata()
        metadata["tracking"]["image_size"] = [736, 1280]
        metadata["imgsz"] = [736, 1280]
        runtime = metadata["tracking"]["runtime"]
        validate_engine(metadata, "original", (736, 1280), runtime)
        for size in (1280, [1280, 736]):
            with self.assertRaises(ValueError):
                validate_engine(metadata, "original", size, runtime)
        metadata["imgsz"] = [1280, 1280]
        with self.assertRaises(ValueError):
            validate_engine(metadata, "original", [736, 1280], runtime)

    def test_legacy_square_sizes_and_invalid_shapes(self):
        metadata = self.metadata()
        validate_engine(metadata, "original", [1280], metadata["tracking"]["runtime"])
        validate_engine(metadata, "original", [1280, 1280], metadata["tracking"]["runtime"])
        self.assertEqual(size_label([1280]), "1280")
        self.assertEqual(size_label([736, 1280]), "736x1280")
        for invalid in (None, [], [1, 2, 3], 0, -32, [720, 1280], [736, 0], 1280.0):
            with self.subTest(size=invalid), self.assertRaises(ValueError):
                image_shape(invalid)
