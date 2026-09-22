import copy
import unittest

from tracking.model_artifacts import validate_engine, validate_source, ENGINE_STRATEGY


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
        metadata["end2end"] = False
        del metadata["tracking"]["postprocess"]
        with self.assertRaises(ValueError):
            validate_source(metadata, "original")
