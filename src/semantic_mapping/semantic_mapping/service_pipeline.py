"""Non-blocking SAM2/RAM++ fan-out followed by bounded SigLIP2 batches."""

from concurrent.futures import Future
from dataclasses import dataclass

from ibrobot_msgs.srv import EncodeEmbeddings, GenerateMasks, RecognizeTags
from perception_service.model_contracts import MAX_MASK_BATCH


@dataclass(frozen=True)
class ServiceFrameResult:
    masks: object
    tags: tuple[str, ...]
    tag_scores: tuple[float, ...]
    embeddings: tuple
    model_diagnostics: dict[str, tuple]


def split_batches(items, batch_size: int = MAX_MASK_BATCH):
    if batch_size <= 0 or batch_size > MAX_MASK_BATCH:
        raise ValueError(f"batch size must be between 1 and {MAX_MASK_BATCH}")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


class ServiceFramePipeline:
    """Compose ROS futures; executor threads remain free and no callback spins."""

    def __init__(self, sam_client, ram_client, siglip_client, *, max_masks_per_batch: int = MAX_MASK_BATCH):
        if not 1 <= max_masks_per_batch <= MAX_MASK_BATCH:
            raise ValueError(f"max_masks_per_batch must be between 1 and {MAX_MASK_BATCH}")
        self.sam_client = sam_client
        self.ram_client = ram_client
        self.siglip_client = siglip_client
        self.max_masks_per_batch = max_masks_per_batch

    def process(self, image, *, mask_options: dict | None = None, score_threshold: float = 0.0) -> Future:
        mask_options = mask_options or {}
        sam_request = GenerateMasks.Request(image=image, **mask_options)
        ram_request = RecognizeTags.Request(image=image, score_threshold=score_threshold)
        sam_future = self.sam_client.call_async(sam_request)
        ram_future = self.ram_client.call_async(ram_request)
        result_future = Future()
        state = {"sam": None, "ram": None}

        def fail(exc):
            if not result_future.done():
                result_future.set_exception(exc)

        def stage_one_done(name, future):
            if result_future.done():
                return
            try:
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError(f"{name} service failed: {getattr(response, 'message', 'no response')}")
                state[name] = response
            except Exception as exc:
                fail(exc)
                return
            if state["sam"] is not None and state["ram"] is not None:
                self._start_siglip(image, state["sam"], state["ram"], result_future)

        sam_future.add_done_callback(lambda future: stage_one_done("sam", future))
        ram_future.add_done_callback(lambda future: stage_one_done("ram", future))
        return result_future

    def _start_siglip(self, image, sam_response, ram_response, result_future: Future) -> None:
        masks = [detection.mask for detection in sam_response.detections.detections]
        batches = split_batches(masks, self.max_masks_per_batch)
        if not batches:
            result_future.set_result(
                ServiceFrameResult(
                    masks=sam_response.detections,
                    tags=tuple(ram_response.tags),
                    tag_scores=tuple(ram_response.scores),
                    embeddings=(),
                    model_diagnostics={
                        "sam2": (sam_response.model,),
                        "ram_plus": (ram_response.model,),
                        "siglip2_image": (),
                    },
                )
            )
            return

        batch_futures = []
        for batch in batches:
            request = EncodeEmbeddings.Request(
                image=image,
                masks=batch,
                candidate_labels=list(ram_response.tags),
            )
            batch_futures.append(self.siglip_client.call_async(request))
        batch_results = [None] * len(batch_futures)

        def batch_done(index, future):
            if result_future.done():
                return
            try:
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError(f"siglip2 service failed: {getattr(response, 'message', 'no response')}")
                batch_results[index] = response
            except Exception as exc:
                result_future.set_exception(exc)
                return
            if all(response is not None for response in batch_results):
                embeddings = []
                offset = 0
                for response in batch_results:
                    for item in response.results:
                        item.mask_index += offset
                        embeddings.append(item)
                    offset += len(response.results)
                result_future.set_result(
                    ServiceFrameResult(
                        masks=sam_response.detections,
                        tags=tuple(ram_response.tags),
                        tag_scores=tuple(ram_response.scores),
                        embeddings=tuple(embeddings),
                        model_diagnostics={
                            "sam2": (sam_response.model,),
                            "ram_plus": (ram_response.model,),
                            "siglip2_image": tuple(response.model for response in batch_results),
                        },
                    )
                )

        for index, future in enumerate(batch_futures):
            future.add_done_callback(lambda future, index=index: batch_done(index, future))
